///
/// @file src/nc4gds.cu
/// @brief parameter study related with VFD for GDS using NetCDF-4
///
/// NetCDF-4 uses HDF5 as storage engine. We configure HDF5 to use GDS VFD
/// via environment variable, then use pure NetCDF-C API.
///
/// @copyright Copyright (c) 2023 Information Technology Center, The University of Tokyo
///
/// The MIT License is applied to this software, see LICENSE
///
#include <curand_mtgp32.h>  // THREAD_NUM
#include <fcntl.h>          // open, O_RDONLY
#include <helper_cuda.h>    // checkCudaErrors
#include <netcdf.h>
#include <unistd.h>  // fsync, close

#include <boost/filesystem.hpp>            // boost::filesystem
#include <boost/lexical_cast.hpp>          // boost::lexical_cast
#include <boost/program_options.hpp>       // boost::program_options
#include <boost/uuid/uuid_generators.hpp>  // boost::uuids::random_generator
#include <boost/uuid/uuid_io.hpp>          // convert boost::uuids::uuid to std::string
#include <algorithm>                       // std::min_element, std::max_element
#include <cstdlib>                         // std::exit, posix_memalign
#include <cstring>                         // strerror
#include <fstream>                         // std::ofstream
#include <numeric>                         // std::accumulate
#include <iostream>                        // std::cout
#include <string>                          // std::string
#include <vector>                          // std::vector

#include "allocate.cuh"
#include "common.cuh"
#include "generate.cuh"
#include "netcdf4.hpp"

static constexpr float newton = 1.0F;  // gravitational constant in the computational unit

// Utility function for rounding up to nearest multiple
constexpr auto round_up(const size_t org, const size_t unit) {
  const size_t mod = org % unit;
  return ((mod == 0) ? org : (org + unit - mod));
}

///
/// @brief main function
///
/// @param[in] argc number of input argument(s)
/// @param[in] argv input argument(s)
///
auto main(const int32_t argc, const char* const* const argv) -> int32_t {
  // use scientific notation for floating-point number
  std::cout << std::scientific;

  // initialize the simulation
  // prepare options
  boost::program_options::options_description opt("List of options");
  opt.add_options()(
      "num", boost::program_options::value<type::idx>()->default_value(1024), "number of particles")(
      "vfd", boost::program_options::value<std::string>()->default_value("gds"), "VFD driver to use: sec2, gds, or direct")(
      "skip", boost::program_options::bool_switch()->default_value(false), "skip consistency check between read and original data")(
      "virial", boost::program_options::value<float>()->default_value(0.2), "Virial ratio of the system")(
      "radius", boost::program_options::value<float>()->default_value(1.0), "radius of the system")(
      "mass", boost::program_options::value<float>()->default_value(1.0), "total mass of the system")(
      "runs", boost::program_options::value<size_t>()->default_value(3), "number of benchmark runs for min/max/avg")(
      "warmup-runs", boost::program_options::value<size_t>()->default_value(0), "number of warm-up I/O runs (discarded, no timing)")(
      "align", boost::program_options::bool_switch()->default_value(false), "use 4KB page-aligned allocations (malloc/first-touch and host buffers)")(
      "help,h", "Help");
  // read input arguments
  boost::program_options::variables_map vm;
  boost::program_options::store(boost::program_options::parse_command_line(argc, argv, opt), vm);
  boost::program_options::notify(vm);
  if (vm.count("help") == 1UL) {
    std::cout << opt << std::endl;
    std::exit(EXIT_SUCCESS);
  }
  // configure the benchmark
  const auto num = vm["num"].as<type::idx>();
  const auto vfd_name = vm["vfd"].as<std::string>();
  const auto virial = vm["virial"].as<float>();
  const auto radius = vm["radius"].as<float>();
  const auto mass = vm["mass"].as<float>();
  const auto skip = vm["skip"].as<bool>();
  const auto num_runs = vm["runs"].as<size_t>();
  const auto warmup_runs = vm["warmup-runs"].as<size_t>();
  const auto use_align = vm["align"].as<bool>();
  vm.clear();
  if (num_runs < 1UL) {
    std::cerr << "runs must be >= 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // VFD is configured externally via HDF5_DRIVER environment variable
  std::cout << "Using HDF5 VFD from HDF5_DRIVER environment variable for NetCDF-4" << std::endl;
  if (use_align) {
    std::cout << "Using 4KB page-aligned allocations" << std::endl;
  }

  // memory allocation - NetCDF-compatible layout (Nx3 position, Nx3 velocity, N mass, N id)
  cudaSetDevice(0);
  float* position = nullptr;
  float* velocity = nullptr;
  float* mass_buf = nullptr;
  type::idx* id = nullptr;
  allocate_particles_netcdf(&position, &velocity, &mass_buf, &id, num, use_align);

  // Generate initial data directly in NetCDF-compatible layout
  set_uniform_sphere_netcdf(num, position, velocity, mass_buf, id, mass, radius, virial, newton);

  // Host buffers for non-first-touch mode (cudaMalloc case needs explicit copy)
  float* position_host = nullptr;
  float* velocity_host = nullptr;
  float* mass_host = nullptr;
  type::idx* id_host = nullptr;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // Allocate host staging buffers for NetCDF I/O (NetCDF cannot write from GPU memory)
  constexpr size_t PAGE_SIZE = 4096;
  if (use_align) {
    if (posix_memalign((void**)&position_host, PAGE_SIZE, num * 3 * sizeof(float)) != 0 ||
        posix_memalign((void**)&velocity_host, PAGE_SIZE, num * 3 * sizeof(float)) != 0 ||
        posix_memalign((void**)&mass_host, PAGE_SIZE, num * sizeof(float)) != 0 ||
        posix_memalign((void**)&id_host, PAGE_SIZE, num * sizeof(type::idx)) != 0) {
      std::cerr << "Failed to allocate page-aligned host staging buffers" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    position_host = (float*)malloc(num * 3 * sizeof(float));
    velocity_host = (float*)malloc(num * 3 * sizeof(float));
    mass_host = (float*)malloc(num * sizeof(float));
    id_host = (type::idx*)malloc(num * sizeof(type::idx));
  }

  if (!position_host || !velocity_host || !mass_host || !id_host) {
    std::cerr << "Failed to allocate host staging buffers" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif

  // Pointers for write operations
  auto* position_write = position;
  auto* velocity_write = velocity;
  auto* mass_write = mass_buf;
  auto* id_write = id;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  position_write = position_host;
  velocity_write = velocity_host;
  mass_write = mass_host;
  id_write = id_host;
#endif

  constexpr auto benchmark = [](const auto func) noexcept(false) {
    struct timespec ini;
    clock_gettime(CLOCK_MONOTONIC, &ini);
    func();
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (std::fma(1.0e-9, static_cast<double>(end.tv_nsec - ini.tv_nsec), end.tv_sec - ini.tv_sec));
  };

  //
  // Setup and allocate read buffers (reused across runs)
  //
  boost::filesystem::path dat_dir("./dat");
  if (!boost::filesystem::exists(dat_dir)) {
    boost::filesystem::create_directories(dat_dir);
  }

  std::vector<double> write_times;
  std::vector<double> read_times;
  std::vector<std::string> file_names;
  write_times.reserve(num_runs);
  read_times.reserve(num_runs);
  file_names.reserve(num_runs);

  float* position_read = nullptr;
  float* velocity_read = nullptr;
  float* mass_read = nullptr;
  type::idx* id_read = nullptr;
  allocate_particles_netcdf(&position_read, &velocity_read, &mass_read, &id_read, num, use_align);

  float* position_read_host = nullptr;
  float* velocity_read_host = nullptr;
  float* mass_read_host = nullptr;
  type::idx* id_read_host = nullptr;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  constexpr size_t PAGE_SIZE_READ = 4096;
  if (use_align) {
    if (posix_memalign((void**)&position_read_host, PAGE_SIZE_READ, num * 3 * sizeof(float)) != 0 ||
        posix_memalign((void**)&velocity_read_host, PAGE_SIZE_READ, num * 3 * sizeof(float)) != 0 ||
        posix_memalign((void**)&mass_read_host, PAGE_SIZE_READ, num * sizeof(float)) != 0 ||
        posix_memalign((void**)&id_read_host, PAGE_SIZE_READ, num * sizeof(type::idx)) != 0) {
      std::cerr << "Failed to allocate page-aligned host read buffers" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    position_read_host = (float*)malloc(num * 3 * sizeof(float));
    velocity_read_host = (float*)malloc(num * 3 * sizeof(float));
    mass_read_host = (float*)malloc(num * sizeof(float));
    id_read_host = (type::idx*)malloc(num * sizeof(type::idx));
  }
#endif

  auto* position_read_ptr = position_read;
  auto* velocity_read_ptr = velocity_read;
  auto* mass_read_ptr = mass_read;
  auto* id_read_ptr = id_read;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  position_read_ptr = position_read_host;
  velocity_read_ptr = velocity_read_host;
  mass_read_ptr = mass_read_host;
  id_read_ptr = id_read_host;
#endif

  bool success = true;

  //
  // Warm-up runs (discard timings)
  //
  for (size_t w = 0UL; w < warmup_runs; w++) {
    const auto filename_w = (dat_dir / (boost::lexical_cast<std::string>(boost::uuids::random_generator()()) + ".nc")).string();

    int ncid_w;
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
    checkCudaErrors(cudaMemcpy(position_host, position, num * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(velocity_host, velocity, num * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mass_host, mass_buf, num * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(id_host, id, num * sizeof(type::idx), cudaMemcpyDeviceToHost));
#endif
    NC_CHECK(nc_create(filename_w.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid_w));
    int dim_n_w, dim_3_w;
    NC_CHECK(nc_def_dim(ncid_w, "num_particles", num, &dim_n_w));
    NC_CHECK(nc_def_dim(ncid_w, "coord", 3, &dim_3_w));
    int dims_n3_w[2] = {dim_n_w, dim_3_w};
    int var_pos_w, var_vel_w, var_mass_w, var_id_w;
    NC_CHECK(nc_def_var(ncid_w, "position", NC_FLOAT, 2, dims_n3_w, &var_pos_w));
    NC_CHECK(nc_def_var(ncid_w, "velocity", NC_FLOAT, 2, dims_n3_w, &var_vel_w));
    NC_CHECK(nc_def_var(ncid_w, "mass", NC_FLOAT, 1, &dim_n_w, &var_mass_w));
    NC_CHECK(nc_def_var(ncid_w, "id", NC_UINT64, 1, &dim_n_w, &var_id_w));
    unsigned long long num_ull_w = static_cast<unsigned long long>(num);
    unsigned long long id_ull_w = static_cast<unsigned long long>(num);
    NC_CHECK(nc_put_att_ulonglong(ncid_w, NC_GLOBAL, "num", NC_UINT64, 1, &num_ull_w));
    NC_CHECK(nc_put_att_ulonglong(ncid_w, NC_GLOBAL, "id", NC_UINT64, 1, &id_ull_w));
    NC_CHECK(nc_put_var_float(ncid_w, var_pos_w, position_write));
    NC_CHECK(nc_put_var_float(ncid_w, var_vel_w, velocity_write));
    NC_CHECK(nc_put_var_float(ncid_w, var_mass_w, mass_write));
    NC_CHECK(nc_put_var_ulonglong(ncid_w, var_id_w, reinterpret_cast<const unsigned long long*>(id_write)));
    NC_CHECK(nc_close(ncid_w));

    const int fd_w = open(filename_w.c_str(), O_RDONLY);
    if (fd_w != -1) {
      fsync(fd_w);
      posix_fadvise(fd_w, 0, 0, POSIX_FADV_DONTNEED);
      close(fd_w);
    }

    int ncid_r;
    NC_CHECK(nc_open(filename_w.c_str(), NC_NOWRITE, &ncid_r));
    int var_pos_r, var_vel_r, var_mass_r, var_id_r;
    NC_CHECK(nc_inq_varid(ncid_r, "position", &var_pos_r));
    NC_CHECK(nc_inq_varid(ncid_r, "velocity", &var_vel_r));
    NC_CHECK(nc_inq_varid(ncid_r, "mass", &var_mass_r));
    NC_CHECK(nc_inq_varid(ncid_r, "id", &var_id_r));
    NC_CHECK(nc_get_var_float(ncid_r, var_pos_r, position_read_ptr));
    NC_CHECK(nc_get_var_float(ncid_r, var_vel_r, velocity_read_ptr));
    NC_CHECK(nc_get_var_float(ncid_r, var_mass_r, mass_read_ptr));
    NC_CHECK(nc_get_var_ulonglong(ncid_r, var_id_r, reinterpret_cast<unsigned long long*>(id_read_ptr)));
    NC_CHECK(nc_close(ncid_r));
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
    checkCudaErrors(cudaMemcpy(position_read, position_read_host, num * 3 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(velocity_read, velocity_read_host, num * 3 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mass_read, mass_read_host, num * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(id_read, id_read_host, num * sizeof(type::idx), cudaMemcpyHostToDevice));
#endif

    boost::filesystem::remove(filename_w);
  }

  //
  // Phase 1: write loop (create file, write, nc_close, fsync, posix_fadvise for each run)
  //
  for (size_t run = 0UL; run < num_runs; run++) {
    const auto filename = (dat_dir / (boost::lexical_cast<std::string>(boost::uuids::random_generator()()) + ".nc")).string();
    file_names.push_back(filename);

    int ncid;
    const auto elapse_write = benchmark([&]() {
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
      checkCudaErrors(cudaMemcpy(position_host, position, num * 3 * sizeof(float), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(velocity_host, velocity, num * 3 * sizeof(float), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(mass_host, mass_buf, num * sizeof(float), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(id_host, id, num * sizeof(type::idx), cudaMemcpyDeviceToHost));
#endif

      NC_CHECK(nc_create(filename.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid));

      int dim_n, dim_3;
      NC_CHECK(nc_def_dim(ncid, "num_particles", num, &dim_n));
      NC_CHECK(nc_def_dim(ncid, "coord", 3, &dim_3));
      int dims_n3[2] = {dim_n, dim_3};

      int var_pos, var_vel, var_mass, var_id;
      NC_CHECK(nc_def_var(ncid, "position", NC_FLOAT, 2, dims_n3, &var_pos));
      NC_CHECK(nc_def_var(ncid, "velocity", NC_FLOAT, 2, dims_n3, &var_vel));
      NC_CHECK(nc_def_var(ncid, "mass", NC_FLOAT, 1, &dim_n, &var_mass));
      NC_CHECK(nc_def_var(ncid, "id", NC_UINT64, 1, &dim_n, &var_id));

      unsigned long long num_ull = static_cast<unsigned long long>(num);
      unsigned long long id_ull = static_cast<unsigned long long>(num);
      NC_CHECK(nc_put_att_ulonglong(ncid, NC_GLOBAL, "num", NC_UINT64, 1, &num_ull));
      NC_CHECK(nc_put_att_ulonglong(ncid, NC_GLOBAL, "id", NC_UINT64, 1, &id_ull));

      NC_CHECK(nc_put_var_float(ncid, var_pos, position_write));
      NC_CHECK(nc_put_var_float(ncid, var_vel, velocity_write));
      NC_CHECK(nc_put_var_float(ncid, var_mass, mass_write));
      NC_CHECK(nc_put_var_ulonglong(ncid, var_id, reinterpret_cast<const unsigned long long*>(id_write)));
    });
    NC_CHECK(nc_close(ncid));
    write_times.push_back(elapse_write);

    const int fd = open(filename.c_str(), O_RDONLY);
    if (fd != -1) {
      fsync(fd);
      int ret = posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
      if (ret != 0) {
        std::cerr << "Warning: posix_fadvise failed: " << strerror(ret) << std::endl;
      }
      close(fd);
    } else {
      std::cerr << "Warning: Failed to open file for cache drop: " << filename << std::endl;
    }
  }

  // Ensure all writes are fully persisted before reads (cold read)
  sleep(5);

  //
  // Phase 2: read loop (open, read, nc_close, validate for each run)
  //
  for (size_t run = 0UL; run < num_runs; run++) {
    const auto& filename = file_names[run];

    const auto elapse_read = benchmark([&]() {
      int ncid;
      NC_CHECK(nc_open(filename.c_str(), NC_NOWRITE, &ncid));

      int var_pos, var_vel, var_mass, var_id;
      NC_CHECK(nc_inq_varid(ncid, "position", &var_pos));
      NC_CHECK(nc_inq_varid(ncid, "velocity", &var_vel));
      NC_CHECK(nc_inq_varid(ncid, "mass", &var_mass));
      NC_CHECK(nc_inq_varid(ncid, "id", &var_id));

      unsigned long long num_read_attr;
      NC_CHECK(nc_get_att_ulonglong(ncid, NC_GLOBAL, "num", &num_read_attr));

      NC_CHECK(nc_get_var_float(ncid, var_pos, position_read_ptr));
      NC_CHECK(nc_get_var_float(ncid, var_vel, velocity_read_ptr));
      NC_CHECK(nc_get_var_float(ncid, var_mass, mass_read_ptr));
      NC_CHECK(nc_get_var_ulonglong(ncid, var_id, reinterpret_cast<unsigned long long*>(id_read_ptr)));

      NC_CHECK(nc_close(ncid));

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
      checkCudaErrors(cudaMemcpy(position_read, position_read_host, num * 3 * sizeof(float), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(velocity_read, velocity_read_host, num * 3 * sizeof(float), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(mass_read, mass_read_host, num * sizeof(float), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(id_read, id_read_host, num * sizeof(type::idx), cudaMemcpyHostToDevice));
#endif
    });
    read_times.push_back(elapse_read);

    if (!skip) {
      auto* pos_verify = position_write;
      auto* vel_verify = velocity_write;
      auto* mass_verify = mass_write;
      auto* id_verify = id_write;
      auto* pos_read_verify = position_read_ptr;
      auto* vel_read_verify = velocity_read_ptr;
      auto* mass_read_verify = mass_read_ptr;
      auto* id_read_verify = id_read_ptr;

      for (size_t i = 0; i < num * 3 && success; i++) {
        if (pos_verify[i] != pos_read_verify[i]) {
          std::cerr << "run " << run << ": position mismatch at " << i << std::endl;
          success = false;
          std::exit(EXIT_FAILURE);
        }
      }
      for (size_t i = 0; i < num * 3 && success; i++) {
        if (vel_verify[i] != vel_read_verify[i]) {
          std::cerr << "run " << run << ": velocity mismatch at " << i << std::endl;
          success = false;
          std::exit(EXIT_FAILURE);
        }
      }
      for (size_t i = 0; i < num && success; i++) {
        if (mass_verify[i] != mass_read_verify[i]) {
          std::cerr << "run " << run << ": mass mismatch at " << i << std::endl;
          success = false;
          std::exit(EXIT_FAILURE);
        }
      }
      for (size_t i = 0; i < num && success; i++) {
        if (id_verify[i] != id_read_verify[i]) {
          std::cerr << "run " << run << ": id mismatch at " << i << std::endl;
          success = false;
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }

  if (success && !skip) {
    std::cout << "Data verification: PASSED" << std::endl;
  }

  //
  // BENCHMARK RESULTS
  //
  const auto datasize = static_cast<double>(num) * static_cast<double>(3 * sizeof(float) + 3 * sizeof(float) + sizeof(float) + sizeof(type::idx));

  const auto elapse_write_min = *std::min_element(write_times.begin(), write_times.end());
  const auto elapse_write_max = *std::max_element(write_times.begin(), write_times.end());
  const auto elapse_write_avg = std::accumulate(write_times.begin(), write_times.end(), 0.0) / static_cast<double>(num_runs);
  const auto elapse_read_min = *std::min_element(read_times.begin(), read_times.end());
  const auto elapse_read_max = *std::max_element(read_times.begin(), read_times.end());
  const auto elapse_read_avg = std::accumulate(read_times.begin(), read_times.end(), 0.0) / static_cast<double>(num_runs);

  std::cout << "=== NetCDF-4 GDS Benchmark Results ===" << std::endl;
  std::cout << "VFD: " << vfd_name << " (via HDF5_DRIVER env)" << std::endl;
  std::cout << "Particles: " << num << std::endl;
  std::cout << "Runs: " << num_runs << std::endl;
  std::cout << "Data size: " << datasize << " bytes" << std::endl;
  std::cout << "Write time (min/max/avg): " << elapse_write_min << " / " << elapse_write_max << " / " << elapse_write_avg << " s" << std::endl;
  std::cout << "Read time (min/max/avg): " << elapse_read_min << " / " << elapse_read_max << " / " << elapse_read_avg << " s" << std::endl;
  std::cout << "Verification: " << (success ? "PASSED" : "FAILED") << std::endl;

  //
  // WRITE CSV (matching h5gds.cu format)
  //
  boost::filesystem::path log_dir("./log");
  if (!boost::filesystem::exists(log_dir)) {
    boost::filesystem::create_directories(log_dir);
  }
  const std::string report = (log_dir / "nc4gds_benchmark.csv").string();
  const boost::filesystem::path previous(report);
  boost::system::error_code err;
  const auto exist = boost::filesystem::exists(previous, err);

  std::ofstream output(report, std::ios::app);
  if (!exist || err) {
    output << "VFD";
    output << ",skip";
    output << ",N";
    output << ",runs";
    output << ",data size [byte]";
    output << ",latency (write) min [s]";
    output << ",latency (write) max [s]";
    output << ",latency (write) avg [s]";
    output << ",latency (read) min [s]";
    output << ",latency (read) max [s]";
    output << ",latency (read) avg [s]";
    output << ",bandwidth (write) min [byte/s]";
    output << ",bandwidth (write) max [byte/s]";
    output << ",bandwidth (write) avg [byte/s]";
    output << ",bandwidth (read) min [byte/s]";
    output << ",bandwidth (read) max [byte/s]";
    output << ",bandwidth (read) avg [byte/s]";
    output << ",filename";
    output << std::endl;
  }

  output << std::scientific;
  output << vfd_name;
  output << "," << (skip ? "true" : "false");
  output << "," << num;
  output << "," << num_runs;
  output << "," << datasize;
  output << "," << elapse_write_min;
  output << "," << elapse_write_max;
  output << "," << elapse_write_avg;
  output << "," << elapse_read_min;
  output << "," << elapse_read_max;
  output << "," << elapse_read_avg;
  output << "," << datasize / elapse_write_max;
  output << "," << datasize / elapse_write_min;
  output << "," << datasize / elapse_write_avg;
  output << "," << datasize / elapse_read_max;
  output << "," << datasize / elapse_read_min;
  output << "," << datasize / elapse_read_avg;
  output << "," << (file_names.empty() ? "" : file_names.front());
  output << std::endl;
  output.close();

  // Cleanup - remove all test files
  for (const auto& fname : file_names) {
    boost::filesystem::remove(fname);
  }

  // Release memory
  release_particles_netcdf(position, velocity, mass_buf, id);
  release_particles_netcdf(position_read, velocity_read, mass_read, id_read);

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  free(position_host);
  free(velocity_host);
  free(mass_host);
  free(id_host);
  free(position_read_host);
  free(velocity_read_host);
  free(mass_read_host);
  free(id_read_host);
#endif

  return (success ? EXIT_SUCCESS : EXIT_FAILURE);
}
