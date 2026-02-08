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
#include <helper_cuda.h>    // checkCudaErrors
#include <netcdf.h>

#include <boost/filesystem.hpp>            // boost::filesystem
#include <boost/lexical_cast.hpp>          // boost::lexical_cast
#include <boost/program_options.hpp>       // boost::program_options
#include <boost/uuid/uuid_generators.hpp>  // boost::uuids::random_generator
#include <boost/uuid/uuid_io.hpp>          // convert boost::uuids::uuid to std::string
#include <cstdlib>                         // std::exit
#include <fstream>                         // std::ofstream
#include <iostream>                        // std::cout
#include <string>                          // std::string

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
  vm.clear();

  // VFD is configured externally via HDF5_DRIVER environment variable
  std::cout << "Using HDF5 VFD from HDF5_DRIVER environment variable for NetCDF-4" << std::endl;

  // memory allocation - NetCDF-compatible layout (Nx3 position, Nx3 velocity, N mass, N id)
  cudaSetDevice(0);
  float* position = nullptr;
  float* velocity = nullptr;
  float* mass_buf = nullptr;
  type::idx* id = nullptr;
  allocate_particles_netcdf(&position, &velocity, &mass_buf, &id, num);

  // Generate initial data directly in NetCDF-compatible layout
  set_uniform_sphere_netcdf(num, position, velocity, mass_buf, id, mass, radius, virial, newton);

  // Host buffers for non-first-touch mode (cudaMalloc case needs explicit copy)
  float* position_host = nullptr;
  float* velocity_host = nullptr;
  float* mass_host = nullptr;
  type::idx* id_host = nullptr;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // Allocate host staging buffers for NetCDF I/O (NetCDF cannot write from GPU memory)
  position_host = (float*)malloc(num * 3 * sizeof(float));
  velocity_host = (float*)malloc(num * 3 * sizeof(float));
  mass_host = (float*)malloc(num * sizeof(float));
  id_host = (type::idx*)malloc(num * sizeof(type::idx));

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
  // BENCHMARK: NetCDF-4 WRITE
  //
  boost::filesystem::path dat_dir("./dat");
  if (!boost::filesystem::exists(dat_dir)) {
    boost::filesystem::create_directories(dat_dir);
  }
  const auto filename = (dat_dir / (boost::lexical_cast<std::string>(boost::uuids::random_generator()()) + ".nc")).string();

  const auto elapse_write = benchmark([&]() {
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
    // Copy GPU data to host buffers (INSIDE timing, consistent with h5gds.cu)
    checkCudaErrors(cudaMemcpy(position_host, position, num * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(velocity_host, velocity, num * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(mass_host, mass_buf, num * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(id_host, id, num * sizeof(type::idx), cudaMemcpyDeviceToHost));
#endif

    int ncid;
    NC_CHECK(nc_create(filename.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid));

    // Define dimensions
    int dim_n, dim_3;
    NC_CHECK(nc_def_dim(ncid, "num_particles", num, &dim_n));
    NC_CHECK(nc_def_dim(ncid, "coord", 3, &dim_3));
    int dims_n3[2] = {dim_n, dim_3};

    // Define variables
    int var_pos, var_vel, var_mass, var_id;
    NC_CHECK(nc_def_var(ncid, "position", NC_FLOAT, 2, dims_n3, &var_pos));
    NC_CHECK(nc_def_var(ncid, "velocity", NC_FLOAT, 2, dims_n3, &var_vel));
    NC_CHECK(nc_def_var(ncid, "mass", NC_FLOAT, 1, &dim_n, &var_mass));
    NC_CHECK(nc_def_var(ncid, "id", NC_UINT64, 1, &dim_n, &var_id));

    // Store metadata as global attributes
    unsigned long long num_ull = static_cast<unsigned long long>(num);
    unsigned long long id_ull = static_cast<unsigned long long>(num);
    NC_CHECK(nc_put_att_ulonglong(ncid, NC_GLOBAL, "num", NC_UINT64, 1, &num_ull));
    NC_CHECK(nc_put_att_ulonglong(ncid, NC_GLOBAL, "id", NC_UINT64, 1, &id_ull));

    // Write data
    NC_CHECK(nc_put_var_float(ncid, var_pos, position_write));
    NC_CHECK(nc_put_var_float(ncid, var_vel, velocity_write));
    NC_CHECK(nc_put_var_float(ncid, var_mass, mass_write));
    NC_CHECK(nc_put_var_ulonglong(ncid, var_id, reinterpret_cast<const unsigned long long*>(id_write)));

    NC_CHECK(nc_close(ncid));
  });

  //
  // BENCHMARK: NetCDF-4 READ
  //

  // Allocate read buffers
  float* position_read = nullptr;
  float* velocity_read = nullptr;
  float* mass_read = nullptr;
  type::idx* id_read = nullptr;
  allocate_particles_netcdf(&position_read, &velocity_read, &mass_read, &id_read, num);

  // Host buffers for read (non-first-touch mode)
  float* position_read_host = nullptr;
  float* velocity_read_host = nullptr;
  float* mass_read_host = nullptr;
  type::idx* id_read_host = nullptr;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  position_read_host = (float*)malloc(num * 3 * sizeof(float));
  velocity_read_host = (float*)malloc(num * 3 * sizeof(float));
  mass_read_host = (float*)malloc(num * sizeof(float));
  id_read_host = (type::idx*)malloc(num * sizeof(type::idx));
#endif

  // Pointers for read operations
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

  const auto elapse_read = benchmark([&]() {
    int ncid;
    NC_CHECK(nc_open(filename.c_str(), NC_NOWRITE, &ncid));

    // Get variable IDs
    int var_pos, var_vel, var_mass, var_id;
    NC_CHECK(nc_inq_varid(ncid, "position", &var_pos));
    NC_CHECK(nc_inq_varid(ncid, "velocity", &var_vel));
    NC_CHECK(nc_inq_varid(ncid, "mass", &var_mass));
    NC_CHECK(nc_inq_varid(ncid, "id", &var_id));

    // Read attributes for verification
    unsigned long long num_read_attr;
    NC_CHECK(nc_get_att_ulonglong(ncid, NC_GLOBAL, "num", &num_read_attr));

    // Read data
    NC_CHECK(nc_get_var_float(ncid, var_pos, position_read_ptr));
    NC_CHECK(nc_get_var_float(ncid, var_vel, velocity_read_ptr));
    NC_CHECK(nc_get_var_float(ncid, var_mass, mass_read_ptr));
    NC_CHECK(nc_get_var_ulonglong(ncid, var_id, reinterpret_cast<unsigned long long*>(id_read_ptr)));

    NC_CHECK(nc_close(ncid));

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
    // Copy host data back to GPU (INSIDE timing, consistent with h5gds.cu)
    checkCudaErrors(cudaMemcpy(position_read, position_read_host, num * 3 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(velocity_read, velocity_read_host, num * 3 * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(mass_read, mass_read_host, num * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(id_read, id_read_host, num * sizeof(type::idx), cudaMemcpyHostToDevice));
#endif
  });

  //
  // VERIFICATION (compare on host)
  //
  bool success = true;
  if (!skip) {
    // For verification, use the host pointers
    auto* pos_verify = position_write;
    auto* vel_verify = velocity_write;
    auto* mass_verify = mass_write;
    auto* id_verify = id_write;
    auto* pos_read_verify = position_read_ptr;
    auto* vel_read_verify = velocity_read_ptr;
    auto* mass_read_verify = mass_read_ptr;
    auto* id_read_verify = id_read_ptr;

    // Compare position
    for (size_t i = 0; i < num * 3 && success; i++) {
      if (pos_verify[i] != pos_read_verify[i]) {
        std::cerr << "Position mismatch at index " << i << ": " << pos_verify[i] << " vs " << pos_read_verify[i] << std::endl;
        success = false;
      }
    }
    // Compare velocity
    for (size_t i = 0; i < num * 3 && success; i++) {
      if (vel_verify[i] != vel_read_verify[i]) {
        std::cerr << "Velocity mismatch at index " << i << ": " << vel_verify[i] << " vs " << vel_read_verify[i] << std::endl;
        success = false;
      }
    }
    // Compare mass
    for (size_t i = 0; i < num && success; i++) {
      if (mass_verify[i] != mass_read_verify[i]) {
        std::cerr << "Mass mismatch at index " << i << ": " << mass_verify[i] << " vs " << mass_read_verify[i] << std::endl;
        success = false;
      }
    }
    // Compare id
    for (size_t i = 0; i < num && success; i++) {
      if (id_verify[i] != id_read_verify[i]) {
        std::cerr << "ID mismatch at index " << i << ": " << id_verify[i] << " vs " << id_read_verify[i] << std::endl;
        success = false;
      }
    }
    if (success) {
      std::cout << "Data verification: PASSED" << std::endl;
    }
  }

  //
  // BENCHMARK RESULTS
  //
  const size_t file_bytes = num * (3 * sizeof(float) + 3 * sizeof(float) + sizeof(float) + sizeof(type::idx));
  const double file_size_MB = static_cast<double>(file_bytes) / (1024.0 * 1024.0);
  const double write_bw = file_size_MB / elapse_write;
  const double read_bw = file_size_MB / elapse_read;

  std::cout << "=== NetCDF-4 GDS Benchmark Results ===" << std::endl;
  std::cout << "VFD: " << vfd_name << " (via HDF5_DRIVER env)" << std::endl;
  std::cout << "Particles: " << num << std::endl;
  std::cout << "File size: " << file_size_MB << " MB" << std::endl;
  std::cout << "Write time: " << elapse_write << " s (" << write_bw << " MB/s)" << std::endl;
  std::cout << "Read time: " << elapse_read << " s (" << read_bw << " MB/s)" << std::endl;
  std::cout << "Verification: " << (success ? "PASSED" : "FAILED") << std::endl;

  //
  // WRITE CSV
  //
  boost::filesystem::path log_dir("./log");
  if (!boost::filesystem::exists(log_dir)) {
    boost::filesystem::create_directories(log_dir);
  }
  const auto csv_file = (log_dir / "nc4gds_benchmark.csv").string();
  bool write_header = !boost::filesystem::exists(csv_file);
  std::ofstream csv(csv_file, std::ios::app);
  if (write_header) {
    csv << "vfd,num,file_bytes,write_s,read_s,write_MBps,read_MBps,verified" << std::endl;
  }
  csv << vfd_name << "," << num << "," << file_bytes << ","
      << elapse_write << "," << elapse_read << ","
      << write_bw << "," << read_bw << ","
      << (success ? "true" : "false") << std::endl;
  csv.close();

  // Cleanup - remove test file
  boost::filesystem::remove(filename);

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
