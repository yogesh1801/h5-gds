///
/// @file src/h5gds.cu
/// @author Yohei MIKI (The University of Tokyo)
/// @brief parameter study related with VFD for GDS
///
/// @copyright Copyright (c) 2023 Information Technology Center, The University of Tokyo
///
/// The MIT License is applied to this software, see LICENSE
///
#include <H5FDgds.h>  // VFD for GDS
#include <hdf5.h>
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <boost/filesystem.hpp>            // boost::filesystem
#include <boost/lexical_cast.hpp>          // boost::lexical_cast
#include <boost/program_options.hpp>       // boost::program_options
#include <boost/uuid/uuid_generators.hpp>  // boost::uuids::random_generator
#include <boost/uuid/uuid_io.hpp>          // convert boost::uuids::uuid to std::string
#include <algorithm>                       // std::min_element, std::max_element
#include <cstdlib>                         // std::exit
#include <fcntl.h>                         // open(), O_RDONLY
#include <fstream>                         // std::ofstream
#include <iomanip>                         // std::setw
#include <iostream>                        // std::cout
#include <mutex>                           // std::mutex
#include <numeric>                         // std::accumulate
#include <sstream>                         // std::stringstream
#include <string>                          // std::string
#include <thread>                          // std::thread
#include <unistd.h>                        // close(), posix_fadvise()
#include <vector>                          // std::vector

#include "allocate.cuh"
#include "common.cuh"
#include "generate.cuh"
#include "hdf5.hpp"

static constexpr type::vel_z newton = 1.0F;  // gravitational constant in the computational unit

struct compare_pos {
  __host__ __device__ bool operator()(type::pos a, type::pos b) const {
    return ((a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w));
  }
};
struct compare_vel_xy {
  __host__ __device__ bool operator()(type::vel_xy a, type::vel_xy b) const {
    return ((a.x == b.x) && (a.y == b.y));
  }
};

///
/// @brief Drop filesystem cache for a file (doesn't require root)
///
/// @param[in] filepath Path to the file
///
static void drop_file_cache(const std::string &filepath) {
  int fd = open(filepath.c_str(), O_RDONLY);
  if (fd >= 0) {
    // Tell kernel we don't need this file's pages in cache
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
  }
}

struct WorkerResult {
  double write_time;
  double read_time;
  bool success;
  std::string filename;  // Store filename to share between write and read phases
};

std::mutex cout_mutex;


///
/// @brief Worker function for WRITE phase: creates and writes HDF5 file
///
void worker_write(
    int thread_id,
    type::idx num,
    size_t cbuf,
    size_t fblk,
    size_t memb,
    bool asis,
    bool write_xdmf,
    const std::string &vfd_name,
    bool force_sync,
    type::idx *idx,
    type::pos *pos,
    type::vel_xy *vel_xy,
    type::vel_z *vel_z,
    WorkerResult &result) {
  
  cudaSetDevice(0);

  auto uuid = boost::uuids::random_generator{}();
  const auto series = boost::lexical_cast<std::string>(uuid);
  auto name = "dat/" + series + "_" + std::to_string(thread_id) + ".h5";
  
  // Store filename for read phase
  result.filename = name;

  constexpr auto benchmark = [](const auto func) noexcept(false) {
    struct timespec ini;
    clock_gettime(CLOCK_MONOTONIC, &ini);
    func();
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (std::fma(1.0e-9, static_cast<double>(end.tv_nsec - ini.tv_nsec), end.tv_sec - ini.tv_sec));
  };

  // prepare dataspaces for HDF5 - these are thread-local and safe
  const auto hdf5_dataspace_N = util::hdf5::setup_dataspace(num);
  const auto hdf5_dataspace_1 = util::hdf5::setup_dataspace();
  const auto [hdf5_dataspace_Nx3, hdf5_dataspace_Nx2, hdf5_dataspace_Nx1, hdf5_dataspace_Nx2_3, hdf5_dataspace_Nx1_3, hdf5_dataspace_Nx4, hdf5_dataspace_Nx3_4, hdf5_dataspace_Nx1_4] = util::hdf5::prepare_hyperslab_Nx3(num);
  
  auto h5write = util::hdf5::h5multi_write{};
  h5write.allocate(5);

  // Configure File Access Property List based on selected VFD
  auto fapl = H5Pcreate(H5P_FILE_ACCESS);
  
  if (vfd_name == "gds") {
    H5Pset_fapl_gds(fapl, memb, fblk, cbuf);
  } else if (vfd_name == "sec2") {
    H5Pset_fapl_sec2(fapl);
  } else if (vfd_name == "direct") {
    constexpr size_t alignment = 512;
    H5Pset_fapl_direct(fapl, alignment, fblk, cbuf);
  } else {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cerr << "Thread " << thread_id << ": Unknown VFD: " << vfd_name << std::endl;
    result.success = false;
    return;
  }

  // create HDF5 file
  auto target = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  if (target < 0) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cerr << "Thread " << thread_id << ": Failed to create HDF5 file: " << name << std::endl;
    result.success = false;
    return;
  }

  // preparation for H5Dwrite_multi()
  h5write.commit(hdf5_dataspace_N, target, "id", util::hdf5::h5type(*idx), idx);
  const auto FPtype = util::hdf5::h5type(*vel_z);
  if (!asis) {
    h5write.commit(hdf5_dataspace_Nx3, target, "velocity", FPtype, vel_xy, hdf5_dataspace_Nx2, hdf5_dataspace_Nx2_3);
    h5write.commit(vel_z, h5write.get_last_dataset(), FPtype, hdf5_dataspace_Nx1, hdf5_dataspace_Nx1_3);
    h5write.commit(hdf5_dataspace_Nx3, target, "position", FPtype, pos, hdf5_dataspace_Nx3_4, hdf5_dataspace_Nx3);
    h5write.commit(hdf5_dataspace_Nx1, target, "mass", FPtype, pos, hdf5_dataspace_Nx1_4, hdf5_dataspace_Nx1);
  } else {
    h5write.commit(hdf5_dataspace_N, target, "pos", util::hdf5::h5type(*pos), pos);
    h5write.commit(hdf5_dataspace_N, target, "vel_xy", util::hdf5::h5type(*vel_xy), vel_xy);
    h5write.commit(hdf5_dataspace_N, target, "vel_z", util::hdf5::h5type(*vel_z), vel_z);
  }

  // execute H5Dwrite_multi() and ensure data hits the disk
  result.write_time = benchmark([&]() {
    h5write.execute();

    // write attribute
    util::hdf5::write_attr(hdf5_dataspace_1, target, "num", &num);
    H5Fflush(target, H5F_SCOPE_GLOBAL);
    
    // Force physical disk write when --force flag is set
    // Ensures consistent benchmarking methodology across all VFDs
    if (force_sync) {
      int fd = open(name.c_str(), O_RDONLY);
      if (fd >= 0) {
        fsync(fd);
        close(fd);
      }
    }
    
    H5Fclose(target);
    H5Pclose(fapl);
  });

  // Cleanup dataspaces
  util::hdf5::close_dataspace(hdf5_dataspace_N);
  util::hdf5::close_dataspace(hdf5_dataspace_1);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx1_3);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx2_3);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx1);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx2);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx3);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx1_4);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx3_4);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx4);

  result.success = true;
}

///
/// @brief Worker function for READ phase: reads and verifies existing HDF5 file
/// 
void worker_read(
    int thread_id,
    type::idx num,
    size_t cbuf,
    size_t fblk,
    size_t memb,
    bool skip,
    bool asis,
    const std::string &vfd_name,
    type::idx *idx,
    type::pos *pos,
    type::vel_xy *vel_xy,
    type::vel_z *vel_z,
    WorkerResult &result) {
  
  cudaSetDevice(0);

  const auto &name = result.filename;  // Use filename from write phase

  constexpr auto benchmark = [](const auto func) noexcept(false) {
    cudaDeviceSynchronize(); 
    struct timespec ini;
    clock_gettime(CLOCK_MONOTONIC, &ini);
    func();
    cudaDeviceSynchronize();
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (std::fma(1.0e-9, static_cast<double>(end.tv_nsec - ini.tv_nsec), end.tv_sec - ini.tv_sec));
  };

  // prepare dataspaces for HDF5
  const auto hdf5_dataspace_N = util::hdf5::setup_dataspace(num);
  const auto [hdf5_dataspace_Nx3, hdf5_dataspace_Nx2, hdf5_dataspace_Nx1, hdf5_dataspace_Nx2_3, hdf5_dataspace_Nx1_3, hdf5_dataspace_Nx4, hdf5_dataspace_Nx3_4, hdf5_dataspace_Nx1_4] = util::hdf5::prepare_hyperslab_Nx3(num);
  
  auto h5read = util::hdf5::h5multi_read{};
  h5read.allocate(5);

  // Configure VFD
  auto fapl = H5Pcreate(H5P_FILE_ACCESS);
  
  if (vfd_name == "gds") {
    H5Pset_fapl_gds(fapl, memb, fblk, cbuf);
  } else if (vfd_name == "sec2") {
    H5Pset_fapl_sec2(fapl);
  } else if (vfd_name == "direct") {
    constexpr size_t alignment = 512;
    H5Pset_fapl_direct(fapl, alignment, fblk, cbuf);
  } else {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cerr << "Thread " << thread_id << ": Unknown VFD: " << vfd_name << std::endl;
    result.success = false;
    return;
  }

  cudaDeviceSynchronize();
  drop_file_cache(name);

  // Read back
  auto target = H5Fopen(name.c_str(), H5F_ACC_RDONLY, fapl);
  if (target < 0) {
    std::lock_guard<std::mutex> lock(cout_mutex);
    std::cerr << "Thread " << thread_id << ": Failed to open HDF5 file: " << name << std::endl;
    result.success = false;
    return;
  }

  auto num_read = std::remove_const_t<decltype(num)>{};
  util::hdf5::read_attr(target, "num", &num_read);
  
  // Allocate read buffers for this thread
  std::remove_reference_t<decltype(*idx)> *idx_read = nullptr;
  std::remove_reference_t<decltype(*pos)> *pos_read = nullptr;
  std::remove_reference_t<decltype(*vel_xy)> *vel_xy_read = nullptr;
  std::remove_reference_t<decltype(*vel_z)> *vel_z_read = nullptr;
  allocate_particles(&pos_read, &vel_xy_read, &vel_z_read, &idx_read, num_read);

  h5read.commit(target, "id", util::hdf5::h5type(*idx_read), idx_read);
  const auto FPtype_read = util::hdf5::h5type(*vel_z_read);
  if (!asis) {
    h5read.commit(target, "velocity", FPtype_read, vel_xy_read, hdf5_dataspace_Nx2, hdf5_dataspace_Nx2_3);
    h5read.commit(vel_z_read, h5read.get_last_dataset(), FPtype_read, hdf5_dataspace_Nx1, hdf5_dataspace_Nx1_3);
    h5read.commit(target, "position", FPtype_read, pos_read, hdf5_dataspace_Nx3_4, hdf5_dataspace_Nx3);
    h5read.commit(target, "mass", FPtype_read, pos_read, hdf5_dataspace_Nx1_4, hdf5_dataspace_Nx1);
  } else {
    h5read.commit(target, "pos", util::hdf5::h5type(*pos_read), pos_read);
    h5read.commit(target, "vel_xy", util::hdf5::h5type(*vel_xy_read), vel_xy_read);
    h5read.commit(target, "vel_z", util::hdf5::h5type(*vel_z_read), vel_z_read);
  }

  // Move H5Fclose inside timing for consistency with write phase
  // (write phase times H5Dwrite + H5Fflush + fsync + H5Fclose)
  result.read_time = benchmark([&]() { 
    h5read.execute();
    H5Fclose(target);
    H5Pclose(fapl);
  });

  // Cleanup dataspaces
  util::hdf5::close_dataspace(hdf5_dataspace_N);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx1_3);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx2_3);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx1);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx2);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx3);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx1_4);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx3_4);
  util::hdf5::close_dataspace(hdf5_dataspace_Nx4);

  // Verification
  bool local_success = skip ? true : (thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*idx)>>)idx, (thrust::device_ptr<std::remove_reference_t<decltype(*idx)>>)(idx + num), (thrust::device_ptr<std::remove_reference_t<decltype(*idx_read)>>)idx_read) && thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*pos)>>)pos, (thrust::device_ptr<std::remove_reference_t<decltype(*pos)>>)(pos + num), (thrust::device_ptr<std::remove_reference_t<decltype(*pos_read)>>)pos_read, compare_pos()) && thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_xy)>>)vel_xy, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_xy)>>)(vel_xy + num), (thrust::device_ptr<std::remove_reference_t<decltype(*vel_xy_read)>>)vel_xy_read, compare_vel_xy()) && thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_z)>>)vel_z, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_z)>>)(vel_z + num), (thrust::device_ptr<std::remove_reference_t<decltype(*vel_z_read)>>)vel_z_read));

  result.success = local_success;

  release_particles(pos_read, vel_xy_read, vel_z_read, idx_read);

  // Delete file after verification
  if (std::remove(name.c_str()) != 0) {
     // Warning ignored
  }
}

///
/// @brief main function
///
/// @param[in] argc number of input argument(s)
/// @param[in] argv input argument(s)
///
auto main(const int32_t argc, const char *const *const argv) -> int32_t {
  // use scientific notation for floating-point number
  std::cout << std::scientific;

  // initialize the simulation
  // prepare options
  boost::program_options::options_description opt("List of options");
  opt.add_options()(
      "num", boost::program_options::value<type::idx>()->default_value(1024), "number of particles")(
      "cbuf", boost::program_options::value<size_t>()->default_value(CBSIZE_DEF), "copy buffer size (byte)")(
      "fblk", boost::program_options::value<size_t>()->default_value(FBSIZE_DEF), "file block size (byte)")(
      "memb", boost::program_options::value<size_t>()->default_value(MBOUNDARY_DEF), "memory boundary (byte)")(
      "skip", boost::program_options::bool_switch()->default_value(false), "skip consistency check between read and original data")(
      "asis", boost::program_options::bool_switch()->default_value(false), "read/write without hyperslab")(
      "virial", boost::program_options::value<std::remove_const_t<decltype(newton)>>()->default_value(0.2), "Virial ratio of the system")(
      "radius", boost::program_options::value<std::remove_const_t<decltype(newton)>>()->default_value(1.0), "radius of the system")(
      "mass", boost::program_options::value<std::remove_const_t<decltype(newton)>>()->default_value(1.0), "total mass of the system")(
      "xdmf", boost::program_options::bool_switch()->default_value(false), "generate XDMF file to visualize the snapshot")(
      "threads", boost::program_options::value<int>()->default_value(1), "number of concurrent threads")(
      "vfd", boost::program_options::value<std::string>()->default_value("gds"), "VFD to use: gds (GPUDirect Storage), sec2 (POSIX unbuffered), direct (O_DIRECT)")(
      "iterations", boost::program_options::value<int>()->default_value(3), "number of benchmark iterations to average (reduces variance)")(
      "force", boost::program_options::bool_switch()->default_value(false), "force physical disk write (fsync) for fair storage benchmarking")(
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
  const auto cbuf = vm["cbuf"].as<size_t>();
  const auto fblk = vm["fblk"].as<size_t>();
  const auto memb = vm["memb"].as<size_t>();
  const auto virial = vm["virial"].as<decltype(newton)>();
  const auto radius = vm["radius"].as<decltype(newton)>();
  const auto mass = vm["mass"].as<decltype(newton)>();
  const auto skip = vm["skip"].as<bool>();
  const auto asis = vm["asis"].as<bool>();
  const auto write_xdmf = vm["xdmf"].as<bool>();
  const auto num_threads = vm["threads"].as<int>();
  const auto vfd_name = vm["vfd"].as<std::string>();
  const auto iterations = vm["iterations"].as<int>();
  const auto force_sync = vm["force"].as<bool>();
  vm.clear();

  // Validate iterations
  if (iterations < 1) {
    std::cerr << "ERROR: iterations must be at least 1" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Validate VFD selection
  if (vfd_name != "gds" && vfd_name != "sec2" && vfd_name != "direct") {
    std::cerr << "ERROR: Invalid VFD '" << vfd_name << "'" << std::endl;
    std::cerr << "Valid options: gds, sec2, direct" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // copy buffer size must be a multiple of block size
  if ((cbuf % fblk) != 0U) {
    std::cerr << "copy buffer size (" << cbuf << ") must be a multiple of block size (" << fblk << ")";
    std::cerr << std::endl;
    std::cerr << std::fflush;
    std::exit(EXIT_FAILURE);
  }

  // memory allocation (Source Data - Shared by all threads for writing)
  cudaSetDevice(0);
  type::idx *idx = nullptr;        // particle ID
  type::pos *pos = nullptr;        // position (x, y, z) and mass (w)
  type::vel_xy *vel_xy = nullptr;  // velocity (x, y)
  type::vel_z *vel_z = nullptr;    // velocity (z)

  allocate_particles(&pos, &vel_xy, &vel_z, &idx, num);

  // initialize data on GPU
  set_uniform_sphere(num, pos, vel_xy, vel_z, idx, mass, radius, virial, newton);

  // Initialize HDF5 compound types ONCE in main thread before spawning workers
  // This prevents race conditions where multiple threads try to create the same type definitions
  util::hdf5::create_h5t_real2();
  util::hdf5::create_h5t_real4();

  // ===============================================
  // WARMUP ITERATION (excluded from statistics)
  // ===============================================
  std::cout << "Running warmup iteration to initialize libraries (HDF5, CUDA)..." << std::endl;
  {
    std::vector<WorkerResult> warmup_results(num_threads);
    
    // Warmup write phase
    std::vector<std::thread> warmup_write_threads;
    for (int i = 0; i < num_threads; ++i) {
      warmup_write_threads.emplace_back(worker_write, i, num, cbuf, fblk, memb, asis, write_xdmf, vfd_name, force_sync, idx, pos, vel_xy, vel_z, std::ref(warmup_results[i]));
    }
    for (auto &t : warmup_write_threads) {
      t.join();
    }
    
    // Warmup read phase
    std::vector<std::thread> warmup_read_threads;
    for (int i = 0; i < num_threads; ++i) {
      warmup_read_threads.emplace_back(worker_read, i, num, cbuf, fblk, memb, skip, asis, vfd_name, idx, pos, vel_xy, vel_z, std::ref(warmup_results[i]));
    }
    for (auto &t : warmup_read_threads) {
      t.join();
    }
    
    std::cout << "Warmup completed (not included in statistics)\n" << std::endl;
  }

  // ===============================================
  // MULTIPLE ITERATIONS FOR STATISTICAL AVERAGING
  // ===============================================
  std::vector<double> all_write_times;
  std::vector<double> all_read_times;
  all_write_times.reserve(iterations);
  all_read_times.reserve(iterations);

  std::cout << "Running " << iterations << " benchmark iteration(s) for statistical averaging..." << std::endl;

  for (int iter = 0; iter < iterations; ++iter) {
    if (iterations > 1) {
      std::cout << "\n=== Iteration " << (iter + 1) << "/" << iterations << " ===" << std::endl;
    }

    std::vector<WorkerResult> results(num_threads);

    // ===============================================
    // PHASE 1: WRITE BENCHMARK
    // ===============================================
    std::cout << "Phase 1: Write benchmark - " << num_threads << " threads writing..." << std::endl;
    
    std::vector<std::thread> write_threads;
    for (int i = 0; i < num_threads; ++i) {
      write_threads.emplace_back(worker_write, i, num, cbuf, fblk, memb, asis, write_xdmf, vfd_name, force_sync, idx, pos, vel_xy, vel_z, std::ref(results[i]));
    }

    // Join write threads
    for (auto &t : write_threads) {
      t.join();
    }

    // Check write phase success
    bool write_success = true;
    for (const auto &res : results) {
      if (!res.success) {
        write_success = false;
        break;
      }
    }

    if (!write_success) {
      std::cerr << "ERROR: One or more threads failed during write phase." << std::endl;
      util::hdf5::remove_h5t_real2();
      util::hdf5::remove_h5t_real4();
      release_particles(pos, vel_xy, vel_z, idx);
      std::exit(EXIT_FAILURE);
    }

    // Collect write time (max across threads)
    double max_write_time = 0.0;
    for (const auto &res : results) {
      if (res.write_time > max_write_time) max_write_time = res.write_time;
    }
    all_write_times.push_back(max_write_time);

    std::cout << "Write phase completed: " << max_write_time << " s" << std::endl;

    // ===============================================
    // PHASE 2: READ BENCHMARK
    // ===============================================
    std::cout << "Phase 2: Read benchmark - " << num_threads << " threads reading..." << std::endl;

    std::vector<std::thread> read_threads;
    for (int i = 0; i < num_threads; ++i) {
      read_threads.emplace_back(worker_read, i, num, cbuf, fblk, memb, skip, asis, vfd_name, idx, pos, vel_xy, vel_z, std::ref(results[i]));
    }

    // Join read threads
    for (auto &t : read_threads) {
      t.join();
    }

    // Check read phase success
    bool all_success = true;
    for (const auto &res : results) {
      if (!res.success) {
        all_success = false;
        break;
      }
    }

    if (!all_success) {
      std::cerr << "ERROR: One or more threads failed during read phase." << std::endl;
      util::hdf5::remove_h5t_real2();
      util::hdf5::remove_h5t_real4();
      release_particles(pos, vel_xy, vel_z, idx);
      std::exit(EXIT_FAILURE);
    }

    // Collect read time (max across threads)
    double max_read_time = 0.0;
    for (const auto &res : results) {
      if (res.read_time > max_read_time) max_read_time = res.read_time;
    }
    all_read_times.push_back(max_read_time);

    std::cout << "Read phase completed: " << max_read_time << " s" << std::endl;
  }

  // Cleanup HDF5 compound types after all iterations completed
  util::hdf5::remove_h5t_real2();
  util::hdf5::remove_h5t_real4();

  // ===============================================
  // CALCULATE STATISTICS
  // ===============================================
  auto calc_mean = [](const std::vector<double>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
  };

  double avg_write_time = calc_mean(all_write_times);
  double avg_read_time = calc_mean(all_read_times);
  double min_write_time = *std::min_element(all_write_times.begin(), all_write_times.end());
  double max_write_time = *std::max_element(all_write_times.begin(), all_write_times.end());
  double min_read_time = *std::min_element(all_read_times.begin(), all_read_times.end());
  double max_read_time = *std::max_element(all_read_times.begin(), all_read_times.end());

  std::cout << "\n=== Statistics over " << iterations << " iteration(s) ===" << std::endl;
  std::cout << "Write: avg=" << avg_write_time << "s, min=" << min_write_time << "s, max=" << max_write_time << "s" << std::endl;
  std::cout << "Read:  avg=" << avg_read_time << "s, min=" << min_read_time << "s, max=" << max_read_time << "s" << std::endl;

  // output the benchmark result
  const std::string report = "log/h5gds_benchmark.csv";
  const boost::filesystem::path previous(report);
  boost::system::error_code err;
  const auto exist = boost::filesystem::exists(previous, err);

    // write header if report is a new file
    std::ofstream output(report, std::ios::app);
    if (!exist || err) {
      output << "N";
      output << ",threads";
      output << ",VFD";
      output << ",iterations";
      output << ",data size [byte]";
      output << ",copy buffer size [byte]";
      output << ",file block size [byte]";
      output << ",memory boundary [byte]";
      output << ",latency (write avg) [s]";
      output << ",latency (write min) [s]";
      output << ",latency (write max) [s]";
      output << ",latency (read avg) [s]";
      output << ",latency (read min) [s]";
      output << ",latency (read max) [s]";
      output << ",bandwidth (write avg) [byte/s]";
      output << ",bandwidth (read avg) [byte/s]";
      output << std::endl;
    }

    // write statistics of the simulation
    output << std::scientific;
    output << num;
    output << "," << num_threads;
    output << "," << vfd_name;
    output << "," << iterations;
    const auto datasize = static_cast<double>(num) * static_cast<double>(sizeof(std::remove_reference_t<decltype(*idx)>) + sizeof(std::remove_reference_t<decltype(*pos)>) + sizeof(std::remove_reference_t<decltype(*vel_xy)>) + sizeof(std::remove_reference_t<decltype(*vel_z)>));
    // Total data size processed is datasize * num_threads? 
    // Usually throughput is Total Bytes / Time.
    // If we want per-thread throughput, we use datasize. 
    // If we want aggregate, we use datasize * num_threads.
    // Let's report Aggregate Bandwidth.
    const auto total_datasize = datasize * num_threads;

    output << "," << total_datasize;
    output << "," << cbuf;
    output << "," << fblk;
    output << "," << memb;
    output << "," << avg_write_time;
    output << "," << min_write_time;
    output << "," << max_write_time;
    output << "," << avg_read_time;
    output << "," << min_read_time;
    output << "," << max_read_time;
    output << "," << total_datasize / avg_write_time;
    output << "," << total_datasize / avg_read_time;
    output << std::endl;
    output.close();

  release_particles(pos, vel_xy, vel_z, idx);

  std::exit(EXIT_SUCCESS);
}
