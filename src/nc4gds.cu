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
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <boost/filesystem.hpp>            // boost::filesystem
#include <boost/lexical_cast.hpp>          // boost::lexical_cast
#include <boost/program_options.hpp>       // boost::program_options
#include <boost/uuid/uuid_generators.hpp>  // boost::uuids::random_generator
#include <boost/uuid/uuid_io.hpp>          // convert boost::uuids::uuid to std::string
#include <cstdlib>                         // std::exit, setenv
#include <fstream>                         // std::ofstream
#include <iostream>                        // std::cout
#include <string>                          // std::string

#include "allocate.cuh"
#include "common.cuh"
#include "generate.cuh"
#include "netcdf4.hpp"

// Utility function for rounding up to nearest multiple
constexpr auto round_up(const size_t org, const size_t unit) {
  const size_t mod = org % unit;
  return ((mod == 0) ? org : (org + unit - mod));
}

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
      "virial", boost::program_options::value<std::remove_const_t<decltype(newton)>>()->default_value(0.2), "Virial ratio of the system")(
      "radius", boost::program_options::value<std::remove_const_t<decltype(newton)>>()->default_value(1.0), "radius of the system")(
      "mass", boost::program_options::value<std::remove_const_t<decltype(newton)>>()->default_value(1.0), "total mass of the system")(
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
  const auto virial = vm["virial"].as<decltype(newton)>();
  const auto radius = vm["radius"].as<decltype(newton)>();
  const auto mass = vm["mass"].as<decltype(newton)>();
  const auto skip = vm["skip"].as<bool>();
  vm.clear();

  // VFD is configured externally via HDF5_DRIVER environment variable
  // e.g., export HDF5_DRIVER=gds (or direct, sec2)
  std::cout << "Using HDF5 VFD from HDF5_DRIVER environment variable for NetCDF-4" << std::endl;

  // memory allocation
  cudaSetDevice(0);
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  type::idx* idx = nullptr;        // particle ID
  type::pos* pos = nullptr;        // position (x, y, z) and mass (w)
  type::vel_xy* vel_xy = nullptr;  // velocity (x, y)
  type::vel_z* vel_z = nullptr;    // velocity (z)
#else
  type::idx* idx;        // particle ID
  type::pos* pos;        // position (x, y, z) and mass (w)
  type::vel_xy* vel_xy;  // velocity (x, y)
  type::vel_z* vel_z;    // velocity (z)
#endif
  allocate_particles(&pos, &vel_xy, &vel_z, &idx, num);

  // Allocate host buffers for first-touch with cudaMalloc
  type::idx* idx_host = nullptr;
  type::pos* pos_host = nullptr;
  type::vel_xy* vel_xy_host = nullptr;
  type::vel_z* vel_z_host = nullptr;
  float* position_buf = nullptr;
  float* velocity_buf = nullptr;
  float* mass_buf = nullptr;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // For device memory, we need host buffers for NetCDF I/O
  {
    auto size = round_up(num, NTHREADS);
    size = round_up(size, THREAD_NUM);
    idx_host = (type::idx*)malloc(size * sizeof(type::idx));
    pos_host = (type::pos*)malloc(size * sizeof(type::pos));
    vel_xy_host = (type::vel_xy*)malloc(size * sizeof(type::vel_xy));
    vel_z_host = (type::vel_z*)malloc(size * sizeof(type::vel_z));

    if (!idx_host || !pos_host || !vel_xy_host || !vel_z_host) {
      std::cerr << "Failed to allocate host buffers" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif

  // Allocate reorganization buffers (NetCDF expects Nx3 layout)
  position_buf = (float*)malloc(num * 3 * sizeof(float));
  velocity_buf = (float*)malloc(num * 3 * sizeof(float));
  mass_buf = (float*)malloc(num * sizeof(float));

  if (!position_buf || !velocity_buf || !mass_buf) {
    std::cerr << "Failed to allocate reorganization buffers" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  set_uniform_sphere(num, pos, vel_xy, vel_z, idx, mass, radius, virial, newton);

  constexpr auto benchmark = [](const auto func) noexcept(false) {
    struct timespec ini;
    clock_gettime(CLOCK_MONOTONIC, &ini);
    func();
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    return (std::fma(1.0e-9, static_cast<double>(end.tv_nsec - ini.tv_nsec), end.tv_sec - ini.tv_sec));
  };

  // Create NetCDF-4 file using pure NetCDF-C API
  // HDF5 will use the VFD configured via HDF5_DRIVER environment variable
  auto uuid = boost::uuids::random_generator{}();
  const auto series = boost::lexical_cast<std::string>(uuid);
  auto name = "dat/" + series + ".nc";

  int ncid;
  int dim_n, dim_3;
  int var_pos, var_vel, var_mass, var_id;

  // Prepare write pointers
  auto* idx_write = idx;
  auto* pos_write = pos;
  auto* vel_xy_write = vel_xy;
  auto* vel_z_write = vel_z;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  idx_write = idx_host;
  pos_write = pos_host;
  vel_xy_write = vel_xy_host;
  vel_z_write = vel_z_host;
#endif

  const auto elapse_write = benchmark([&]() {
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
    // Copy GPU data to host buffers (INSIDE timing - this is part of the I/O pipeline)
    auto size = round_up(num, NTHREADS);
    size = round_up(size, THREAD_NUM);

    checkCudaErrors(cudaMemcpy(idx_host, idx, size * sizeof(type::idx), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(pos_host, pos, size * sizeof(type::pos), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(vel_xy_host, vel_xy, size * sizeof(type::vel_xy), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(vel_z_host, vel_z, size * sizeof(type::vel_z), cudaMemcpyDeviceToHost));
#endif

    // Reorganize data from SoA to Nx3 layout for NetCDF
    for (size_t i = 0; i < num; i++) {
      position_buf[i * 3 + 0] = pos_write[i].x;
      position_buf[i * 3 + 1] = pos_write[i].y;
      position_buf[i * 3 + 2] = pos_write[i].z;
      mass_buf[i] = pos_write[i].w;
      velocity_buf[i * 3 + 0] = vel_xy_write[i].x;
      velocity_buf[i * 3 + 1] = vel_xy_write[i].y;
      velocity_buf[i * 3 + 2] = vel_z_write[i];
    }

    // Create NetCDF-4 file (HDF5 engine uses VFD from HDF5_DRIVER env var)
    NC_CHECK(nc_create(name.c_str(), NC_NETCDF4 | NC_CLOBBER, &ncid));

    // Define dimensions
    NC_CHECK(nc_def_dim(ncid, "N", num, &dim_n));
    NC_CHECK(nc_def_dim(ncid, "three", 3, &dim_3));

    // Define variables
    int dims_2d[2] = {dim_n, dim_3};
    NC_CHECK(nc_def_var(ncid, "position", NC_FLOAT, 2, dims_2d, &var_pos));
    NC_CHECK(nc_def_var(ncid, "velocity", NC_FLOAT, 2, dims_2d, &var_vel));
    NC_CHECK(nc_def_var(ncid, "mass", NC_FLOAT, 1, &dim_n, &var_mass));
    NC_CHECK(nc_def_var(ncid, "id", NC_UINT64, 1, &dim_n, &var_id));

    // Write num as global attribute
    NC_CHECK(nc_put_att_ulonglong(ncid, NC_GLOBAL, "num", NC_UINT64, 1, &num));

    // End define mode
    NC_CHECK(nc_enddef(ncid));

    // Write data using pure NetCDF API
    NC_CHECK(nc_put_var_float(ncid, var_pos, position_buf));
    NC_CHECK(nc_put_var_float(ncid, var_vel, velocity_buf));
    NC_CHECK(nc_put_var_float(ncid, var_mass, mass_buf));
    NC_CHECK(nc_put_var_ulonglong(ncid, var_id, idx_write));

    // Close file
    NC_CHECK(nc_close(ncid));
  });

  // Allocate read buffers
  std::remove_reference_t<decltype(*idx)>* idx_read = nullptr;
  std::remove_reference_t<decltype(*pos)>* pos_read = nullptr;
  std::remove_reference_t<decltype(*vel_xy)>* vel_xy_read = nullptr;
  std::remove_reference_t<decltype(*vel_z)>* vel_z_read = nullptr;
  allocate_particles(&pos_read, &vel_xy_read, &vel_z_read, &idx_read, num);

  type::idx* idx_read_host = nullptr;
  type::pos* pos_read_host = nullptr;
  type::vel_xy* vel_xy_read_host = nullptr;
  type::vel_z* vel_z_read_host = nullptr;
  float* position_read_buf = nullptr;
  float* velocity_read_buf = nullptr;
  float* mass_read_buf = nullptr;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  {
    auto size = round_up(num, NTHREADS);
    size = round_up(size, THREAD_NUM);

    idx_read_host = (type::idx*)malloc(size * sizeof(type::idx));
    pos_read_host = (type::pos*)malloc(size * sizeof(type::pos));
    vel_xy_read_host = (type::vel_xy*)malloc(size * sizeof(type::vel_xy));
    vel_z_read_host = (type::vel_z*)malloc(size * sizeof(type::vel_z));

    if (!idx_read_host || !pos_read_host || !vel_xy_read_host || !vel_z_read_host) {
      std::cerr << "Failed to allocate host read buffers" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif

  position_read_buf = (float*)malloc(num * 3 * sizeof(float));
  velocity_read_buf = (float*)malloc(num * 3 * sizeof(float));
  mass_read_buf = (float*)malloc(num * sizeof(float));

  auto* idx_read_ptr = idx_read;
  auto* pos_read_ptr = pos_read;
  auto* vel_xy_read_ptr = vel_xy_read;
  auto* vel_z_read_ptr = vel_z_read;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  idx_read_ptr = idx_read_host;
  pos_read_ptr = pos_read_host;
  vel_xy_read_ptr = vel_xy_read_host;
  vel_z_read_ptr = vel_z_read_host;
#endif

  const auto elapse_read = benchmark([&]() {
    // Open NetCDF file for reading
    NC_CHECK(nc_open(name.c_str(), NC_NOWRITE, &ncid));

    // Read num attribute and verify
    type::idx num_read;
    NC_CHECK(nc_get_att_ulonglong(ncid, NC_GLOBAL, "num", &num_read));
    if (num_read != num) {
      std::cerr << "num_read (" << num_read << ") does not match num (" << num << ")" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Get variable IDs
    int var_pos_r, var_vel_r, var_mass_r, var_id_r;
    NC_CHECK(nc_inq_varid(ncid, "position", &var_pos_r));
    NC_CHECK(nc_inq_varid(ncid, "velocity", &var_vel_r));
    NC_CHECK(nc_inq_varid(ncid, "mass", &var_mass_r));
    NC_CHECK(nc_inq_varid(ncid, "id", &var_id_r));

    // Read data using pure NetCDF API
    NC_CHECK(nc_get_var_float(ncid, var_pos_r, position_read_buf));
    NC_CHECK(nc_get_var_float(ncid, var_vel_r, velocity_read_buf));
    NC_CHECK(nc_get_var_float(ncid, var_mass_r, mass_read_buf));
    NC_CHECK(nc_get_var_ulonglong(ncid, var_id_r, idx_read_ptr));

    // Close file
    NC_CHECK(nc_close(ncid));

    // Reorganize from Nx3 back to SoA
    for (size_t i = 0; i < num; i++) {
      pos_read_ptr[i].x = position_read_buf[i * 3 + 0];
      pos_read_ptr[i].y = position_read_buf[i * 3 + 1];
      pos_read_ptr[i].z = position_read_buf[i * 3 + 2];
      pos_read_ptr[i].w = mass_read_buf[i];
      vel_xy_read_ptr[i].x = velocity_read_buf[i * 3 + 0];
      vel_xy_read_ptr[i].y = velocity_read_buf[i * 3 + 1];
      vel_z_read_ptr[i] = velocity_read_buf[i * 3 + 2];
    }

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
    // Copy host data to GPU buffers (INSIDE timing - part of I/O pipeline)
    auto size = round_up(num, NTHREADS);
    size = round_up(size, THREAD_NUM);

    checkCudaErrors(cudaMemcpy(idx_read, idx_read_host, size * sizeof(type::idx), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(pos_read, pos_read_host, size * sizeof(type::pos), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(vel_xy_read, vel_xy_read_host, size * sizeof(type::vel_xy), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(vel_z_read, vel_z_read_host, size * sizeof(type::vel_z), cudaMemcpyHostToDevice));
#endif
  });

  // Free temporary buffers
  free(position_buf);
  free(velocity_buf);
  free(mass_buf);
  free(position_read_buf);
  free(velocity_read_buf);
  free(mass_read_buf);

  // check the read results
  const auto success = skip ? true : (thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*idx)>>)idx, (thrust::device_ptr<std::remove_reference_t<decltype(*idx)>>)(idx + num), (thrust::device_ptr<std::remove_reference_t<decltype(*idx_read)>>)idx_read) && thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*pos)>>)pos, (thrust::device_ptr<std::remove_reference_t<decltype(*pos)>>)(pos + num), (thrust::device_ptr<std::remove_reference_t<decltype(*pos_read)>>)pos_read, compare_pos()) && thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_xy)>>)vel_xy, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_xy)>>)(vel_xy + num), (thrust::device_ptr<std::remove_reference_t<decltype(*vel_xy_read)>>)vel_xy_read, compare_vel_xy()) && thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_z)>>)vel_z, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_z)>>)(vel_z + num), (thrust::device_ptr<std::remove_reference_t<decltype(*vel_z_read)>>)vel_z_read));

  if (success) {
    // output the benchmark result
    const std::string report = "log/nc4gds_benchmark.csv";
    const boost::filesystem::path previous(report);
    boost::system::error_code err;
    const auto exist = boost::filesystem::exists(previous, err);

    // write header if report is a new file
    std::ofstream output(report, std::ios::app);
    if (!exist || err) {
      output << "VFD";
      output << ",skip";
      output << ",N";
      output << ",data size [byte]";
      output << ",latency (write) [s]";
      output << ",latency (read) [s]";
      output << ",bandwidth (write) [byte/s]";
      output << ",bandwidth (read) [byte/s]";
      output << ",filename";
      output << std::endl;
    }

    // write statistics of the simulation
    output << std::scientific;
    output << vfd_name;
    output << "," << (skip ? "true" : "false");
    output << "," << num;
    const auto datasize = static_cast<double>(num) * static_cast<double>(sizeof(std::remove_reference_t<decltype(*idx)>) + sizeof(std::remove_reference_t<decltype(*pos)>) + sizeof(std::remove_reference_t<decltype(*vel_xy)>) + sizeof(std::remove_reference_t<decltype(*vel_z)>));
    output << "," << datasize;
    output << "," << elapse_write;
    output << "," << elapse_read;
    output << "," << datasize / elapse_write;
    output << "," << datasize / elapse_read;
    output << "," << name;
    output << std::endl;
    output.close();
  } else {
    std::cerr << __FILE__ << "(" << __LINE__ << "): " << __func__ << ": ERROR: read data does not match with the original data" << std::endl
              << std::flush;
    std::exit(EXIT_FAILURE);
  }

  // Free host buffers
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  free(idx_host);
  free(pos_host);
  free(vel_xy_host);
  free(vel_z_host);
  free(idx_read_host);
  free(pos_read_host);
  free(vel_xy_read_host);
  free(vel_z_read_host);
#endif

  release_particles(pos, vel_xy, vel_z, idx);
  release_particles(pos_read, vel_xy_read, vel_z_read, idx_read);

  // delete the file to save space
  boost::filesystem::remove(name);

  std::exit(EXIT_SUCCESS);
}
