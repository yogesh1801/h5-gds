///
/// @file src/h5gds.cu
/// @author Yohei MIKI (The University of Tokyo)
/// @brief parameter study related with VFD for GDS
///
/// @copyright Copyright (c) 2023 Information Technology Center, The University of Tokyo
///
/// The MIT License is applied to this software, see LICENSE
///
#include <H5FDgds.h>        // VFD for GDS
#include <curand_mtgp32.h>  // THREAD_NUM
#include <hdf5.h>
#include <helper_cuda.h>  // checkCudaErrors
#include <thrust/device_ptr.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <boost/filesystem.hpp>            // boost::filesystem
#include <boost/lexical_cast.hpp>          // boost::lexical_cast
#include <boost/program_options.hpp>       // boost::program_options
#include <boost/uuid/uuid_generators.hpp>  // boost::uuids::random_generator
#include <boost/uuid/uuid_io.hpp>          // convert boost::uuids::uuid to std::string
#include <cstdlib>                         // std::exit
#include <fstream>                         // std::ofstream
#include <iomanip>                         // std::setw
#include <iostream>                        // std::cout
#include <sstream>                         // std::stringstream
#include <string>                          // std::string

#include "allocate.cuh"
#include "common.cuh"
#include "generate.cuh"
#include "hdf5.hpp"

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
      "cbuf", boost::program_options::value<size_t>()->default_value(CBSIZE_DEF), "copy buffer size (byte)")(
      "fblk", boost::program_options::value<size_t>()->default_value(FBSIZE_DEF), "file block size (byte)")(
      "memb", boost::program_options::value<size_t>()->default_value(MBOUNDARY_DEF), "memory boundary (byte)")(
      "vfd", boost::program_options::value<std::string>()->default_value("gds"), "VFD driver to use: sec2, gds, or direct")(
      "skip", boost::program_options::bool_switch()->default_value(false), "skip consistency check between read and original data")(
      "asis", boost::program_options::bool_switch()->default_value(false), "read/write without hyperslab")(
      "virial", boost::program_options::value<std::remove_const_t<decltype(newton)>>()->default_value(0.2), "Virial ratio of the system")(
      "radius", boost::program_options::value<std::remove_const_t<decltype(newton)>>()->default_value(1.0), "radius of the system")(
      "mass", boost::program_options::value<std::remove_const_t<decltype(newton)>>()->default_value(1.0), "total mass of the system")(
      "xdmf", boost::program_options::bool_switch()->default_value(false), "generate XDMF file to visualize the snapshot")(
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
  const auto vfd_name = vm["vfd"].as<std::string>();
  const auto virial = vm["virial"].as<decltype(newton)>();
  const auto radius = vm["radius"].as<decltype(newton)>();
  const auto mass = vm["mass"].as<decltype(newton)>();
  const auto skip = vm["skip"].as<bool>();
  const auto asis = vm["asis"].as<bool>();
  const auto write_xdmf = vm["xdmf"].as<bool>();
  vm.clear();
  // validate VFD choice
  if (vfd_name != "sec2" && vfd_name != "gds" && vfd_name != "direct") {
    std::cerr << "Invalid VFD driver: " << vfd_name << ". Must be one of: sec2, gds, direct" << std::endl;
    std::cerr << std::fflush;
    std::exit(EXIT_FAILURE);
  }
  // copy buffer size must be a multiple of block size
  if ((cbuf % fblk) != 0U) {
    std::cerr << "copy buffer size (" << cbuf << ") must be a multiple of block size (" << fblk << ")";
    std::cerr << std::endl;
    std::cerr << std::fflush;
    std::exit(EXIT_FAILURE);
  }

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

  // Allocate host buffers for sec2/direct VFDs when using cudaMalloc
  type::idx* idx_host = nullptr;
  type::pos* pos_host = nullptr;
  type::vel_xy* vel_xy_host = nullptr;
  type::vel_z* vel_z_host = nullptr;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // Only allocate host buffers if using sec2 or direct VFD with cudaMalloc
  if (vfd_name == "sec2" || vfd_name == "direct") {
    auto size = round_up(num, NTHREADS);
    size = round_up(size, THREAD_NUM);
    idx_host = (type::idx*)malloc(size * sizeof(type::idx));
    pos_host = (type::pos*)malloc(size * sizeof(type::pos));
    vel_xy_host = (type::vel_xy*)malloc(size * sizeof(type::vel_xy));
    vel_z_host = (type::vel_z*)malloc(size * sizeof(type::vel_z));

    if (!idx_host || !pos_host || !vel_xy_host || !vel_z_host) {
      std::cerr << "Failed to allocate host buffers for " << vfd_name << " VFD" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif

  set_uniform_sphere(num, pos, vel_xy, vel_z, idx, mass, radius, virial, newton);

  constexpr auto benchmark = [](const auto func) noexcept(false) {
    // cudaDeviceSynchronize();
    struct timespec ini;
    // clock_gettime(CLOCK_MONOTONIC_RAW, &ini);
    clock_gettime(CLOCK_MONOTONIC, &ini);
    // clock_gettime(CLOCK_BOOTTIME, &ini);
    func();
    // cudaDeviceSynchronize();
    struct timespec end;
    // clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    clock_gettime(CLOCK_MONOTONIC, &end);
    // clock_gettime(CLOCK_BOOTTIME, &end);
    return (std::fma(1.0e-9, static_cast<double>(end.tv_nsec - ini.tv_nsec), end.tv_sec - ini.tv_sec));
  };

  // prepare dataspaces for HDF5
  util::hdf5::create_h5t_real2();
  util::hdf5::create_h5t_real4();
  const auto hdf5_dataspace_N = util::hdf5::setup_dataspace(num);
  const auto hdf5_dataspace_1 = util::hdf5::setup_dataspace();
  const auto [hdf5_dataspace_Nx3, hdf5_dataspace_Nx2, hdf5_dataspace_Nx1, hdf5_dataspace_Nx2_3, hdf5_dataspace_Nx1_3, hdf5_dataspace_Nx4, hdf5_dataspace_Nx3_4, hdf5_dataspace_Nx1_4] = util::hdf5::prepare_hyperslab_Nx3(num);
  auto h5write = util::hdf5::h5multi_write{};
  auto h5read = util::hdf5::h5multi_read{};
  h5write.allocate(5);  // idx, position (x, y, z), velocity (x, y), velocity (z), and mass
  h5read.allocate(5);   // idx, position (x, y, z), velocity (x, y), velocity (z), and mass

  // prepare file access property list based on selected VFD
  auto fapl = H5Pcreate(H5P_FILE_ACCESS);
  if (vfd_name == "gds") {
    std::cout << "using HDF5 gds fapl" << std::endl;
    H5Pset_fapl_gds(fapl, memb, fblk, cbuf);
  } else if (vfd_name == "direct") {
    std::cout << "using HDF5 direct fapl" << std::endl;
    H5Pset_fapl_direct(fapl, memb, fblk, cbuf);
  } else {  // sec2
    std::cout << "using HDF5 sec2 fapl" << std::endl;
    H5Pset_fapl_sec2(fapl);
  }

  H5Pset_alignment(fapl, 0, 4096);

  // create HDF5 file
  auto uuid = boost::uuids::random_generator{}();
  const auto series = boost::lexical_cast<std::string>(uuid);
  auto name = "dat/" + series + ".h5";
  auto target = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);

  // preparation for H5Dwrite_multi()
  // Use host buffers for sec2/direct with cudaMalloc, otherwise use original pointers
  auto* idx_write = idx;
  auto* pos_write = pos;
  auto* vel_xy_write = vel_xy;
  auto* vel_z_write = vel_z;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  if (vfd_name == "sec2" || vfd_name == "direct") {
    idx_write = idx_host;
    pos_write = pos_host;
    vel_xy_write = vel_xy_host;
    vel_z_write = vel_z_host;
  }
#endif

  h5write.commit(hdf5_dataspace_N, target, "id", util::hdf5::h5type(*idx), idx_write);
  const auto FPtype = util::hdf5::h5type(*vel_z);
  if (!asis) {
    h5write.commit(hdf5_dataspace_Nx3, target, "velocity", FPtype, vel_xy_write, hdf5_dataspace_Nx2, hdf5_dataspace_Nx2_3);
    h5write.commit(vel_z_write, h5write.get_last_dataset(), FPtype, hdf5_dataspace_Nx1, hdf5_dataspace_Nx1_3);
    h5write.commit(hdf5_dataspace_Nx3, target, "position", FPtype, pos_write, hdf5_dataspace_Nx3_4, hdf5_dataspace_Nx3);
    h5write.commit(hdf5_dataspace_Nx1, target, "mass", FPtype, pos_write, hdf5_dataspace_Nx1_4, hdf5_dataspace_Nx1);
  } else {
    h5write.commit(hdf5_dataspace_N, target, "pos", util::hdf5::h5type(*pos), pos_write);
    h5write.commit(hdf5_dataspace_N, target, "vel_xy", util::hdf5::h5type(*vel_xy), vel_xy_write);
    h5write.commit(hdf5_dataspace_N, target, "vel_z", util::hdf5::h5type(*vel_z), vel_z_write);
  }
  // execute H5Dwrite_multi()
  // h5write.execute();
  const auto elapse_write = benchmark([&]() {
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
    // Copy GPU data to host buffers for sec2/direct VFDs (INSIDE timing)
    if (vfd_name == "sec2" || vfd_name == "direct") {
      auto size = round_up(num, NTHREADS);
      size = round_up(size, THREAD_NUM);

      checkCudaErrors(cudaMemcpy(idx_host, idx, size * sizeof(type::idx), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(pos_host, pos, size * sizeof(type::pos), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(vel_xy_host, vel_xy, size * sizeof(type::vel_xy), cudaMemcpyDeviceToHost));
      checkCudaErrors(cudaMemcpy(vel_z_host, vel_z, size * sizeof(type::vel_z), cudaMemcpyDeviceToHost));
    }
#endif

    h5write.execute();
  });
  // write attribute
  util::hdf5::write_attr(hdf5_dataspace_1, target, "num", &num);
  // close the file
  H5Fclose(target);

  // generate XDMF file if requested
  if (!asis && write_xdmf) {
    std::ofstream xml("dat/" + series + ".xdmf", std::ios::out);

    xml << R"(<?xml version="1.0" ?>)" << std::endl;
    xml << R"(<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>)" << std::endl;
    xml << R"(<Xdmf Version="3.0">)" << std::endl;
    xml << "  <Domain>" << std::endl;
    xml << R"(    <Grid Name="particle" GridType="Uniform">)" << std::endl;
    xml << R"(      <Topology TopologyType="Polyvertex" NumberOfElements=")" << num << R"("/>)" << std::endl;

    xml << R"(      <Geometry GeometryType="XYZ">)" << std::endl;
    xml << R"(        <DataItem Dimensions=")" << num << R"( 3" NumberType="Float" Precision=")" << sizeof(decltype(*vel_z)) << R"(" Format="HDF">)" << std::endl;
    xml << "          " << series + ".h5"
        << ":/"
        << "position" << std::endl;
    xml << "        </DataItem>" << std::endl;
    xml << "      </Geometry>" << std::endl;

    xml << R"(      <Attribute Name="velocity" AttributeType="Vector" Center="Node">)" << std::endl;
    xml << R"(        <DataItem Dimensions=")" << num << R"( 3" NumberType="Float" Precision=")" << sizeof(decltype(*vel_z)) << R"(" Format="HDF">)" << std::endl;
    xml << "          " << series + ".h5"
        << ":/"
        << "velocity" << std::endl;
    xml << "        </DataItem>" << std::endl;
    xml << "      </Attribute>" << std::endl;

    xml << R"(      <Attribute Name="mass" AttributeType="Scalar" Center="Node">)" << std::endl;
    xml << R"(        <DataItem Dimensions=")" << num << R"(" NumberType="Float" Precision=")" << sizeof(decltype(*vel_z)) << R"(" Format="HDF">)" << std::endl;
    xml << "          " << series + ".h5"
        << ":/"
        << "mass" << std::endl;
    xml << "        </DataItem>" << std::endl;
    xml << "      </Attribute>" << std::endl;

    xml << R"(      <Attribute Name="ID" AttributeType="Scalar" Center="Node">)" << std::endl;
    xml << R"(        <DataItem Dimensions=")" << num << R"(" NumberType="UInt" Precision=")" << sizeof(decltype(*idx)) << R"(" Format="HDF">)" << std::endl;
    xml << "          " << series + ".h5"
        << ":/"
        << "id" << std::endl;
    xml << "        </DataItem>" << std::endl;
    xml << "      </Attribute>" << std::endl;

    xml << "    </Grid>" << std::endl;
    xml << "  </Domain>" << std::endl;
    xml << "</Xdmf>" << std::endl;
    xml.close();
  }

  // read the file and compare
  target = H5Fopen(name.c_str(), H5F_ACC_RDONLY, fapl);
  auto num_read = std::remove_const_t<decltype(num)>{};
  util::hdf5::read_attr(target, "num", &num_read);
  if (num_read != num) {
    std::cerr << __FILE__ << "(" << __LINE__ << "): " << __func__ << ": ERROR: num_read (" << num_read << ") does not match with num (" << num << ")" << std::endl
              << std::flush;
    std::exit(EXIT_FAILURE);
  }
  std::remove_reference_t<decltype(*idx)>* idx_read = nullptr;        // particle ID
  std::remove_reference_t<decltype(*pos)>* pos_read = nullptr;        // position (x, y, z) and mass (w)
  std::remove_reference_t<decltype(*vel_xy)>* vel_xy_read = nullptr;  // velocity (x, y)
  std::remove_reference_t<decltype(*vel_z)>* vel_z_read = nullptr;    // velocity (z)
  allocate_particles(&pos_read, &vel_xy_read, &vel_z_read, &idx_read, num_read);

  // Allocate host buffers for reading with sec2/direct VFDs
  type::idx* idx_read_host = nullptr;
  type::pos* pos_read_host = nullptr;
  type::vel_xy* vel_xy_read_host = nullptr;
  type::vel_z* vel_z_read_host = nullptr;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  if (vfd_name == "sec2" || vfd_name == "direct") {
    auto size = round_up(num_read, NTHREADS);
    size = round_up(size, THREAD_NUM);

    idx_read_host = (type::idx*)malloc(size * sizeof(type::idx));
    pos_read_host = (type::pos*)malloc(size * sizeof(type::pos));
    vel_xy_read_host = (type::vel_xy*)malloc(size * sizeof(type::vel_xy));
    vel_z_read_host = (type::vel_z*)malloc(size * sizeof(type::vel_z));

    if (!idx_read_host || !pos_read_host || !vel_xy_read_host || !vel_z_read_host) {
      std::cerr << "Failed to allocate host read buffers for " << vfd_name << " VFD" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
#endif

  // preparation for H5Dread_multi()
  // Use host buffers for sec2/direct with cudaMalloc, otherwise use original pointers
  auto* idx_read_ptr = idx_read;
  auto* pos_read_ptr = pos_read;
  auto* vel_xy_read_ptr = vel_xy_read;
  auto* vel_z_read_ptr = vel_z_read;

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  if (vfd_name == "sec2" || vfd_name == "direct") {
    idx_read_ptr = idx_read_host;
    pos_read_ptr = pos_read_host;
    vel_xy_read_ptr = vel_xy_read_host;
    vel_z_read_ptr = vel_z_read_host;
  }
#endif

  h5read.commit(target, "id", util::hdf5::h5type(*idx_read), idx_read_ptr);
  const auto FPtype_read = util::hdf5::h5type(*vel_z_read);
  if (!asis) {
    h5read.commit(target, "velocity", FPtype_read, vel_xy_read_ptr, hdf5_dataspace_Nx2, hdf5_dataspace_Nx2_3);
    h5read.commit(vel_z_read_ptr, h5read.get_last_dataset(), FPtype_read, hdf5_dataspace_Nx1, hdf5_dataspace_Nx1_3);
    h5read.commit(target, "position", FPtype_read, pos_read_ptr, hdf5_dataspace_Nx3_4, hdf5_dataspace_Nx3);
    h5read.commit(target, "mass", FPtype_read, pos_read_ptr, hdf5_dataspace_Nx1_4, hdf5_dataspace_Nx1);
  } else {
    h5read.commit(target, "pos", util::hdf5::h5type(*pos_read), pos_read_ptr);
    h5read.commit(target, "vel_xy", util::hdf5::h5type(*vel_xy_read), vel_xy_read_ptr);
    h5read.commit(target, "vel_z", util::hdf5::h5type(*vel_z_read), vel_z_read_ptr);
  }
  // execute H5Dread_multi()
  // h5read.execute();
  const auto elapse_read = benchmark([&]() {
    h5read.execute();

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
    // Copy host data to GPU buffers for sec2/direct VFDs (INSIDE timing)
    if (vfd_name == "sec2" || vfd_name == "direct") {
      auto size = round_up(num_read, NTHREADS);
      size = round_up(size, THREAD_NUM);

      checkCudaErrors(cudaMemcpy(idx_read, idx_read_host, size * sizeof(type::idx), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(pos_read, pos_read_host, size * sizeof(type::pos), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(vel_xy_read, vel_xy_read_host, size * sizeof(type::vel_xy), cudaMemcpyHostToDevice));
      checkCudaErrors(cudaMemcpy(vel_z_read, vel_z_read_host, size * sizeof(type::vel_z), cudaMemcpyHostToDevice));
    }
#endif
  });

  // close the file
  H5Fclose(target);
  H5Pclose(fapl);

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
  util::hdf5::remove_h5t_real2();
  util::hdf5::remove_h5t_real4();

  // check the read results
  const auto success = skip ? true : (thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*idx)>>)idx, (thrust::device_ptr<std::remove_reference_t<decltype(*idx)>>)(idx + num), (thrust::device_ptr<std::remove_reference_t<decltype(*idx_read)>>)idx_read) && thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*pos)>>)pos, (thrust::device_ptr<std::remove_reference_t<decltype(*pos)>>)(pos + num), (thrust::device_ptr<std::remove_reference_t<decltype(*pos_read)>>)pos_read, compare_pos()) && thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_xy)>>)vel_xy, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_xy)>>)(vel_xy + num), (thrust::device_ptr<std::remove_reference_t<decltype(*vel_xy_read)>>)vel_xy_read, compare_vel_xy()) && thrust::equal(thrust::device, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_z)>>)vel_z, (thrust::device_ptr<std::remove_reference_t<decltype(*vel_z)>>)(vel_z + num), (thrust::device_ptr<std::remove_reference_t<decltype(*vel_z_read)>>)vel_z_read));

  if (success) {
    // output the benchmark result
    const std::string report = "log/h5gds_benchmark.csv";
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
      output << ",copy buffer size [byte]";
      output << ",file block size [byte]";
      output << ",memory boundary [byte]";
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
    output << "," << cbuf;
    output << "," << fblk;
    output << "," << memb;
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

  // Free host buffers for sec2/direct VFDs
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  if (vfd_name == "sec2" || vfd_name == "direct") {
    free(idx_host);
    free(pos_host);
    free(vel_xy_host);
    free(vel_z_host);
    free(idx_read_host);
    free(pos_read_host);
    free(vel_xy_read_host);
    free(vel_z_read_host);
  }
#endif

  release_particles(pos, vel_xy, vel_z, idx);
  release_particles(pos_read, vel_xy_read, vel_z_read, idx_read);

  // delete the file to save space
  boost::filesystem::remove(name);
  if (!asis && write_xdmf) {
    boost::filesystem::remove("dat/" + series + ".xdmf");
  }

  std::exit(EXIT_SUCCESS);
}
