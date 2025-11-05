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
#include <cstdlib>                         // std::exit
#include <fcntl.h>                         // open(), O_RDONLY
#include <fstream>                         // std::ofstream
#include <iomanip>                         // std::setw
#include <iostream>                        // std::cout
#include <sstream>                         // std::stringstream
#include <string>                          // std::string
#include <unistd.h>                        // close(), posix_fadvise()

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
  vm.clear();
  // copy buffer size must be a multiple of block size
  if ((cbuf % fblk) != 0U) {
    std::cerr << "copy buffer size (" << cbuf << ") must be a multiple of block size (" << fblk << ")";
    std::cerr << std::endl;
    std::cerr << std::fflush;
    std::exit(EXIT_FAILURE);
  }

  // memory allocation
  cudaSetDevice(0);
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH)
  type::idx *idx = nullptr;        // particle ID
  type::pos *pos = nullptr;        // position (x, y, z) and mass (w)
  type::vel_xy *vel_xy = nullptr;  // velocity (x, y)
  type::vel_z *vel_z = nullptr;    // velocity (z)
#else                              //! defined(HOST_MALLOC_AND_FIRST_TOUCH)
  type::idx *idx;        // particle ID
  type::pos *pos;        // position (x, y, z) and mass (w)
  type::vel_xy *vel_xy;  // velocity (x, y)
  type::vel_z *vel_z;  // velocity (z)
#endif                             //! defined(HOST_MALLOC_AND_FIRST_TOUCH)
  allocate_particles(&pos, &vel_xy, &vel_z, &idx, num);

  // initialize data on GPU
  set_uniform_sphere(num, pos, vel_xy, vel_z, idx, mass, radius, virial, newton);

  constexpr auto benchmark = [](const auto func) noexcept(false) {
    cudaDeviceSynchronize();  // Ensure all prior GPU work is complete
    struct timespec ini;
    clock_gettime(CLOCK_MONOTONIC, &ini);
    func();
    cudaDeviceSynchronize();  // Wait for GPU work to complete
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
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

  // prepare to use GPUDirect Storage via HDF5 with VFD
  auto fapl = H5Pcreate(H5P_FILE_ACCESS);
  H5Pset_fapl_gds(fapl, memb, fblk, cbuf);

  // create HDF5 file
  auto uuid = boost::uuids::random_generator{}();
  const auto series = boost::lexical_cast<std::string>(uuid);
  auto name = "dat/" + series + ".h5";
  auto target = H5Fcreate(name.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
  if (target < 0) {
    std::cerr << __FILE__ << "(" << __LINE__ << "): " << __func__ << ": ERROR: Failed to create HDF5 file: " << name << std::endl
              << std::flush;
    std::exit(EXIT_FAILURE);
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
  // execute H5Dwrite_multi()
  const auto elapse_write = benchmark([&h5write]() { h5write.execute(); });
  // write attribute
  util::hdf5::write_attr(hdf5_dataspace_1, target, "num", &num);
  // flush to ensure data is written to storage before timing read
  H5Fflush(target, H5F_SCOPE_GLOBAL);
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

  // drop filesystem cache for the file to ensure cold read
  drop_file_cache(name);
  
  // read the file and compare
  target = H5Fopen(name.c_str(), H5F_ACC_RDONLY, fapl);
  if (target < 0) {
    std::cerr << __FILE__ << "(" << __LINE__ << "): " << __func__ << ": ERROR: Failed to open HDF5 file: " << name << std::endl
              << std::flush;
    std::exit(EXIT_FAILURE);
  }
  auto num_read = std::remove_const_t<decltype(num)>{};
  util::hdf5::read_attr(target, "num", &num_read);
  if (num_read != num) {
    std::cerr << __FILE__ << "(" << __LINE__ << "): " << __func__ << ": ERROR: num_read (" << num_read << ") does not match with num (" << num << ")" << std::endl
              << std::flush;
    std::exit(EXIT_FAILURE);
  }
  std::remove_reference_t<decltype(*idx)> *idx_read = nullptr;        // particle ID
  std::remove_reference_t<decltype(*pos)> *pos_read = nullptr;        // position (x, y, z) and mass (w)
  std::remove_reference_t<decltype(*vel_xy)> *vel_xy_read = nullptr;  // velocity (x, y)
  std::remove_reference_t<decltype(*vel_z)> *vel_z_read = nullptr;    // velocity (z)
  allocate_particles(&pos_read, &vel_xy_read, &vel_z_read, &idx_read, num_read);
  // preparation for H5Dread_multi()
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
  // execute H5Dread_multi()
  // h5read.execute();
  const auto elapse_read = benchmark([&h5read]() { h5read.execute(); });

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
      output << "N";
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
    output << num;
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

  release_particles(pos, vel_xy, vel_z, idx);
  release_particles(pos_read, vel_xy_read, vel_z_read, idx_read);

  // clean up test file to prevent disk filling
  if (std::remove(name.c_str()) != 0) {
    std::cerr << "Warning: Failed to delete test file: " << name << std::endl;
  }

  std::exit(EXIT_SUCCESS);
}
