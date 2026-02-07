///
/// @file src/netcdf4.hpp
/// @brief utility functions for NetCDF-4
///
/// @copyright Copyright (c) 2023 Information Technology Center, The University of Tokyo
///
/// The MIT License is applied to this software, see LICENSE
///
#ifndef NETCDF4_HPP
#define NETCDF4_HPP

#include <netcdf.h>

#include <cstdlib>   // std::exit
#include <iostream>  // std::cerr
#include <string>    // std::string

///
/// @brief utility functions for NetCDF-4
///
namespace util::netcdf4 {

///
/// @brief error handler for NetCDF function
///
/// @param[in] status return status of NetCDF function
/// @param[in] file file name who calls the NetCDF function
/// @param[in] line number of line which calls the NetCDF function
/// @param[in] func name of function which calls the NetCDF function
///
inline void _check_nc(const int status, const char* file, const int32_t line, const char* func) noexcept(false) {
  if (status != NC_NOERR) {
    std::cerr << file << "(" << line << "): " << func << ": ERROR: NetCDF returns error: " << nc_strerror(status) << std::endl
              << std::flush;
    std::exit(EXIT_FAILURE);
  }
}

///
/// @brief macro to call error handler of NetCDF function
///
#define NC_CHECK(call) util::netcdf4::_check_nc((call), __FILE__, __LINE__, __func__)

}  // namespace util::netcdf4

#endif  // NETCDF4_HPP
