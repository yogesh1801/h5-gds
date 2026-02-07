# Try to find NetCDF-C library
# search include directory
find_path(NETCDF_INCLUDE_DIR netcdf.h
  PATHS
  ENV NETCDF_DIR
  ENV NETCDF_INC
  ENV CPATH
  ${NETCDF_DIR}
  ${NETCDF_INC}
  /usr/include
  /usr/local/include
  PATH_SUFFIXES
  include
)
set(NETCDF_INCLUDE_DIRS ${NETCDF_INCLUDE_DIR})

# search library path
find_library(NETCDF_LIBRARY
  NAMES
  netcdf
  PATHS
  ENV NETCDF_DIR
  ENV NETCDF_LIB
  ENV LD_LIBRARY_PATH
  ${NETCDF_DIR}
  ${NETCDF_LIB}
  /usr/lib
  /usr/lib64
  /usr/local/lib
  /usr/local/lib64
  PATH_SUFFIXES
  lib
  lib64
)
set(NETCDF_LIBRARIES ${NETCDF_LIBRARY})

# hide variables except in advanced mode
mark_as_advanced(NETCDF_INCLUDE_DIR NETCDF_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NetCDF
  REQUIRED_VARS
  NETCDF_INCLUDE_DIRS
  NETCDF_LIBRARIES
)
