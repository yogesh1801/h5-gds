cmake_minimum_required(VERSION 3.20) # Boost warning about new version can be suppressed in 3.20+

# global setting(s)
set(USER_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

# global options
option(BENCHMARK_MODE "On to perform benchmark" ON)
message(STATUS "BENCHMARK_MODE is ${BENCHMARK_MODE}")
set(FIRST_TOUCH_MODE "NONE" CACHE STRING "Memory allocation strategy: NONE (cudaMalloc), GPU (host malloc + GPU first-touch), or CPU (host malloc + CPU first-touch)")
set_property(CACHE FIRST_TOUCH_MODE PROPERTY STRINGS "NONE" "GPU" "CPU")
message(STATUS "FIRST_TOUCH_MODE is ${FIRST_TOUCH_MODE}")

# enforce Release as build type in benchmark mode
if(BENCHMARK_MODE)
  set(CMAKE_BUILD_TYPE "Release")
endif(BENCHMARK_MODE)

# check loaded modulefiles
find_program(MODULECMD_CONFIG "modulecmd")

if(MODULECMD_CONFIG)
  get_filename_component(SH_NAME "$ENV{SHELL}" NAME)
  execute_process(COMMAND ${MODULECMD_CONFIG} ${SH_NAME} list OUTPUT_VARIABLE LOADED_MODULE_LIST)
  message(STATUS "${LOADED_MODULE_LIST}")
endif(MODULECMD_CONFIG)

# set C/CXX compilers
project(dummy NONE)

# set(CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER_INIT})
# set(CMAKE_C_COMPILER ${CMAKE_C_COMPILER_INIT})
# message(STATUS "CMAKE_CXX_COMPILER is ${CMAKE_CXX_COMPILER}")
# message(STATUS "CMAKE_C_COMPILER is ${CMAKE_C_COMPILER}")

# specify CXX standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED_ON)
set(CMAKE_CXX_EXTENSIONS OFF) # use -std=c++17 instead of -std=gnu++17

# specify options for Makefile
set(CMAKE_VERBOSE_MAKEFILE TRUE)
set(CMAKE_COLOR_MAKEFILE ON)

# set directory for executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# generate directories
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/dat)
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/log)

file(COPY sh DESTINATION ${CMAKE_BINARY_DIR})
file(COPY jl DESTINATION ${CMAKE_BINARY_DIR})
file(COPY disable_compat.json DESTINATION ${CMAKE_BINARY_DIR})
file(COPY force_compat.json DESTINATION ${CMAKE_BINARY_DIR})

# preparation for finding library path
string(REPLACE ":" ";" SEARCH_PATH $ENV{CPATH})

# compile benchmark code: GPUDirect Storage via HDF5
project(h5gds CXX CUDA)
message(STATUS "Generate ${PROJECT_NAME} ...")

# define the executable
set(SOURCES ${PROJECT_SOURCE_DIR}/src/h5gds.cu ${PROJECT_SOURCE_DIR}/src/allocate.cu ${PROJECT_SOURCE_DIR}/src/generate.cu ${PROJECT_SOURCE_DIR}/src/compute.cu)

add_executable(${PROJECT_NAME}
  ${SOURCES}
)

find_package(CUDA REQUIRED)
find_package(CUDA_samples REQUIRED)

# apply common settings
include(${USER_MODULE_PATH}/apply_common_settings.cmake)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
  ${CUDA_samples_INCLUDE_DIRS}
)


# Configure first-touch memory allocation mode
if(FIRST_TOUCH_MODE STREQUAL "GPU")
  target_compile_definitions(${PROJECT_NAME} PUBLIC HOST_MALLOC_AND_FIRST_TOUCH_GPU)
  message(STATUS "Using host malloc with GPU first-touch")
elseif(FIRST_TOUCH_MODE STREQUAL "CPU")
  target_compile_definitions(${PROJECT_NAME} PUBLIC HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  message(STATUS "Using host malloc with CPU first-touch")
else()
  message(STATUS "Using standard cudaMalloc")
endif()
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${NVIDIA_GPU_HARD})

# compile benchmark code: GPUDirect Storage with binary files
project(bingds CXX CUDA)
message(STATUS "Generate ${PROJECT_NAME} ...")

# define the executable
set(SOURCES ${PROJECT_SOURCE_DIR}/src/bingds.cu ${PROJECT_SOURCE_DIR}/src/allocate.cu ${PROJECT_SOURCE_DIR}/src/generate.cu)

add_executable(${PROJECT_NAME}
  ${SOURCES}
)

find_package(CUDA REQUIRED)
find_package(CUDA_samples REQUIRED)

# apply common settings
include(${USER_MODULE_PATH}/apply_common_settings.cmake)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
  ${CUDA_samples_INCLUDE_DIRS}
)

# Link cuFile library
find_library(CUFILE_LIBRARY cufile HINTS /usr/local/cuda/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/lib64 ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)
if(NOT CUFILE_LIBRARY)
  message(WARNING "cuFile library not found. Binary GDS benchmark may not link properly.")
else()
  message(STATUS "Found cuFile library: ${CUFILE_LIBRARY}")
  target_link_libraries(${PROJECT_NAME} PRIVATE ${CUFILE_LIBRARY})
endif()

# Configure first-touch memory allocation mode (same as h5gds)
if(FIRST_TOUCH_MODE STREQUAL "GPU")
  target_compile_definitions(${PROJECT_NAME} PUBLIC HOST_MALLOC_AND_FIRST_TOUCH_GPU)
  message(STATUS "bingds: Using host malloc with GPU first-touch")
elseif(FIRST_TOUCH_MODE STREQUAL "CPU")
  target_compile_definitions(${PROJECT_NAME} PUBLIC HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  message(STATUS "bingds: Using host malloc with CPU first-touch")
else()
  message(STATUS "bingds: Using standard cudaMalloc")
endif()
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${NVIDIA_GPU_HARD})

# compile benchmark code: GPUDirect Storage via NetCDF-4
project(nc4gds CXX CUDA)
message(STATUS "Generate ${PROJECT_NAME} ...")

# define the executable
set(SOURCES ${PROJECT_SOURCE_DIR}/src/nc4gds.cu ${PROJECT_SOURCE_DIR}/src/allocate.cu ${PROJECT_SOURCE_DIR}/src/generate.cu)

add_executable(${PROJECT_NAME}
  ${SOURCES}
)

find_package(CUDA REQUIRED)
find_package(CUDA_samples REQUIRED)

# apply common settings
include(${USER_MODULE_PATH}/apply_common_settings.cmake)

target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
  ${CUDA_samples_INCLUDE_DIRS}
)

# Configure first-touch memory allocation mode (same as h5gds)
if(FIRST_TOUCH_MODE STREQUAL "GPU")
  target_compile_definitions(${PROJECT_NAME} PUBLIC HOST_MALLOC_AND_FIRST_TOUCH_GPU)
  message(STATUS "nc4gds: Using host malloc with GPU first-touch")
elseif(FIRST_TOUCH_MODE STREQUAL "CPU")
  target_compile_definitions(${PROJECT_NAME} PUBLIC HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  message(STATUS "nc4gds: Using host malloc with CPU first-touch")
else()
  message(STATUS "nc4gds: Using standard cudaMalloc")
endif()
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${NVIDIA_GPU_HARD})
