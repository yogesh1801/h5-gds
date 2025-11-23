cmake_minimum_required(VERSION 3.20) # Boost warning about new version can be suppressed in 3.20+

# global setting(s)
set(USER_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules")

# global options
option(BENCHMARK_MODE "On to perform benchmark" ON)
message(STATUS "BENCHMARK_MODE is ${BENCHMARK_MODE}")
option(USE_SYSTEM_MALLOC "On to use system-allocated memory for GPU (NVIDIA GH200 or similar architectures)" OFF)
message(STATUS "USE_SYSTEM_MALLOC is ${USE_SYSTEM_MALLOC}")
option(USE_MANAGED_MEMORY "On to use CUDA Managed Memory (Recommended for GH200)" OFF)
message(STATUS "USE_MANAGED_MEMORY is ${USE_MANAGED_MEMORY}")

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
set(SOURCES ${PROJECT_SOURCE_DIR}/src/h5gds.cu ${PROJECT_SOURCE_DIR}/src/allocate.cu ${PROJECT_SOURCE_DIR}/src/generate.cu)

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

if(USE_SYSTEM_MALLOC)
  target_compile_definitions(${PROJECT_NAME} PUBLIC
  $<$<BOOL:${USE_SYSTEM_MALLOC}>:HOST_MALLOC_AND_FIRST_TOUCH>
  )
endif(USE_SYSTEM_MALLOC)

if(USE_MANAGED_MEMORY)
  target_compile_definitions(${PROJECT_NAME} PUBLIC USE_MANAGED_MEMORY)
endif(USE_MANAGED_MEMORY)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${NVIDIA_GPU_HARD})
