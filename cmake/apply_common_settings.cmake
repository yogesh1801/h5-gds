# set CPU architecture
include(${USER_MODULE_PATH}/set_target_cpu.cmake)

# set GPU architecture
include(${USER_MODULE_PATH}/set_target_gpu.cmake)

# find OpenMP
find_package(OpenMP)

# set compilation flags
include(${USER_MODULE_PATH}/set_compile_flag.cmake)

# find Boost
set(BOOST_INCLUDEDIR ${SEARCH_PATH})

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)
  set(Boost_NO_WARN_NEW_VERSIONS ON)
endif(CMAKE_VERSION VERSION_GREATER_EQUAL 3.20)

set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_DEBUG_LIBS OFF)
set(Boost_USE_RELEASE_LIBS ON)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS program_options filesystem timer system)

# find HDF5 (for h5gds and nc4gds projects)
if(PROJECT_NAME STREQUAL "h5gds" OR PROJECT_NAME STREQUAL "nc4gds")
  enable_language(C)
  find_package(HDF5 REQUIRED COMPONENTS C)
  
  # find VFD for GDS
  find_package(HDF5VFD_GDS REQUIRED COMPONENTS C)
endif()

# find NetCDF (only for nc4gds project)
if(PROJECT_NAME STREQUAL "nc4gds")
  find_package(NetCDF REQUIRED)
endif()

# link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE
  ${Boost_LIBRARIES}
)

# Link HDF5 for h5gds and nc4gds projects
if(PROJECT_NAME STREQUAL "h5gds" OR PROJECT_NAME STREQUAL "nc4gds")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    ${HDF5_LIBRARIES}
    ${HDF5VFD_GDS_LIBRARIES}
  )
endif()

# Link NetCDF for nc4gds project
if(PROJECT_NAME STREQUAL "nc4gds")
  target_link_libraries(${PROJECT_NAME} PRIVATE
    ${NETCDF_LIBRARIES}
  )
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE
  # OpenMP
  $<$<AND:$<BOOL:${OpenMP_FOUND}>,$<NOT:$<CXX_COMPILER_ID:NVHPC>>>:${OpenMP_CXX_FLAGS}>

  # memory sanitizer
  $<$<BOOL:${USE_SANITIZER_ADDRESS}>:-fsanitize=address>
  $<$<BOOL:${USE_SANITIZER_LEAK}>:-fsanitize=leak>
  $<$<BOOL:${USE_SANITIZER_UNDEFINED}>:-fsanitize=undefined>
  $<$<BOOL:${USE_SANITIZER_THREAD}>:-fsanitize=thread>
)

# include directories
target_include_directories(${PROJECT_NAME} PRIVATE
  ${PROJECT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src
)
target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
  ${Boost_INCLUDE_DIRS}
)

# Include HDF5 for h5gds and nc4gds projects
if(PROJECT_NAME STREQUAL "h5gds" OR PROJECT_NAME STREQUAL "nc4gds")
  target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${HDF5_INCLUDE_DIRS}
    ${HDF5VFDS_GDS_INCLUDE_DIRS}
  )
endif()

# Include NetCDF for nc4gds project
if(PROJECT_NAME STREQUAL "nc4gds")
  target_include_directories(${PROJECT_NAME} SYSTEM PRIVATE
    ${NETCDF_INCLUDE_DIRS}
  )
endif()

# add definitions
target_compile_definitions(${PROJECT_NAME} PUBLIC
  $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
)
