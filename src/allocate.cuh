///
/// @file allocate.cuh
/// @author Yohei MIKI (The University of Tokyo)
/// @brief memory allocation
///
/// @copyright Copyright (c) 2023 Information Technology Center, The University of Tokyo
///
/// The MIT License is applied to this software, see LICENSE
///
#ifndef ALLOCATE_CUH
#define ALLOCATE_CUH

#include "common.cuh"

///
/// @brief allocate memory on GPU
///
/// @param[out] pos particle position
/// @param[out] vel_xy particle velocity (x and y)
/// @param[out] vel_z particle velocity (z)
/// @param[out] idx particle ID
/// @param[in] num number of particles
/// @param[in] align_bytes alignment in bytes (0=disabled; e.g. 4096 for 4KB, 65536 for 64KB; power of 2)
///
void allocate_particles(type::pos** pos, type::vel_xy** vel_xy, type::vel_z** vel_z, type::idx** idx, type::idx num, size_t align_bytes = 0);

///
/// @brief release memory on GPU
///
/// @param[in] pos particle position
/// @param[in] vel_xy particle velocity (x and y)
/// @param[in] vel_z particle velocity (z)
/// @param[in] idx particle ID
///
void release_particles(type::pos* pos, type::vel_xy* vel_xy, type::vel_z* vel_z, type::idx* idx);

///
/// @brief allocate memory for NetCDF-compatible particle layout (Nx3 position, Nx3 velocity, N mass, N id)
///
/// @param[out] position particle position (Nx3 contiguous: x0,y0,z0, x1,y1,z1, ...)
/// @param[out] velocity particle velocity (Nx3 contiguous: vx0,vy0,vz0, ...)
/// @param[out] mass particle mass (N elements)
/// @param[out] id particle ID (N elements)
/// @param[in] num number of particles
/// @param[in] align_bytes alignment in bytes (0=disabled; e.g. 4096 for 4KB, 65536 for 64KB; power of 2)
///
void allocate_particles_netcdf(float** position, float** velocity, float** mass, type::idx** id, type::idx num, size_t align_bytes = 0);

///
/// @brief release memory for NetCDF-compatible particle layout
///
void release_particles_netcdf(float* position, float* velocity, float* mass, type::idx* id);

#endif  // ALLOCATE_CUH
