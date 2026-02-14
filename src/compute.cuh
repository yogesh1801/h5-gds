///
/// @file compute.cuh
/// @brief synthetic compute kernel for benchmarking memory access patterns
///
/// Kinetic energy reduction over particle data. Used to measure compute
/// performance under different allocation strategies (cudaMalloc, first-touch
/// GPU, first-touch CPU).
///
#ifndef COMPUTE_CUH
#define COMPUTE_CUH

#include "common.cuh"

///
/// @brief Run one kinetic energy reduction step (for benchmarking)
///
/// Caller allocates d_total (1 float), zeros it, calls this, then syncs.
/// This does: cudaMemset(d_total,0) + kernel launch + cudaDeviceSynchronize().
///
/// @param[in] num number of particles
/// @param[in] pos position and mass (pos[].w)
/// @param[in] vel_xy velocity (x, y)
/// @param[in] vel_z velocity (z)
/// @param[in,out] d_total device buffer (1 float) for reduction result, must be zeroed before call
///
void compute_kinetic_energy_step(const type::idx num, const type::pos* pos, const type::vel_xy* vel_xy,
                                 const type::vel_z* vel_z, float* d_total);

#endif  // COMPUTE_CUH
