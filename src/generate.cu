///
/// @file generate.cu
/// @author Yohei MIKI (The University of Tokyo)
/// @brief generate initial-condition on GPU
///
/// @copyright Copyright (c) 2023 Information Technology Center, The University of Tokyo
///
/// The MIT License is applied to this software, see LICENSE
///
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#include <boost/math/constants/constants.hpp>  // boost::math::constants::pi
#include <type_traits>                         // std::remove_const_t

// use cuRAND
#include <curand_kernel.h>
#include <curand_mtgp32.h>            // defines THREAD_NUM and CURAND_NUM_MTGP32_PARAMS
#include <curand_mtgp32_host.h>       // use MTGP host helper functions
#include <curand_mtgp32dc_p_11213.h>  // use MTGP pre-computed parameter sets
#include <helper_cuda.h>              // use checkCudaErrors()

#include "common.cuh"
#include "cudalib.cuh"
#include "generate.cuh"

///
/// @brief Set the uniform sphere on device
///
/// @param[in] num number of N-body particles
/// @param[out] pos position and mass of N-body particles
/// @param[in] rad radius of the sphere
/// @param[in] Mtot total mass of the sphere
/// @param[out] vel velocity of N-body particles
/// @param[in] sig1d 1-dimensional velocity dispersion of the system
/// @param[in,out] state state of the Mersenne Twister
/// @param[in] offset offset for particle array
///
__global__ void set_uniform_sphere_dev(const type::idx num, type::pos *pos, const type::vel_z rad, const decltype(rad) Mtot, type::vel_xy *vel_xy, type::vel_z *vel_z, type::idx *id, const decltype(rad) sig1d, curandStateMtgp32 *state, const type::idx offset = 0) {
  const auto ii = offset + GLOBALIDX_X1D;
  
  // Early exit for out-of-bounds threads to prevent memory corruption
  if (ii >= num) return;
  
  const auto mass = Mtot / static_cast<decltype(Mtot)>(num);
  // solve the warp divergence if necessary
#if __CUDA_ARCH__ >= 700
  __syncwarp();
#endif  //__CUDA_ARCH__ >= 700

  // set particle position
  const auto rr = rad * std::cbrt(curand_uniform(&state[BLOCKIDX_X1D]));
  static constexpr auto one = static_cast<decltype(rr)>(1.0);
  static constexpr auto two = static_cast<decltype(rr)>(2.0);
  const auto prj = -one + two * curand_uniform(&state[BLOCKIDX_X1D]);
  const auto RR = rr * std::sqrt(one - prj * prj);
  const auto theta = two * boost::math::constants::pi<std::remove_const_t<decltype(RR)>>() * curand_uniform(&state[BLOCKIDX_X1D]);
  auto pi = std::remove_reference_t<decltype(*pos)>{};
  pi.x = RR * std::cos(theta);
  pi.y = RR * std::sin(theta);
  pi.z = rr * prj;
  pi.w = mass;
  pos[ii] = pi;

  // set particle velocity
  auto vi_xy = std::remove_reference_t<decltype(*vel_xy)>{};
  vi_xy.x = sig1d * curand_normal(&state[BLOCKIDX_X1D]);
  vi_xy.y = sig1d * curand_normal(&state[BLOCKIDX_X1D]);
  vel_xy[ii] = vi_xy;
  vel_z[ii] = sig1d * curand_normal(&state[BLOCKIDX_X1D]);

  // set particle ID
  id[ii] = ii;
}

void set_uniform_sphere(const type::idx num, type::pos *pos, type::vel_xy *vel_xy, type::vel_z *vel_z, type::idx *id, const type::vel_z Mtot, const decltype(Mtot) rad, const decltype(Mtot) virial, const decltype(Mtot) newton) noexcept(false) {
  curandStateMtgp32 *MTstate_dev;
  mtgp32_kernel_params *MTparam_dev;

  // set appropriate number of thread-blocks
  const auto Nblk_tot = BLOCKSIZE(num, THREAD_NUM);
  const auto Nblk = (Nblk_tot < CURAND_NUM_MTGP32_PARAMS) ? Nblk_tot : CURAND_NUM_MTGP32_PARAMS;

  // memory allocation for MT states
  checkCudaErrors(cudaMalloc((void **)&MTstate_dev, Nblk * sizeof(decltype(*MTstate_dev))));

  // initialize MT states (cudaSuccess = CURAND_STATUS_SUCCESS = 0)
  checkCudaErrors(cudaMalloc((void **)&MTparam_dev, sizeof(decltype(*MTparam_dev))));
  checkCudaErrors(curandMakeMTGP32Constants(mtgp32dc_params_fast_11213, MTparam_dev));
  checkCudaErrors(curandMakeMTGP32KernelState(MTstate_dev, mtgp32dc_params_fast_11213, MTparam_dev, Nblk, 5489));  // 5489 is seed (optimal value for 32-bit MT)

  // set appropriate velocity dispersion for the given Virial ratio
  const auto sigma = std::sqrt(static_cast<decltype(newton)>(1.2) * newton * Mtot * virial / rad);
  const auto sig1d = sigma / boost::math::constants::root_three<decltype(sigma)>();

  // generate uniform sphere
  checkCudaErrors(cudaFuncSetAttribute(set_uniform_sphere_dev, cudaFuncAttributePreferredSharedMemoryCarveout, 0));
  set_uniform_sphere_dev<<<Nblk, THREAD_NUM>>>(num, pos, rad, Mtot, vel_xy, vel_z, id, sig1d, MTstate_dev);
  getLastCudaError("set_uniform_sphere");
  auto Nrem = Nblk_tot - Nblk;
  auto offset = Nblk * THREAD_NUM;
  while (Nrem > 0) {
    const auto Nrun = (Nrem < CURAND_NUM_MTGP32_PARAMS) ? Nrem : CURAND_NUM_MTGP32_PARAMS;
    set_uniform_sphere_dev<<<Nrun, THREAD_NUM>>>(num, pos, rad, Mtot, vel_xy, vel_z, id, sig1d, MTstate_dev, offset);
    getLastCudaError("set_uniform_sphere_offset");
    offset += Nrun * THREAD_NUM;
    Nrem -= Nrun;
  }
  checkCudaErrors(cudaDeviceSynchronize());

  // release device memory
  checkCudaErrors(cudaFree(MTstate_dev));
  checkCudaErrors(cudaFree(MTparam_dev));
}
