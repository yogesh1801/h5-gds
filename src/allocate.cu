///
/// @file allocate.cu
/// @author Yohei MIKI (The University of Tokyo)
/// @brief memory allocation
///
/// @copyright Copyright (c) 2023 Information Technology Center, The University of Tokyo
///
/// The MIT License is applied to this software, see LICENSE
///
#include <cuda.h>
#include <curand_mtgp32.h>  // defines THREAD_NUM
#include <helper_cuda.h>    // use checkCudaErrors()

#include <limits>       // std::numeric_limits
#include <type_traits>  // std::remove_reference_t

#include "allocate.cuh"
#include "common.cuh"  // NTHREADS

constexpr auto round_up(const size_t org, const size_t unit) {
  const size_t mod = org % unit;
  return ((mod == 0) ? org : (org + unit - mod));
}

#if defined(HOST_MALLOC_AND_FIRST_TOUCH)
__global__ void first_touch(type::pos *const pos, type::vel_xy *const vel_xy, type::vel_z *const vel_z, type::idx *const idx, const type::idx num) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    pos[i] = type::pos{0.0F, 0.0F, 0.0F, 0.0F};
    vel_xy[i] = type::vel_xy{0.0F, 0.0F};
    vel_z[i] = type::vel_z{0.0F};
    idx[i] = std::numeric_limits<type::idx>::min();
  }
}
#endif  // defined(HOST_MALLOC_AND_FIRST_TOUCH)

void allocate_particles(type::pos **pos, type::vel_xy **vel_xy, type::vel_z **vel_z, type::idx **idx, const type::idx num) noexcept(false) {
  auto size = round_up(num, NTHREADS);
  size = round_up(size, THREAD_NUM);  // for Mersenne Twister

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH)
  checkCudaErrors(cudaMalloc((void **)pos, size * sizeof(std::remove_reference_t<decltype(**pos)>)));
  checkCudaErrors(cudaMalloc((void **)vel_xy, size * sizeof(std::remove_reference_t<decltype(**vel_xy)>)));
  checkCudaErrors(cudaMalloc((void **)vel_z, size * sizeof(std::remove_reference_t<decltype(**vel_z)>)));
  checkCudaErrors(cudaMalloc((void **)idx, size * sizeof(std::remove_reference_t<decltype(**idx)>)));

  // zero-clear arrays (for safety of massless particles)
  checkCudaErrors(cudaMemset(*pos, 0, size * sizeof(std::remove_reference_t<decltype(**pos)>)));
  checkCudaErrors(cudaMemset(*vel_xy, 0, size * sizeof(std::remove_reference_t<decltype(**vel_xy)>)));
  checkCudaErrors(cudaMemset(*vel_z, 0, size * sizeof(std::remove_reference_t<decltype(**vel_z)>)));
  checkCudaErrors(cudaMemset(*idx, 0, size * sizeof(std::remove_reference_t<decltype(**idx)>)));
#elif defined(USE_MANAGED_MEMORY)
  checkCudaErrors(cudaMallocManaged((void **)pos, size * sizeof(std::remove_reference_t<decltype(**pos)>)));
  checkCudaErrors(cudaMallocManaged((void **)vel_xy, size * sizeof(std::remove_reference_t<decltype(**vel_xy)>)));
  checkCudaErrors(cudaMallocManaged((void **)vel_z, size * sizeof(std::remove_reference_t<decltype(**vel_z)>)));
  checkCudaErrors(cudaMallocManaged((void **)idx, size * sizeof(std::remove_reference_t<decltype(**idx)>)));

  // zero-clear arrays
  checkCudaErrors(cudaMemset(*pos, 0, size * sizeof(std::remove_reference_t<decltype(**pos)>)));
  checkCudaErrors(cudaMemset(*vel_xy, 0, size * sizeof(std::remove_reference_t<decltype(**vel_xy)>)));
  checkCudaErrors(cudaMemset(*vel_z, 0, size * sizeof(std::remove_reference_t<decltype(**vel_z)>)));
  checkCudaErrors(cudaMemset(*idx, 0, size * sizeof(std::remove_reference_t<decltype(**idx)>)));

  // Prefetch to GPU device to ensure physical backing on HBM
  int device = 0;
  checkCudaErrors(cudaGetDevice(&device));
  checkCudaErrors(cudaMemPrefetchAsync(*pos, size * sizeof(std::remove_reference_t<decltype(**pos)>), device, NULL));
  checkCudaErrors(cudaMemPrefetchAsync(*vel_xy, size * sizeof(std::remove_reference_t<decltype(**vel_xy)>), device, NULL));
  checkCudaErrors(cudaMemPrefetchAsync(*vel_z, size * sizeof(std::remove_reference_t<decltype(**vel_z)>), device, NULL));
  checkCudaErrors(cudaMemPrefetchAsync(*idx, size * sizeof(std::remove_reference_t<decltype(**idx)>), device, NULL));
  
  // Synchronize to ensure prefetching is complete before use
  checkCudaErrors(cudaDeviceSynchronize());

#else   //! defined(HOST_MALLOC_AND_FIRST_TOUCH)
  *pos = (type::pos *)malloc(size * sizeof(std::remove_reference_t<decltype(**pos)>));
  if (*pos == nullptr) throw std::bad_alloc();
  *vel_xy = (type::vel_xy *)malloc(size * sizeof(std::remove_reference_t<decltype(**vel_xy)>));
  if (*vel_xy == nullptr) throw std::bad_alloc();
  *vel_z = (type::vel_z *)malloc(size * sizeof(std::remove_reference_t<decltype(**vel_z)>));
  if (*vel_z == nullptr) throw std::bad_alloc();
  *idx = (type::idx *)malloc(size * sizeof(std::remove_reference_t<decltype(**idx)>));
  if (*idx == nullptr) throw std::bad_alloc();
  first_touch<<<(size + NTHREADS - 1) / NTHREADS, NTHREADS>>>(*pos, *vel_xy, *vel_z, *idx, size);
  checkCudaErrors(cudaDeviceSynchronize());
#endif  //! defined(HOST_MALLOC_AND_FIRST_TOUCH)
}

void release_particles(type::pos *pos, type::vel_xy *vel_xy, type::vel_z *vel_z, type::idx *idx) noexcept(false) {
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH)
  checkCudaErrors(cudaFree(pos));
  checkCudaErrors(cudaFree(vel_xy));
  checkCudaErrors(cudaFree(vel_z));
  checkCudaErrors(cudaFree(idx));
#else   //! defined(HOST_MALLOC_AND_FIRST_TOUCH)
  free(pos);
  free(vel_xy);
  free(vel_z);
  free(idx);
#endif  //! defined(HOST_MALLOC_AND_FIRST_TOUCH)
}
