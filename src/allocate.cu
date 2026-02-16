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

#include <cstdlib>      // posix_memalign
#include <iostream>     // std::cout, std::endl
#include <limits>       // std::numeric_limits
#include <type_traits>  // std::remove_reference_t

#include "allocate.cuh"
#include "common.cuh"  // NTHREADS

constexpr auto round_up(const size_t org, const size_t unit) {
  const size_t mod = org % unit;
  return ((mod == 0) ? org : (org + unit - mod));
}

#if defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU)
// GPU first touch: pages backed by the HBM GPU memory
__global__ void first_touch_gpu(type::pos* const pos, type::vel_xy* const vel_xy, type::vel_z* const vel_z, type::idx* const idx, const type::idx num) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    pos[i] = type::pos{0.0F, 0.0F, 0.0F, 0.0F};
    vel_xy[i] = type::vel_xy{0.0F, 0.0F};
    vel_z[i] = type::vel_z{0.0F};
    idx[i] = std::numeric_limits<type::idx>::min();
  }
}
#endif  // defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU)

#if defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
// CPU first touch: pages backed by the CPU memory
void first_touch_cpu(type::pos* const pos, type::vel_xy* const vel_xy, type::vel_z* const vel_z, type::idx* const idx, const type::idx num) {
  for (type::idx i = 0; i < num; i++) {
    pos[i] = type::pos{0.0F, 0.0F, 0.0F, 0.0F};
    vel_xy[i] = type::vel_xy{0.0F, 0.0F};
    vel_z[i] = type::vel_z{0.0F};
    idx[i] = std::numeric_limits<type::idx>::min();
  }
}
#endif  // defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)

void allocate_particles(type::pos** pos, type::vel_xy** vel_xy, type::vel_z** vel_z, type::idx** idx, const type::idx num, const bool page_align) noexcept(false) {
  auto size = round_up(num, NTHREADS);
  size = round_up(size, THREAD_NUM);  // for Mersenne Twister

#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // Standard GPU allocation with cudaMalloc
  checkCudaErrors(cudaMalloc((void**)pos, size * sizeof(std::remove_reference_t<decltype(**pos)>)));
  checkCudaErrors(cudaMalloc((void**)vel_xy, size * sizeof(std::remove_reference_t<decltype(**vel_xy)>)));
  checkCudaErrors(cudaMalloc((void**)vel_z, size * sizeof(std::remove_reference_t<decltype(**vel_z)>)));
  checkCudaErrors(cudaMalloc((void**)idx, size * sizeof(std::remove_reference_t<decltype(**idx)>)));

  // zero-clear arrays (for safety of massless particles)
  checkCudaErrors(cudaMemset(*pos, 0.0F, size * sizeof(std::remove_reference_t<decltype(**pos)>)));
  checkCudaErrors(cudaMemset(*vel_xy, 0.0F, size * sizeof(std::remove_reference_t<decltype(**vel_xy)>)));
  checkCudaErrors(cudaMemset(*vel_z, 0.0F, size * sizeof(std::remove_reference_t<decltype(**vel_z)>)));
  checkCudaErrors(cudaMemset(*idx, std::numeric_limits<std::remove_reference_t<decltype(**idx)>>::min(), size * sizeof(std::remove_reference_t<decltype(**idx)>)));
#else
  // Host malloc for unified memory (Grace Hopper)
  constexpr size_t PAGE_SIZE = 4096;
  if (page_align) {
    if (posix_memalign((void**)pos, PAGE_SIZE, size * sizeof(std::remove_reference_t<decltype(**pos)>)) != 0 ||
        posix_memalign((void**)vel_xy, PAGE_SIZE, size * sizeof(std::remove_reference_t<decltype(**vel_xy)>)) != 0 ||
        posix_memalign((void**)vel_z, PAGE_SIZE, size * sizeof(std::remove_reference_t<decltype(**vel_z)>)) != 0 ||
        posix_memalign((void**)idx, PAGE_SIZE, size * sizeof(std::remove_reference_t<decltype(**idx)>)) != 0) {
      std::cerr << "Failed to allocate page-aligned particle buffers" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    *pos = (type::pos*)malloc(size * sizeof(std::remove_reference_t<decltype(**pos)>));
    *vel_xy = (type::vel_xy*)malloc(size * sizeof(std::remove_reference_t<decltype(**vel_xy)>));
    *vel_z = (type::vel_z*)malloc(size * sizeof(std::remove_reference_t<decltype(**vel_z)>));
    *idx = (type::idx*)malloc(size * sizeof(std::remove_reference_t<decltype(**idx)>));
  }

#if defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU)
  // GPU first touch: trigger page allocation on GPU
  std::cout << "GPU First touch kernel running" << std::endl;
  first_touch_gpu<<<(size + NTHREADS - 1) / NTHREADS, NTHREADS>>>(*pos, *vel_xy, *vel_z, *idx, size);
  checkCudaErrors(cudaDeviceSynchronize());
  std::cout << "GPU First touch kernel done" << std::endl;
#elif defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // CPU first touch: trigger page allocation on CPU
  std::cout << "CPU First touch kernel running" << std::endl;
  first_touch_cpu(*pos, *vel_xy, *vel_z, *idx, size);
  std::cout << "CPU First touch kernel done" << std::endl;
#endif
#endif
}

void release_particles(type::pos* pos, type::vel_xy* vel_xy, type::vel_z* vel_z, type::idx* idx) noexcept(false) {
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // Standard GPU deallocation
  checkCudaErrors(cudaFree(pos));
  checkCudaErrors(cudaFree(vel_xy));
  checkCudaErrors(cudaFree(vel_z));
  checkCudaErrors(cudaFree(idx));
#else
  // Host deallocation (both GPU and CPU first-touch modes use malloc)
  free(pos);
  free(vel_xy);
  free(vel_z);
  free(idx);
#endif
}

// ============================================
// NetCDF-compatible particle layout (Nx3 position, Nx3 velocity, N mass, N id)
// ============================================

#if defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU)
// GPU first touch kernel for NetCDF buffers
__global__ void first_touch_netcdf_gpu(float* const position, float* const velocity, float* const mass, type::idx* const id, const type::idx num) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    // Touch position (Nx3 layout)
    position[i * 3 + 0] = 0.0F;
    position[i * 3 + 1] = 0.0F;
    position[i * 3 + 2] = 0.0F;
    // Touch velocity (Nx3 layout)
    velocity[i * 3 + 0] = 0.0F;
    velocity[i * 3 + 1] = 0.0F;
    velocity[i * 3 + 2] = 0.0F;
    // Touch mass and id
    mass[i] = 0.0F;
    id[i] = std::numeric_limits<type::idx>::min();
  }
}
#endif

#if defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
// CPU first touch for NetCDF buffers
void first_touch_netcdf_cpu(float* const position, float* const velocity, float* const mass, type::idx* const id, const type::idx num) {
  for (type::idx i = 0; i < num; i++) {
    position[i * 3 + 0] = 0.0F;
    position[i * 3 + 1] = 0.0F;
    position[i * 3 + 2] = 0.0F;
    velocity[i * 3 + 0] = 0.0F;
    velocity[i * 3 + 1] = 0.0F;
    velocity[i * 3 + 2] = 0.0F;
    mass[i] = 0.0F;
    id[i] = std::numeric_limits<type::idx>::min();
  }
}
#endif

void allocate_particles_netcdf(float** position, float** velocity, float** mass, type::idx** id, const type::idx num, const bool page_align) noexcept(false) {
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // Standard GPU allocation with cudaMalloc (same as allocate_particles)
  checkCudaErrors(cudaMalloc((void**)position, num * 3 * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)velocity, num * 3 * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)mass, num * sizeof(float)));
  checkCudaErrors(cudaMalloc((void**)id, num * sizeof(type::idx)));

  // zero-clear arrays
  checkCudaErrors(cudaMemset(*position, 0, num * 3 * sizeof(float)));
  checkCudaErrors(cudaMemset(*velocity, 0, num * 3 * sizeof(float)));
  checkCudaErrors(cudaMemset(*mass, 0, num * sizeof(float)));
  checkCudaErrors(cudaMemset(*id, 0, num * sizeof(type::idx)));
#else
  // Host malloc for unified memory (Grace Hopper)
  constexpr size_t PAGE_SIZE = 4096;
  if (page_align) {
    if (posix_memalign((void**)position, PAGE_SIZE, num * 3 * sizeof(float)) != 0 ||
        posix_memalign((void**)velocity, PAGE_SIZE, num * 3 * sizeof(float)) != 0 ||
        posix_memalign((void**)mass, PAGE_SIZE, num * sizeof(float)) != 0 ||
        posix_memalign((void**)id, PAGE_SIZE, num * sizeof(type::idx)) != 0) {
      std::cerr << "Failed to allocate page-aligned NetCDF particle buffers" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  } else {
    *position = (float*)malloc(num * 3 * sizeof(float));
    *velocity = (float*)malloc(num * 3 * sizeof(float));
    *mass = (float*)malloc(num * sizeof(float));
    *id = (type::idx*)malloc(num * sizeof(type::idx));
  }

  if (!*position || !*velocity || !*mass || !*id) {
    std::cerr << "Failed to allocate NetCDF particle buffers" << std::endl;
    std::exit(EXIT_FAILURE);
  }

#if defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU)
  // GPU first touch: trigger page allocation on GPU HBM
  std::cout << "GPU First touch for NetCDF buffers" << std::endl;
  first_touch_netcdf_gpu<<<(num + NTHREADS - 1) / NTHREADS, NTHREADS>>>(*position, *velocity, *mass, *id, num);
  checkCudaErrors(cudaDeviceSynchronize());
#elif defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // CPU first touch: trigger page allocation on CPU DDR
  std::cout << "CPU First touch for NetCDF buffers" << std::endl;
  first_touch_netcdf_cpu(*position, *velocity, *mass, *id, num);
#endif
#endif
}

void release_particles_netcdf(float* position, float* velocity, float* mass, type::idx* id) noexcept(false) {
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH_GPU) && !defined(HOST_MALLOC_AND_FIRST_TOUCH_CPU)
  // Standard GPU deallocation
  checkCudaErrors(cudaFree(position));
  checkCudaErrors(cudaFree(velocity));
  checkCudaErrors(cudaFree(mass));
  checkCudaErrors(cudaFree(id));
#else
  // Host deallocation
  free(position);
  free(velocity);
  free(mass);
  free(id);
#endif
}
