///
/// @file compute.cu
/// @brief kinetic energy reduction kernel
///
#include <helper_cuda.h>

#include "common.cuh"
#include "compute.cuh"
#include "cudalib.cuh"

__global__ void compute_ke_kernel(const type::idx num, const type::pos* __restrict__ pos,
                                  const type::vel_xy* __restrict__ vel_xy,
                                  const type::vel_z* __restrict__ vel_z, float* __restrict__ total_ke) {
  __shared__ float s_sum[512];
  const int tid = threadIdx.x;
  float my_ke = 0.0f;

  for (type::idx i = blockIdx.x * blockDim.x + tid; i < num; i += blockDim.x * gridDim.x) {
    const float m = pos[i].w;
    const float vx = vel_xy[i].x;
    const float vy = vel_xy[i].y;
    const float vz = vel_z[i];
    my_ke += 0.5f * m * (vx * vx + vy * vy + vz * vz);
  }

  s_sum[tid] = my_ke;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s_sum[tid] += s_sum[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    atomicAdd(total_ke, s_sum[0]);
  }
}

void compute_kinetic_energy_step(const type::idx num, const type::pos* pos, const type::vel_xy* vel_xy,
                                 const type::vel_z* vel_z, float* d_total) {
  checkCudaErrors(cudaMemset(d_total, 0, sizeof(float)));

  const int block_size = 512;
  const int num_blocks = (static_cast<int>(num) + block_size - 1) / block_size;

  compute_ke_kernel<<<num_blocks, block_size>>>(num, pos, vel_xy, vel_z, d_total);
  checkCudaErrors(cudaDeviceSynchronize());
  getLastCudaError("compute_ke_kernel");
}
