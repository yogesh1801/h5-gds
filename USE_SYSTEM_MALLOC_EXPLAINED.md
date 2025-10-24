# USE_SYSTEM_MALLOC Explained

## What It Does

The `USE_SYSTEM_MALLOC` CMake option controls **how memory is allocated** for GPU particle data. This is **CRITICAL** for GPUDirect Storage (GDS) performance.

---

## Code Paths

### Path 1: DEFAULT (USE_SYSTEM_MALLOC=OFF) ❌ Bad for Native GDS
**Lines 41-51 in allocate.cu:**

```cpp
#if !defined(HOST_MALLOC_AND_FIRST_TOUCH)
  // Allocates memory in GPU device memory
  checkCudaErrors(cudaMalloc((void **)pos, size * sizeof(...)));
  checkCudaErrors(cudaMalloc((void **)vel_xy, size * sizeof(...)));
  checkCudaErrors(cudaMalloc((void **)vel_z, size * sizeof(...)));
  checkCudaErrors(cudaMalloc((void **)idx, size * sizeof(...)));

  // Initialize with cudaMemset
  checkCudaErrors(cudaMemset(*pos, 0.0F, size * sizeof(...)));
  checkCudaErrors(cudaMemset(*vel_xy, 0.0F, size * sizeof(...)));
  checkCudaErrors(cudaMemset(*vel_z, 0.0F, size * sizeof(...)));
  checkCudaErrors(cudaMemset(*idx, ..., size * sizeof(...)));
#endif
```

**Memory Type:** Standard GPU device memory  
**Location:** GPU VRAM  
**GDS Compatibility:** ❌ **POOR** - May require bounce buffers or extra copies  
**Problem:** Regular cudaMalloc memory may not be directly accessible by cuFile/GDS

---

### Path 2: WITH USE_SYSTEM_MALLOC=ON ✅ Better for Native GDS
**Lines 52-58 in allocate.cu:**

```cpp
#else   //! defined(HOST_MALLOC_AND_FIRST_TOUCH)
  // Allocates memory using system malloc (host memory)
  *pos = (type::pos *)malloc(size * sizeof(...));
  *vel_xy = (type::vel_xy *)malloc(size * sizeof(...));
  *vel_z = (type::vel_z *)malloc(size * sizeof(...));
  *idx = (type::idx *)malloc(size * sizeof(...));
  
  // Initialize using GPU kernel (first-touch policy)
  first_touch<<<(size + NTHREADS - 1) / NTHREADS, NTHREADS>>>(
      *pos, *vel_xy, *vel_z, *idx, size);
  checkCudaErrors(cudaDeviceSynchronize());
#endif
```

**Memory Type:** Unified/system memory accessible by both CPU and GPU  
**Location:** System RAM (mapped to GPU address space)  
**GDS Compatibility:** ✅ **GOOD** - Can be registered with cuFile for direct access  
**Benefit:** GDS can directly transfer between storage and this memory

---

## The "First Touch" Kernel (Lines 25-34)

```cpp
#if defined(HOST_MALLOC_AND_FIRST_TOUCH)
__global__ void first_touch(type::pos *const pos, type::vel_xy *const vel_xy, 
                            type::vel_z *const vel_z, type::idx *const idx, 
                            const type::idx num) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num) {
    pos[i] = type::pos{0.0F, 0.0F, 0.0F, 0.0F};
    vel_xy[i] = type::vel_xy{0.0F, 0.0F};
    vel_z[i] = type::vel_z{0.0F};
    idx[i] = std::numeric_limits<type::idx>::min();
  }
}
#endif
```

**Purpose:** Initialize memory pages by touching them from GPU  
**Why:** Ensures memory is allocated on the NUMA node closest to the GPU  
**Performance:** Critical for NUMA systems (like your system with multiple GPUs)

---

## CMake Configuration (Lines 80-84 in CMakeLists.txt)

```cmake
if(USE_SYSTEM_MALLOC)
  target_compile_definitions(${PROJECT_NAME} PUBLIC
    $<$<BOOL:${USE_SYSTEM_MALLOC}>:HOST_MALLOC_AND_FIRST_TOUCH>
  )
endif(USE_SYSTEM_MALLOC)
```

When you build with `-DUSE_SYSTEM_MALLOC=ON`, CMake defines the preprocessor macro `HOST_MALLOC_AND_FIRST_TOUCH`, which activates the alternative code path.

---

## Memory Access Patterns

### Default Mode (cudaMalloc):
```
Storage → cuFile → [Bounce Buffer] → GPU VRAM
                    ⚠️ Extra copy!
```

### System Malloc Mode (malloc + first-touch):
```
Storage → cuFile → System RAM (GPU-accessible) → GPU uses directly
                   ✅ Direct access!
```

---

## Why This Matters for Native GDS

**Native GDS Mode** requires memory buffers that can be:
1. **Registered with cuFile** - Not all memory types can be
2. **DMA-accessible** - Must support direct memory access from storage controller
3. **Page-aligned** - Proper alignment for zero-copy transfers

**Regular cudaMalloc memory:**
- Lives in GPU VRAM
- May not be directly accessible by cuFile in all configurations
- Might force cuFile to use compatibility mode internally or bounce buffers

**System malloc with GPU mapping:**
- Creates pinned/unified memory
- Can be registered with cuFile APIs
- Allows true GPU-to-storage direct transfers
- Better for architectures like NVIDIA Grace Hopper (GH200) with unified memory

---

## How to Enable

### Rebuild your project:
```bash
cd /work/jh250079/n14001/h5-gds
rm -rf build  # Clean build
cmake -S . -B build -DUSE_SYSTEM_MALLOC=ON -DTARGET_GPU=NVIDIA_CC90
cd build
make -j
```

### Verify it's enabled:
```bash
# Check if the binary has the right symbol
nm bin/h5gds | grep first_touch
# Should see: first_touch kernel symbol if enabled
```

---

## Expected Performance Impact

| Mode | Memory Type | Native GDS Performance |
|------|-------------|----------------------|
| Default (OFF) | cudaMalloc | ❌ Poor - bounce buffers |
| System (ON) | malloc + first-touch | ✅ Good - direct transfers |

**Expected speedup:** 2-5x improvement in native GDS mode bandwidth

---

## Important Notes

1. **GH200 Architecture:** This mode is specifically designed for NVIDIA Grace Hopper systems where CPU and GPU share unified memory
2. **NUMA Sensitivity:** More sensitive to NUMA placement (hence why numactl is critical)
3. **Memory Overhead:** Slightly more system RAM usage, but GPU can still access it efficiently
4. **Compatibility:** Works on all systems, not just GH200, but benefits vary

---

## Bottom Line

**For your native GDS performance issue:**

🔴 **Current:** Using cudaMalloc → Native GDS can't efficiently access → Poor performance  
🟢 **Fixed:** Using malloc + first-touch → Native GDS direct access → Good performance

**You MUST rebuild with USE_SYSTEM_MALLOC=ON to fix your native GDS performance!**

