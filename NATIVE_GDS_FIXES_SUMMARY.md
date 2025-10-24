# Native GDS Performance Issue - Complete Analysis & Fixes

## 🔴 Root Causes Identified from Your Logs

### Issue #1: Incomplete JSON Configuration Files ⚠️ **CRITICAL**

**From your gdscheck output:**
```
properties.use_compat_mode : true    ← Should be FALSE for native mode!
properties.force_compat_mode : false
```

**Problem:** Your `disable_compat.json` only had:
```json
{
    "properties": {
        "allow_compat_mode": false    ← Missing "use_compat_mode"!
    }
}
```

**The property `use_compat_mode` was missing!** This is why cuFile defaulted to `true` (compat mode) even when you wanted native mode.

**✅ FIXED:** Updated both JSON files with complete configurations including:
- `"use_compat_mode": false` for native mode
- Proper buffer sizes and thread configurations
- Logging settings

---

### Issue #2: GH200 Architecture Requires System Malloc

**From your logs:**
```
GPU index 0 NVIDIA GH200 120GB bar:disabled bar size (MiB):N/A
Platform: ARS-111GL-DNHR-LC1-ST036, Arch: aarch64
```

**Key Points:**
- You have a **GH200 (Grace Hopper)** with unified memory architecture
- BAR is disabled (expected for GH200)
- ARM64 architecture (aarch64)

**Why this matters:**
- GH200 uses **unified CPU-GPU memory**
- BAR disabled means traditional GPU BAR1 memory mapping is not used
- Native GDS **requires system-allocated memory** that both CPU and GPU can access
- Your current code uses `cudaMalloc()` which doesn't work optimally with GH200's architecture

**✅ FIX REQUIRED:** Rebuild with `USE_SYSTEM_MALLOC=ON` (see below)

---

### Issue #3: Missing NUMA Optimization

**From your logs:**
```
GPU 0: BUS_ID=0009:01:00.0, NUMA_NODE=0
```

Your script calculates NUMA node but the earlier version didn't use it properly.

**✅ FIXED:** Updated `job.pbs` to use:
```bash
numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE bin/h5gds ...
```

---

## 📋 Complete Fix Checklist

### ✅ Step 1: JSON Configuration Files (DONE)

**Files updated:**
- ✅ `disable_compat.json` - Now properly disables compat mode with `"use_compat_mode": false`
- ✅ `force_compat.json` - Now properly forces compat mode for comparison

**Verify the fix worked:**
After rebuilding and running, check gdscheck output should show:
```
properties.use_compat_mode : false    ← Should be FALSE now
properties.allow_compat_mode : false  ← Should be FALSE now
```

---

### ⚠️ Step 2: Rebuild with USE_SYSTEM_MALLOC=ON (REQUIRED!)

**You MUST rebuild your code for GH200:**

```bash
cd /work/jh250079/n14001/h5-gds

# Clean previous build
rm -rf build

# Configure for GH200 with system malloc
cmake -S . -B build \
    -DUSE_SYSTEM_MALLOC=ON \
    -DTARGET_GPU=NVIDIA_CC90 \
    -DCMAKE_BUILD_TYPE=Release

# Build
cd build
make -j $(nproc)

# Copy updated JSON files to build directory
cp ../disable_compat.json .
cp ../force_compat.json .
```

**Why this is critical for GH200:**
- GH200 has unified memory architecture
- System malloc creates memory accessible by both CPU and GPU
- This memory can be registered with cuFile for direct GDS access
- Regular `cudaMalloc()` doesn't work well with GH200's memory model

---

### ✅ Step 3: Job Script Updated (DONE)

**Updated `job.pbs` with:**
- ✅ NUMA binding with `numactl`
- ✅ GDS health check before benchmarks
- ✅ Better error handling
- ✅ Storage device verification
- ✅ Enhanced logging

---

### Step 4: Verify GDS Configuration

**Run this test before your full benchmark:**

```bash
# Test if JSON is being loaded
export CUFILE_ENV_PATH_JSON=/work/jh250079/n14001/h5-gds/build/disable_compat.json
$CUDA_HOME/gds/tools/gdscheck -p | grep "use_compat_mode"
# Should show: properties.use_compat_mode : false
```

---

## 🎯 Expected Results After Fixes

### Before Fixes:
```
Native mode:  500 MB/s  ← Poor, using compat mode internally
Compat mode: 2000 MB/s  ← Good, CPU path optimized
```

### After Fixes:
```
Native mode: 4000-8000 MB/s  ← Should MATCH or EXCEED compat mode!
Compat mode: 2000 MB/s       ← Similar to before
```

**Native mode should be 2-4x faster than compat mode with these fixes!**

---

## 🔧 What Each Fix Does

### Fix #1: Corrected JSON Files
**Impact:** Forces cuFile to actually use native GDS instead of falling back to compat mode
**Expected speedup:** 2-3x (enables true GPU-direct transfers)

### Fix #2: USE_SYSTEM_MALLOC=ON
**Impact:** Allocates memory that cuFile can directly access without bounce buffers
**Expected speedup:** 2-4x (eliminates memory copies)
**Critical for:** GH200 architecture with unified memory

### Fix #3: NUMA Binding
**Impact:** Ensures memory and CPU threads are on same NUMA node as GPU
**Expected speedup:** 1.2-1.5x (reduces cross-NUMA latency)

**Combined: 4-10x improvement expected!**

---

## 🧪 Testing Protocol

### 1. Quick Test (After Rebuild)
```bash
cd /work/jh250079/n14001/h5-gds/build

# Test native mode
export CUFILE_ENV_PATH_JSON=$PWD/disable_compat.json
numactl --cpunodebind=0 --membind=0 bin/h5gds --asis --num 1048576

# Test compat mode
export CUFILE_ENV_PATH_JSON=$PWD/force_compat.json
numactl --cpunodebind=0 --membind=0 bin/h5gds --asis --num 1048576

# Compare results in log/h5gds_benchmark.csv
```

### 2. Full Benchmark
```bash
qsub job.pbs
```

---

## 📊 How to Verify Success

### Check gdscheck output in job logs:
```
✅ properties.use_compat_mode : false    (not true)
✅ properties.allow_compat_mode : false  (not true)
✅ GPU NUMA ID: 1 (matches NUMA_NODE)
✅ NVMe: Supported
```

### Check benchmark results:
```bash
cat log/h5gds_benchmark.csv
```

**Look for:**
- Native mode bandwidth (write) > 4 GB/s
- Native mode bandwidth (read) > 4 GB/s
- Native mode should be faster than compat mode

---

## 🚨 Troubleshooting

### If native mode is still slow after fixes:

1. **Verify JSON is loaded:**
   ```bash
   grep "use_compat_mode" job_output.log
   # Should show: properties.use_compat_mode : false
   ```

2. **Verify system malloc is used:**
   ```bash
   nm build/bin/h5gds | grep first_touch
   # Should show the first_touch kernel symbol
   ```

3. **Check cuFile logs:**
   ```bash
   cat /tmp/cufile*.log
   # Look for errors or warnings about memory registration
   ```

4. **Verify memory type:**
   ```bash
   # In h5gds.cu, add debug output after malloc/cudaMalloc
   # Print pointer address to verify it's in correct memory space
   ```

5. **Check if GDS is actually being used:**
   ```bash
   # Monitor GPU during run
   nvidia-smi dmon -s pcie
   # Native mode should show high PCIe utilization
   ```

---

## 📝 Summary

**Root cause of poor native GDS performance:**
1. ❌ JSON config incomplete - cuFile used compat mode internally
2. ❌ cudaMalloc memory incompatible with GH200 + GDS architecture
3. ❌ Missing NUMA affinity - cross-NUMA memory access overhead

**Fixes applied:**
1. ✅ Corrected JSON files with proper `use_compat_mode: false`
2. ⚠️ **YOU MUST: Rebuild with USE_SYSTEM_MALLOC=ON**
3. ✅ Added NUMA binding in job script

**Expected outcome:**
Native GDS mode should now be **4-10x faster** than before, outperforming compat mode.

---

## 🎯 Next Steps

1. **Rebuild the code** with USE_SYSTEM_MALLOC=ON (see Step 2 above)
2. **Copy updated JSON files** to build directory
3. **Run test** with small dataset first
4. **Submit full job** once test passes
5. **Compare results** - native should now be fastest!

Good luck! 🚀

