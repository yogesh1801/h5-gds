# GDS Benchmark Quick Start Guide

## Build Instructions

### For Standard GPUs (H100, A100, etc.)
```bash
cmake -S . -B build
cd build
make
```

### For GH200 (Grace-Hopper) Systems
```bash
cmake -S . -B build -DUSE_SYSTEM_MALLOC=ON
cd build
make
```

The `USE_SYSTEM_MALLOC` option enables the "first-touch" NUMA optimization which is critical for GH200 performance.

---

## Running Benchmarks

### Test 1: Native GDS Mode (Direct GPU-to-Storage)
```bash
export CUFILE_ENV_PATH_JSON=../disable_compat.json
./bin/h5gds --num 1048576
```

### Test 2: Compatibility Mode (CPU-mediated I/O)
```bash
export CUFILE_ENV_PATH_JSON=../force_compat.json
./bin/h5gds --num 1048576
```

### Test 3: With Custom Parameters
```bash
export CUFILE_ENV_PATH_JSON=../disable_compat.json
./bin/h5gds --num 16777216 --fblk 4096 --cbuf 16777216 --memb 65536
```

---

## Benchmark Parameters

| Parameter | Description | Default | Notes |
|-----------|-------------|---------|-------|
| `--num N` | Number of particles | 1024 | Scales dataset size |
| `--fblk SIZE` | File block size (bytes) | See code | Must be power of 2 |
| `--cbuf SIZE` | Copy buffer size (bytes) | See code | Must be multiple of `fblk` |
| `--memb SIZE` | Memory boundary (bytes) | See code | Alignment for GDS |
| `--asis` | Disable hyperslab mode | false | Uses structure-of-arrays |
| `--skip` | Skip data verification | false | Faster but risky |

---

## Running on GH200 with Optimal Settings

```bash
# Get GPU's NUMA node
GPU_ID=0
BUS_ID=$(nvidia-smi --format=csv,noheader --query-gpu=gpu_bus_id -i $GPU_ID | \
         awk -F ":" '{print "0000:" $2 ":" $3}' | tr '[:upper:]' '[:lower:]')
NUMA_NODE=$(cat /sys/bus/pci/devices/$BUS_ID/numa_node)

# Run with NUMA binding
export CUFILE_ENV_PATH_JSON=../disable_compat.json
timeout 60s numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
    ./bin/h5gds --num 16777216
```

---

## Interpreting Results

Results are saved to `log/h5gds_benchmark.csv`:

```csv
N,data size [byte],copy buffer size [byte],file block size [byte],memory boundary [byte],latency (write) [s],latency (read) [s],bandwidth (write) [byte/s],bandwidth (read) [s],filename
```

### Key Metrics

- **Bandwidth (write):** Higher is better - shows write throughput
- **Bandwidth (read):** Higher is better - shows read throughput
- **Latency:** Lower is better - time to complete operation

### Expected Results

**Native GDS vs Compatibility Mode:**
- Native should show 2-5x higher bandwidth for large datasets
- Native latency should be significantly lower
- Benefits increase with dataset size

**GH200 with vs without `USE_SYSTEM_MALLOC`:**
- With first-touch should show 10-30% better performance
- More pronounced for memory-bound operations

---

## Verification Pre-flight Checks

### 1. Verify GDS Support
```bash
$CUDA_DIR/gds/tools/gdscheck -p
```

Should show:
- ✓ GDS is supported
- ✓ cuFile configuration is valid

### 2. Check Storage is GDS-capable
```bash
# Check filesystem type
df -T dat/

# GDS works best with:
# - Local NVMe (ext4, xfs)
# - Lustre (with proper configuration)
# - Does NOT work with NFS, CIFS, or network filesystems
```

### 3. Verify NUMA Topology
```bash
nvidia-smi topo -m
```

Look for direct NVMe connection to GPU's NUMA node.

---

## Common Issues

### Issue: "Failed to create HDF5 file"
**Solution:** Ensure `dat/` directory exists:
```bash
mkdir -p dat log
```

### Issue: Very slow performance
**Possible causes:**
1. Not using native GDS mode (check `CUFILE_ENV_PATH_JSON`)
2. Storage is not GDS-capable (network filesystem?)
3. Not built with `USE_SYSTEM_MALLOC` on GH200
4. NUMA binding not set correctly

### Issue: "cuFile configuration error"
**Solution:** Check that cuFile JSON files exist:
```bash
ls -la *.json
# Should see: disable_compat.json, force_compat.json
```

### Issue: Results inconsistent between runs
**Before our fixes:** Cache effects, GPU async overlap
**After our fixes:** Should be consistent (±5%)

---

## Performance Tuning Tips

### 1. Optimize Block and Buffer Sizes
```bash
# Try different combinations
for FBLK in 4096 8192 16384; do
  for CBUF in $((FBLK * 1024)) $((FBLK * 2048)) $((FBLK * 4096)); do
    echo "Testing fblk=$FBLK cbuf=$CBUF"
    ./bin/h5gds --num 16777216 --fblk $FBLK --cbuf $CBUF
  done
done
```

### 2. Scale Dataset Size
Start small and scale up to find sweet spot:
```bash
for NUM in 1024 4096 16384 65536 262144 1048576 4194304 16777216; do
  echo "Testing num=$NUM"
  ./bin/h5gds --num $NUM
done
```

### 3. Monitor GPU Utilization
```bash
# In another terminal
watch -n 0.1 nvidia-smi
```

Look for:
- High memory bandwidth utilization during I/O
- No compute utilization (pure I/O test)

---

## Comparing Configurations

### Native GDS vs Compat Mode
```bash
# Run both modes with same parameters
for JSON in disable_compat.json force_compat.json; do
  export CUFILE_ENV_PATH_JSON=../$JSON
  echo "Testing with $JSON"
  ./bin/h5gds --num 16777216
done

# Compare results in log/h5gds_benchmark.csv
```

### Asis vs Hyperslab Mode
```bash
# Test both HDF5 layouts
./bin/h5gds --num 16777216 --asis
./bin/h5gds --num 16777216  # hyperslab (default)
```

Hyperslab mode is more complex but more flexible for partial I/O.

---

## System Requirements

- CUDA Toolkit with GDS support (11.4+)
- NVIDIA GPU with GDS support (Ampere or newer)
- Linux kernel 5.9+ (for io_uring compat mode)
- HDF5 1.14.0+
- VFD-GDS library
- Local NVMe or compatible storage

## For Best Results

1. ✅ Build with `-DUSE_SYSTEM_MALLOC=ON` on GH200
2. ✅ Use NUMA binding (`numactl --membind`)
3. ✅ Test with native GDS mode first
4. ✅ Use local NVMe storage, not network FS
5. ✅ Scale up dataset size to saturate bandwidth
6. ✅ Run multiple iterations to verify consistency

