# cuFile JSON Configuration Reference

## Valid Configuration Options

### Logging Levels

**Valid values for `logging.level`:**
```
"OFF"    - No logging (fastest, no overhead)
"ERROR"  - Only critical errors
"WARN"   - Errors + warnings
"INFO"   - General information (default, recommended for production)
"DEBUG"  - Maximum verbosity (use for troubleshooting)
```

**❌ Invalid:** `"ALL"` (not a standard cuFile level)  
**✅ For troubleshooting:** Use `"DEBUG"`

---

## Complete JSON Configuration Example

### Native GDS Mode (disable_compat.json)

```json
{
    "logging": {
        "dir": "/tmp",
        "level": "DEBUG"
    },
    "properties": {
        "use_compat_mode": false,
        "allow_compat_mode": false,
        "gds_rdma_write_support": true,
        "max_direct_io_size_kb": 16384,
        "max_device_cache_size_kb": 131072,
        "max_device_pinned_mem_size_kb": 33554432,
        "posix_pool_slab_size_kb": [4, 1024, 16384],
        "posix_pool_slab_count": [128, 64, 32]
    },
    "fs": {
        "generic": {
            "posix_unaligned_writes": false
        }
    },
    "profile": {
        "nvtx": true,
        "cufile_stats": 1
    },
    "execution": {
        "max_io_threads": 4,
        "max_io_queue_depth": 128,
        "parallel_io": true,
        "min_io_threshold_size_kb": 8192
    }
}
```

### Compatibility Mode (force_compat.json)

```json
{
    "logging": {
        "dir": "/tmp",
        "level": "DEBUG"
    },
    "properties": {
        "use_compat_mode": true,
        "force_compat_mode": true,
        "allow_compat_mode": true,
        "gds_rdma_write_support": true,
        "max_direct_io_size_kb": 16384,
        "max_device_cache_size_kb": 131072,
        "max_device_pinned_mem_size_kb": 33554432,
        "posix_pool_slab_size_kb": [4, 1024, 16384],
        "posix_pool_slab_count": [128, 64, 32]
    },
    "fs": {
        "generic": {
            "posix_unaligned_writes": false
        }
    },
    "profile": {
        "nvtx": true,
        "cufile_stats": 1
    },
    "execution": {
        "max_io_threads": 4,
        "max_io_queue_depth": 128,
        "parallel_io": true,
        "min_io_threshold_size_kb": 8192
    }
}
```

---

## Key Properties Explained

### Mode Control

| Property | Type | Description | Values |
|----------|------|-------------|--------|
| `use_compat_mode` | bool | Enable/disable compat mode | `true`/`false` |
| `allow_compat_mode` | bool | Allow fallback to compat | `true`/`false` |
| `force_compat_mode` | bool | Force compat mode | `true`/`false` |

**For native GDS:** All three should be `false`  
**For compat mode:** All three should be `true` (or at least `force_compat_mode: true`)

### Buffer Sizes

| Property | Default | Description |
|----------|---------|-------------|
| `max_direct_io_size_kb` | 16384 | Max direct I/O size (16 MB) |
| `max_device_cache_size_kb` | 131072 | Device cache size (128 MB) |
| `max_device_pinned_mem_size_kb` | 33554432 | Max pinned memory (32 GB) |

### Pool Configuration

```json
"posix_pool_slab_size_kb": [4, 1024, 16384]
```
Slab sizes in KB for memory pool: 4KB, 1MB, 16MB

```json
"posix_pool_slab_count": [128, 64, 32]
```
Number of slabs for each size

### Execution Settings

| Property | Default | Description |
|----------|---------|-------------|
| `max_io_threads` | 4 | Number of I/O threads |
| `max_io_queue_depth` | 128 | Queue depth per thread |
| `parallel_io` | true | Enable parallel I/O |
| `min_io_threshold_size_kb` | 8192 | Min size for parallel I/O (8 MB) |

### Profiling

| Property | Type | Description |
|----------|------|-------------|
| `nvtx` | bool | Enable NVIDIA Tools Extension markers |
| `cufile_stats` | int | Enable statistics (0=off, 1=on) |

---

## Log Output Location

With `"dir": "/tmp"`, logs will be created at:
```
/tmp/cufile.log
/tmp/cufile_<PID>.log
/tmp/cufile_stats.log  (if cufile_stats: 1)
```

---

## Environment Variables

```bash
# Specify custom JSON config
export CUFILE_ENV_PATH_JSON=/path/to/config.json

# Alternative (some versions)
export CUFILE_JSON=/path/to/config.json
```

---

## Troubleshooting Log Levels

### For Production:
```json
"level": "INFO"
```
Balanced logging with reasonable overhead.

### For Troubleshooting:
```json
"level": "DEBUG"
```
Maximum verbosity - use when diagnosing issues.

### For Benchmarking:
```json
"level": "WARN"
```
Minimal overhead, only warnings and errors.

---

## Verification After Changes

Check if configuration is loaded:
```bash
# Run gdscheck to see active config
$CUDA_HOME/gds/tools/gdscheck -p | grep -A 20 "CUFILE CONFIGURATION"

# Check logs
ls -lh /tmp/cufile*.log

# Search for your settings in log
grep -i "use_compat_mode" /tmp/cufile*.log
```

---

## Common Issues

### Issue: Config not loaded
**Symptoms:** gdscheck shows different values than JSON  
**Solution:** 
- Check `CUFILE_ENV_PATH_JSON` is set correctly
- Verify JSON is valid: `python3 -m json.tool config.json`
- Ensure file permissions allow reading

### Issue: Logs not created
**Symptoms:** No files in `/tmp/cufile*.log`  
**Solution:**
- Check write permissions to log directory
- Try different log dir: `"dir": "/home/username/logs"`
- Verify cuFile is actually being used

### Issue: Invalid log level error
**Symptoms:** cuFile fails to initialize  
**Solution:**
- Use valid level: `OFF`, `ERROR`, `WARN`, `INFO`, or `DEBUG`
- Check for typos (case-sensitive)

---

## Reference

For more details, see:
- NVIDIA GPUDirect Storage Documentation
- cuFile API Guide: `/usr/local/cuda/gds/docs/`
- Sample configs: `/usr/local/cuda/gds/tools/`


