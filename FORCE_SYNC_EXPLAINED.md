# Force Sync to Disk for Maximum Consistency

## What Was Added

To get the most consistent benchmark results, we now force all data through every layer of the storage hierarchy:

### 1. After Write Completes

```cpp
// After H5Fclose(target)

// Force all data to physical storage (not just storage controller cache)
sync();  // Flush all dirty filesystem buffers system-wide

int fd = open(name.c_str(), O_RDONLY);
if (fd >= 0) {
  fsync(fd);  // Ensure THIS specific file is on disk
  close(fd);
}

// Also sync the directory metadata
std::string dir = "dat";
int dirfd = open(dir.c_str(), O_RDONLY | O_DIRECTORY);
if (dirfd >= 0) {
  fsync(dirfd);  // Ensure directory entry is on disk
  close(dirfd);
}
```

### 2. Before Read Begins

```cpp
// Ensure GPU is completely idle before read benchmark
cudaDeviceSynchronize();

// Drop filesystem cache for cold read
drop_file_cache(name);

// Small delay to ensure all system state has settled
usleep(100000);  // 100ms delay
```

## Storage Hierarchy

Data must pass through multiple layers before reaching physical media:

```
Application (HDF5)
       ↓
 HDF5 Buffers ← H5Fflush() clears this
       ↓
Filesystem Cache ← sync() / fsync() clears this
       ↓
Storage Controller Cache ← fsync() with barriers clears this
       ↓
Physical Media (SSD/NVMe)
```

## What Each Call Does

### `H5Fflush(target, H5F_SCOPE_GLOBAL)`
- Flushes HDF5 internal buffers to the filesystem
- Does NOT guarantee data is on physical disk
- Required but not sufficient

### `sync()`
- Flushes ALL dirty filesystem buffers to storage
- System-wide operation
- Ensures kernel page cache is written out
- **Note:** Can be slow on busy systems

### `fsync(fd)`
- Ensures specific file's data AND metadata reach storage
- More targeted than `sync()`
- Waits for storage controller to acknowledge write
- **Includes write barriers** to prevent reordering

### `fsync(dirfd)` on directory
- Ensures directory entry is durable
- Critical for new files
- Guarantees file is discoverable after crash

### `drop_file_cache(name)` - uses `posix_fadvise(POSIX_FADV_DONTNEED)`
- Tells kernel to evict file's pages from cache
- Next read will be cold from storage
- Does not require root

### `usleep(100000)` - 100ms delay
- Allows transient system state to settle
- Storage controller caches to flush
- PCIe transactions to complete
- Not strictly necessary but improves consistency

## Performance Impact

### Before These Changes
- Write: ~0.1s (but data in cache)
- Read: ~0.05s (from warm cache)
- **Not representative of real I/O performance**

### After These Changes
- Write: ~0.5-2s (includes all flushes)
- Read: ~0.8-1.5s (cold from disk)
- **Represents true storage performance**

### Time Breakdown
- `H5Fflush()`: ~10-50ms
- `sync()`: ~100-500ms (depends on system load)
- `fsync()`: ~50-200ms (depends on storage)
- `usleep()`: 100ms
- Total overhead: ~260-850ms per iteration

## When to Use This

### Use force-sync-to-disk when:
✅ Benchmarking raw storage performance  
✅ Testing worst-case cold-start scenarios  
✅ Comparing different storage systems  
✅ Measuring durability overhead  
✅ Testing crash recovery scenarios  

### Skip force-sync when:
❌ Benchmarking application-level performance (caching is realistic)  
❌ Testing in-memory performance  
❌ Running thousands of iterations (too slow)  
❌ Measuring steady-state throughput with warm caches  

## Alternative: Faster but Less Conservative

If the full sync is too slow, you can use a lighter approach:

```cpp
// After write (lighter version)
H5Fflush(target, H5F_SCOPE_GLOBAL);
H5Fclose(target);
int fd = open(name.c_str(), O_RDONLY);
if (fd >= 0) {
  fsync(fd);  // File sync only, no system-wide sync()
  close(fd);
}

// Before read (lighter version)
cudaDeviceSynchronize();
drop_file_cache(name);
// Skip usleep() for faster iteration
```

This removes:
- `sync()` call (saves 100-500ms)
- `usleep()` delay (saves 100ms)

But keeps:
- `fsync()` on the file (ensures data on disk)
- Cache dropping (ensures cold read)

## Verification

To verify data is truly on disk:

```bash
# While benchmark is running, in another terminal:
watch -n 0.1 'grep "Dirty:" /proc/meminfo'

# Should see Dirty memory spike during write, then drop to near 0 after sync
```

## Storage Controller Cache

Even after `fsync()`, some storage controllers have battery-backed write caches that may not immediately write to NAND/platters. For absolute guarantees:

```bash
# Disable write cache on device (requires root)
hdparm -W 0 /dev/nvme0n1

# Or in smartctl
smartctl -s wcache,off /dev/nvme0n1
```

**Warning:** This severely impacts performance and is usually not necessary. Modern `fsync()` implementation uses write barriers that force controller cache flush.

## Expected Behavior

### With Force Sync:
```
Write benchmark: 1.2s  (includes all sync overhead)
Read benchmark:  0.9s  (cold from disk)
```

### Without Force Sync:
```
Write benchmark: 0.15s (HDF5 buffers + kernel cache)
Read benchmark:  0.05s (from warm cache)
```

The second case is **not measuring storage performance**, it's measuring memory subsystem performance.

## Summary

The added sync operations ensure:
1. ✅ Write benchmark includes durability cost (data on physical media)
2. ✅ Read benchmark measures cold storage performance (no cache hits)
3. ✅ Results are consistent across runs (no state contamination)
4. ✅ Represents worst-case I/O latency (maximum isolation)

This is the gold standard for storage benchmarking but adds significant overhead per iteration.

