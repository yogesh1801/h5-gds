#!/bin/bash
# GDS Diagnostic Script
# Run this on your cluster to diagnose why GDS is not performing

echo "=================================================="
echo "GDS DIAGNOSTIC REPORT"
echo "=================================================="
echo ""

# 1. Check nvidia_fs driver
echo "=== 1. NVIDIA_FS DRIVER STATUS ==="
if lsmod | grep -q nvidia_fs; then
    echo "✓ nvidia_fs driver is LOADED"
    lsmod | grep nvidia_fs
else
    echo "✗ nvidia_fs driver is NOT LOADED!"
    echo "   → GDS will fall back to compatibility mode"
    echo "   → This explains why GDS == SEC2 performance"
fi
echo ""

# 2. Check GDS stats
echo "=== 2. GDS RUNTIME STATS ==="
if [ -f "/proc/driver/nvidia-fs/stats" ]; then
    echo "✓ GDS stats file exists"
    cat /proc/driver/nvidia-fs/stats
else
    echo "✗ GDS stats file NOT found"
    echo "   → nvidia-fs driver not properly installed"
fi
echo ""

# 3. Check GPU GDS support
echo "=== 3. GPU GPUDIRECT STORAGE SUPPORT ==="
nvidia-smi -q | grep -i "GPUDirect" || echo "✗ No GPUDirect info found"
echo ""

# 4. Check cuFile environment
echo "=== 4. CUFILE CONFIGURATION ==="
echo "CUFILE_ENV_PATH_JSON: ${CUFILE_ENV_PATH_JSON:-NOT SET}"
if [ -n "$CUFILE_ENV_PATH_JSON" ] && [ -f "$CUFILE_ENV_PATH_JSON" ]; then
    echo "✓ cuFile config file exists:"
    cat "$CUFILE_ENV_PATH_JSON"
else
    echo "✗ cuFile config file not found or not set"
fi
echo ""

# 5. Check storage type
echo "=== 5. STORAGE FILESYSTEM TYPE ==="
df -T /tmp/yogesh/gpu0 2>/dev/null || df -T ./
echo ""

# 6. Run gdscheck if available
echo "=== 6. GDS HEALTH CHECK ==="
if [ -f "$CUDA_HOME/gds/tools/gdscheck" ]; then
    echo "Running gdscheck..."
    $CUDA_HOME/gds/tools/gdscheck -p 2>&1 | head -50
else
    echo "✗ gdscheck tool not found at $CUDA_HOME/gds/tools/gdscheck"
fi
echo ""

# 7. Check for cuFile library
echo "=== 7. CUFILE LIBRARY ==="
if ldconfig -p | grep -q libcufile; then
    echo "✓ libcufile found:"
    ldconfig -p | grep libcufile
else
    echo "✗ libcufile NOT found in library path"
fi
echo ""

# 8. Summary and diagnosis
echo "=================================================="
echo "DIAGNOSIS SUMMARY"
echo "=================================================="

ISSUES_FOUND=0

if ! lsmod | grep -q nvidia_fs; then
    echo "❌ ISSUE #1: nvidia_fs driver not loaded"
    echo "   Fix: modprobe nvidia_fs  (requires root)"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if [ ! -f "/proc/driver/nvidia-fs/stats" ]; then
    echo "❌ ISSUE #2: GDS driver not
 properly installed"
    echo "   Fix: Install NVIDIA GPUDirect Storage"
    ISSUES_FOUND=$((ISSUES_FOUND + 1))
fi

if [ -z "$CUFILE_ENV_PATH_JSON" ]; then
    echo "⚠️  WARNING: CUFILE_ENV_PATH_JSON not set"
    echo "   Using default cuFile configuration"
fi

if [ $ISSUES_FOUND -eq 0 ]; then
    echo "✓ No obvious GDS configuration issues found"
    echo "  → If GDS still matches SEC2 performance, check:"
    echo "    - Storage is NVMe or GDS-compatible"
    echo "    - GPU supports GDS (run gdscheck)"
    echo "    - Benchmark is using --force flag"
else
    echo ""
    echo "☠️  FOUND $ISSUES_FOUND CRITICAL ISSUE(S)"
    echo "   GDS will NOT work until these are resolved!"
fi

echo "=================================================="
