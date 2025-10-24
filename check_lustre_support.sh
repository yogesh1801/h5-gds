#!/bin/bash
#
# Check if Lustre filesystem support can be enabled in cuFile
#

echo "=========================================="
echo "Checking Lustre Support for cuFile/GDS"
echo "=========================================="
echo ""

# Check cuFile version
echo "1. cuFile Version:"
if [ -f "$CUDA_HOME/gds/tools/gdscheck" ]; then
    $CUDA_HOME/gds/tools/gdscheck -p 2>/dev/null | grep -E "GDS release|libcufile version"
else
    echo "⚠ gdscheck not found"
fi
echo ""

# Check for Lustre filesystem
echo "2. Checking for Lustre filesystems:"
if command -v lfs &> /dev/null; then
    echo "✓ Lustre utilities (lfs) found"
    echo ""
    echo "Lustre mount points:"
    mount | grep lustre
    echo ""
    echo "Lustre version:"
    lfs --version
else
    echo "❌ Lustre utilities not found (lfs command missing)"
fi
echo ""

# Check your current paths
echo "3. Filesystem type for common paths:"
echo "-------------------------------------------"

PATHS=(
    "/tmp"
    "/work/jh250079/n14001/h5-gds"
    "/lustre"
    "/scratch"
)

for path in "${PATHS[@]}"; do
    if [ -d "$path" ]; then
        FS_TYPE=$(df -T "$path" 2>/dev/null | tail -1 | awk '{print $2}')
        DEVICE=$(df "$path" 2>/dev/null | tail -1 | awk '{print $1}')
        echo "$path"
        echo "  Filesystem type: $FS_TYPE"
        echo "  Device: $DEVICE"
        
        # Check if it's Lustre
        if mount | grep "$DEVICE" | grep -q lustre; then
            echo "  ⚠️ This is LUSTRE - native GDS may not work!"
        elif echo "$FS_TYPE" | grep -iq "lustre"; then
            echo "  ⚠️ This is LUSTRE - native GDS may not work!"
        elif echo "$DEVICE" | grep -q "nvme"; then
            echo "  ✅ This is local NVMe - native GDS should work!"
        fi
        echo ""
    fi
done

# Check cuFile configuration for Lustre
echo "4. cuFile Lustre Configuration:"
echo "-------------------------------------------"
if [ -f "$CUDA_HOME/gds/tools/gdscheck" ]; then
    echo "Lustre-related settings from gdscheck:"
    $CUDA_HOME/gds/tools/gdscheck -p 2>/dev/null | grep -i lustre
    echo ""
fi

# Check if Lustre GDS support exists
echo "5. Lustre GDS Kernel Module:"
echo "-------------------------------------------"
if lsmod | grep -q nvidia_fs; then
    echo "✓ nvidia_fs kernel module loaded (GDS driver)"
    lsmod | grep nvidia_fs
else
    echo "⚠ nvidia_fs kernel module not loaded"
fi
echo ""

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "For native GDS to work, you need:"
echo "  1. ✅ Supported filesystem (NVMe, DDN EXAScaler)"
echo "  2. ✅ nvidia_fs kernel module loaded"
echo "  3. ✅ Proper memory allocation (USE_SYSTEM_MALLOC=ON)"
echo "  4. ✅ cuFile configured correctly (disable_compat.json)"
echo ""
echo "⚠️ IMPORTANT: If using Lustre, native GDS may NOT work"
echo "   even with correct configuration. Stick to local NVMe"
echo "   (/tmp/) for benchmarks if native GDS is required."
echo ""
echo "Current recommendation:"
echo "  - Use /tmp/yogesh/ (local NVMe) ✅"
echo "  - Avoid Lustre paths for native GDS benchmarks ❌"
echo "=========================================="

