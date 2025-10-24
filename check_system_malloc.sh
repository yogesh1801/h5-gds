#!/bin/bash
#
# Script to verify if h5gds binary was built with USE_SYSTEM_MALLOC=ON
#

BINARY_PATH="/work/jh250079/n14001/h5-gds/build/bin/h5gds"
BUILD_DIR="/work/jh250079/n14001/h5-gds/build"

echo "=========================================="
echo "Checking USE_SYSTEM_MALLOC Status"
echo "=========================================="
echo ""

# Check 1: Binary exists?
if [ ! -f "$BINARY_PATH" ]; then
    echo "❌ Binary not found at: $BINARY_PATH"
    echo "   You need to build first!"
    exit 1
fi
echo "✓ Binary found: $BINARY_PATH"
echo ""

# Check 2: first_touch kernel (most definitive test)
echo "Test 1: Checking for first_touch kernel symbol..."
if nm "$BINARY_PATH" 2>/dev/null | grep -q "first_touch"; then
    echo "✅ FOUND: first_touch kernel"
    echo "   → USE_SYSTEM_MALLOC=ON is ENABLED"
    SYSTEM_MALLOC_ENABLED=true
else
    echo "❌ NOT FOUND: first_touch kernel"
    echo "   → USE_SYSTEM_MALLOC=OFF (using cudaMalloc)"
    SYSTEM_MALLOC_ENABLED=false
fi
echo ""

# Check 3: CMake cache
echo "Test 2: Checking CMake configuration..."
if [ -f "$BUILD_DIR/CMakeCache.txt" ]; then
    if grep -q "USE_SYSTEM_MALLOC:BOOL=ON" "$BUILD_DIR/CMakeCache.txt"; then
        echo "✅ CMakeCache.txt shows: USE_SYSTEM_MALLOC=ON"
    elif grep -q "USE_SYSTEM_MALLOC:BOOL=OFF" "$BUILD_DIR/CMakeCache.txt"; then
        echo "❌ CMakeCache.txt shows: USE_SYSTEM_MALLOC=OFF"
    else
        echo "⚠ USE_SYSTEM_MALLOC not found in CMakeCache.txt (defaults to OFF)"
    fi
else
    echo "⚠ CMakeCache.txt not found"
fi
echo ""

# Check 4: Allocation functions
echo "Test 3: Checking memory allocation functions used..."
MALLOC_COUNT=$(nm "$BINARY_PATH" 2>/dev/null | grep -c " malloc$" || true)
CUDAMALLOC_COUNT=$(nm "$BINARY_PATH" 2>/dev/null | grep -c "cudaMalloc" || true)
echo "   malloc references: $MALLOC_COUNT"
echo "   cudaMalloc references: $CUDAMALLOC_COUNT"
echo ""

# Check 5: Compiler definitions
echo "Test 4: Checking compiler definitions in build files..."
if grep -r "HOST_MALLOC_AND_FIRST_TOUCH" "$BUILD_DIR" 2>/dev/null | head -n 3; then
    echo "✅ Found HOST_MALLOC_AND_FIRST_TOUCH in build files"
else
    echo "❌ HOST_MALLOC_AND_FIRST_TOUCH not found in build files"
fi
echo ""

# Final verdict
echo "=========================================="
echo "FINAL VERDICT:"
echo "=========================================="
if [ "$SYSTEM_MALLOC_ENABLED" = true ]; then
    echo "✅ Your binary IS built with USE_SYSTEM_MALLOC=ON"
    echo ""
    echo "Memory allocation: malloc() + first_touch kernel"
    echo "GH200 compatibility: GOOD for native GDS"
    echo "Expected native GDS performance: HIGH (12-16 GB/s)"
else
    echo "❌ Your binary is NOT built with USE_SYSTEM_MALLOC=ON"
    echo ""
    echo "Memory allocation: cudaMalloc() (GPU VRAM only)"
    echo "GH200 compatibility: POOR for native GDS"
    echo "Expected native GDS performance: LOW (2-3 GB/s)"
    echo ""
    echo "=========================================="
    echo "TO FIX: Rebuild with USE_SYSTEM_MALLOC=ON"
    echo "=========================================="
    echo ""
    echo "Run these commands:"
    echo "  cd /work/jh250079/n14001/h5-gds"
    echo "  rm -rf build"
    echo "  cmake -S . -B build -DUSE_SYSTEM_MALLOC=ON -DTARGET_GPU=NVIDIA_CC90"
    echo "  cd build && make -j8"
fi
echo "=========================================="

