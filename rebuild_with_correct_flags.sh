#!/bin/bash
#
# Rebuild h5gds with CORRECT USE_SYSTEM_MALLOC flag
#

cd /work/jh250079/n14001/h5-gds

echo "=========================================="
echo "Rebuilding with USE_SYSTEM_MALLOC=ON"
echo "=========================================="
echo ""

# Clean old build
echo "Step 1: Cleaning old build..."
rm -rf build
echo "✓ Done"
echo ""

# Configure with CORRECT flag (note the -D prefix!)
echo "Step 2: Running CMake with corrected flags..."
cmake -S /work/jh250079/n14001/h5-gds \
      -B /work/jh250079/n14001/h5-gds/build \
      -DVFD_GDS_INC=/work/jh250079/n14001/vfd-gds/src \
      -DVFD_GDS_LIB=/work/jh250079/n14001/vfd-gds/build/bin \
      -DCUDA_SAMPLES_DIR=/work/jh250079/n14001/cuda-samples/Common \
      -DHDF5_ROOT=/work/jh250079/n14001/hdf5_install \
      -DTARGET_GPU=NVIDIA_CC90 \
      -DUSE_SYSTEM_MALLOC=ON

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ CMake configuration successful"
    echo ""
else
    echo ""
    echo "✗ CMake configuration failed!"
    exit 1
fi

# Build
echo "Step 3: Building..."
cd build
make -j8

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Build successful"
    echo ""
else
    echo ""
    echo "✗ Build failed!"
    exit 1
fi

# Copy JSON files
echo "Step 4: Copying JSON files..."
cp ../disable_compat.json .
cp ../force_compat.json .
echo "✓ Done"
echo ""

# Verify system malloc is enabled
echo "=========================================="
echo "VERIFICATION"
echo "=========================================="
echo ""
echo "Checking for first_touch kernel symbol..."

if nm bin/h5gds | grep -q "first_touch"; then
    echo "✅ SUCCESS! first_touch kernel found"
    echo ""
    echo "Your binary is NOW built with USE_SYSTEM_MALLOC=ON"
    echo ""
    echo "Expected native GDS performance:"
    echo "  Write: 12-16 GB/s (was 2.6 GB/s)"
    echo "  Read:  14-18 GB/s (was 1.7 GB/s)"
    echo ""
    echo "Ready to run benchmarks!"
else
    echo "⚠ WARNING: first_touch kernel NOT found"
    echo ""
    echo "Something went wrong. Check CMakeCache.txt:"
    grep USE_SYSTEM_MALLOC CMakeCache.txt
fi

echo "=========================================="

