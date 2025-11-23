#!/bin/bash
# Rebuild script for Grace Hopper GDS Compatibility

echo "=== Rebuilding for Grace Hopper (GH200) with Managed Memory ==="

# 1. Clean build directory
echo "Cleaning build directory..."
rm -rf build
mkdir -p build
cd build

# 2. Configure with USE_MANAGED_MEMORY=ON and USE_SYSTEM_MALLOC=OFF
echo "Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_SYSTEM_MALLOC=OFF \
    -DUSE_MANAGED_MEMORY=ON

# 3. Build
echo "Building..."
make -j

echo "=== Build Complete ==="
echo "You can now run the benchmark:"
echo "./bin/h5gds --vfd=gds --force ..."
