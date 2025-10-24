#!/bin/bash
#
# Comprehensive test of all cuFile configuration variants
# Tests: baseline, O_DIRECT, io_uring, and both combined
#

echo "=========================================="
echo "cuFile Configuration Performance Test"
echo "=========================================="
echo ""

# Configuration
TEST_SIZE=33554432  # 32M particles for quick test
RESULTS_DIR="config_test_results"
BUILD_DIR="/work/jh250079/n14001/h5-gds/build"

# Check prerequisites
if [ ! -f "$BUILD_DIR/bin/h5gds" ]; then
    echo "❌ Binary not found! Build first:"
    echo "   cd /work/jh250079/n14001/h5-gds"
    echo "   cmake -S . -B build -DUSE_SYSTEM_MALLOC=ON -DTARGET_GPU=NVIDIA_CC90"
    echo "   cd build && make -j8"
    exit 1
fi

# Check if system malloc is enabled
if ! nm "$BUILD_DIR/bin/h5gds" | grep -q "first_touch"; then
    echo "⚠️  WARNING: Binary not built with USE_SYSTEM_MALLOC=ON"
    echo "   Results will be suboptimal! Rebuild with correct flag."
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Setup NUMA binding
GPU_ID=0
BUS_ID=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $GPU_ID | tr '[:upper:]' '[:lower:]' | sed 's/^0\{0,\}\([0-9]\{4\}\)/\1/')
NUMA_NODE=$(cat /sys/bus/pci/devices/$BUS_ID/numa_node 2>/dev/null || echo "0")

echo "Test Configuration:"
echo "  GPU: $GPU_ID (NUMA node: $NUMA_NODE)"
echo "  Test size: $TEST_SIZE particles"
echo "  Results: $RESULTS_DIR/"
echo ""

# Create results directory
mkdir -p "$RESULTS_DIR"
cd "$RESULTS_DIR" || exit 1

# Test configurations
declare -A CONFIGS=(
    ["baseline"]="disable_compat.json"
    ["odirect"]="disable_compat_odirect.json"
    ["iouring"]="disable_compat_iouring.json"
    ["both"]="disable_compat_both.json"
)

declare -A DESCRIPTIONS=(
    ["baseline"]="Baseline (no O_DIRECT, no io_uring)"
    ["odirect"]="O_DIRECT enabled (bypass page cache)"
    ["iouring"]="io_uring enabled (modern async I/O)"
    ["both"]="Both O_DIRECT + io_uring"
)

# Run tests
for config_name in baseline odirect iouring both; do
    config_file="${CONFIGS[$config_name]}"
    description="${DESCRIPTIONS[$config_name]}"
    
    echo "=========================================="
    echo "Testing: $description"
    echo "Config: $config_file"
    echo "=========================================="
    echo ""
    
    # Setup test directory
    TEST_DIR="test_${config_name}"
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR" || exit 1
    
    # Copy binary and setup
    cp -r "$BUILD_DIR/bin" .
    mkdir -p dat log
    
    # Set cuFile config
    export CUFILE_ENV_PATH_JSON="$BUILD_DIR/$config_file"
    
    if [ ! -f "$CUFILE_ENV_PATH_JSON" ]; then
        echo "⚠️  Config file not found: $CUFILE_ENV_PATH_JSON"
        echo "   Skipping this test..."
        cd ..
        continue
    fi
    
    echo "Running with CUFILE_ENV_PATH_JSON=$CUFILE_ENV_PATH_JSON"
    echo ""
    
    # Run test with NUMA binding
    numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE \
        bin/h5gds --asis --num $TEST_SIZE --fblk 16777216
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✓ Test completed successfully"
        
        # Show results
        if [ -f log/h5gds_benchmark.csv ]; then
            echo ""
            echo "Results:"
            tail -1 log/h5gds_benchmark.csv | awk -F',' '{
                printf "  Write BW: %.2f GB/s\n", $8/1e9
                printf "  Read BW:  %.2f GB/s\n", $9/1e9
                printf "  Write time: %.3f s\n", $6
                printf "  Read time:  %.3f s\n", $7
            }'
        fi
    else
        echo "✗ Test failed with exit code: $EXIT_CODE"
    fi
    
    echo ""
    cd ..
done

# Compare results
echo "=========================================="
echo "PERFORMANCE COMPARISON"
echo "=========================================="
echo ""

printf "%-12s | %12s | %12s | %12s | %12s\n" \
    "Config" "Write BW" "Read BW" "Write Time" "Read Time"
echo "-------------+-------------+-------------+-------------+-------------"

for config_name in baseline odirect iouring both; do
    LOG_FILE="test_${config_name}/log/h5gds_benchmark.csv"
    
    if [ -f "$LOG_FILE" ]; then
        tail -1 "$LOG_FILE" | awk -v name="$config_name" -F',' '{
            printf "%-12s | %9.2f GB/s | %9.2f GB/s | %9.3f s | %9.3f s\n", 
                name, $8/1e9, $9/1e9, $6, $7
        }'
    else
        printf "%-12s | %12s | %12s | %12s | %12s\n" \
            "$config_name" "N/A" "N/A" "N/A" "N/A"
    fi
done

echo ""
echo "=========================================="
echo "ANALYSIS"
echo "=========================================="
echo ""

# Calculate speedups
BASELINE_WRITE=$(awk -F',' 'END {print $8}' test_baseline/log/h5gds_benchmark.csv 2>/dev/null)

if [ ! -z "$BASELINE_WRITE" ] && [ "$BASELINE_WRITE" != "0" ]; then
    echo "Speedup vs Baseline (Write Bandwidth):"
    echo ""
    
    for config_name in odirect iouring both; do
        LOG_FILE="test_${config_name}/log/h5gds_benchmark.csv"
        
        if [ -f "$LOG_FILE" ]; then
            awk -v baseline="$BASELINE_WRITE" -v name="$config_name" -F',' 'END {
                speedup = $8 / baseline
                percent = (speedup - 1) * 100
                printf "  %-12s: %.2fx (%.1f%% %s)\n", 
                    name, speedup, 
                    (percent >= 0 ? percent : -percent),
                    (percent >= 0 ? "faster" : "slower")
            }' "$LOG_FILE"
        fi
    done
fi

echo ""
echo "=========================================="
echo "RECOMMENDATIONS"
echo "=========================================="
echo ""

# Determine best configuration
BEST_CONFIG="baseline"
BEST_BW=0

for config_name in baseline odirect iouring both; do
    LOG_FILE="test_${config_name}/log/h5gds_benchmark.csv"
    
    if [ -f "$LOG_FILE" ]; then
        BW=$(awk -F',' 'END {print $8}' "$LOG_FILE" 2>/dev/null)
        if [ ! -z "$BW" ]; then
            if (( $(echo "$BW > $BEST_BW" | bc -l) )); then
                BEST_BW=$BW
                BEST_CONFIG=$config_name
            fi
        fi
    fi
done

echo "✅ Best configuration: ${DESCRIPTIONS[$BEST_CONFIG]}"
echo "   Config file: ${CONFIGS[$BEST_CONFIG]}"
echo "   Bandwidth: $(echo "scale=2; $BEST_BW / 1e9" | bc) GB/s"
echo ""

case $BEST_CONFIG in
    baseline)
        echo "Note: Baseline is best. O_DIRECT and io_uring may not help"
        echo "      or your kernel/filesystem doesn't support them optimally."
        ;;
    odirect)
        echo "Recommendation: Use force_odirect_mode: true"
        echo "Benefit: Bypasses page cache for better throughput"
        ;;
    iouring)
        echo "Recommendation: Use prefer_iouring: true"
        echo "Benefit: Modern async I/O with lower overhead"
        ;;
    both)
        echo "Recommendation: Use both force_odirect_mode AND prefer_iouring"
        echo "Benefit: Maximum performance for large sequential I/O"
        ;;
esac

echo ""
echo "Use this configuration for production runs:"
echo "  export CUFILE_ENV_PATH_JSON=$BUILD_DIR/${CONFIGS[$BEST_CONFIG]}"
echo ""

echo "Full results saved in: $RESULTS_DIR/"
echo "=========================================="


