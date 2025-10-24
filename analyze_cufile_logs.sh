#!/bin/bash
#
# Analyze cuFile logs to diagnose GDS performance issues
#

LOG_DIR="/tmp"

echo "=========================================="
echo "cuFile Log Analysis"
echo "=========================================="
echo ""

# Find cuFile logs
echo "Looking for cuFile logs in $LOG_DIR..."
LOGS=$(ls $LOG_DIR/cufile*.log 2>/dev/null)

if [ -z "$LOGS" ]; then
    echo "❌ No cuFile logs found in $LOG_DIR"
    echo ""
    echo "Logs may not have been generated yet. Make sure:"
    echo "1. You've run the benchmark with CUFILE_ENV_PATH_JSON set"
    echo "2. The JSON file has logging enabled"
    echo "3. You have write permissions to /tmp/"
    exit 1
fi

echo "✓ Found cuFile logs:"
for log in $LOGS; do
    SIZE=$(du -h "$log" | awk '{print $1}')
    echo "  - $log ($SIZE)"
done
echo ""

# Analyze most recent log
LATEST_LOG=$(ls -t $LOG_DIR/cufile*.log 2>/dev/null | head -1)
echo "Analyzing: $LATEST_LOG"
echo "=========================================="
echo ""

# Check 1: Mode detection
echo "1. GDS Mode Configuration:"
echo "-------------------------------------------"
if grep -q "use_compat_mode.*false" "$LATEST_LOG" 2>/dev/null; then
    echo "✅ Native GDS mode configured"
elif grep -q "use_compat_mode.*true" "$LATEST_LOG" 2>/dev/null; then
    echo "❌ Compat mode configured (should be native!)"
else
    echo "⚠ Mode not clearly indicated in log"
fi
echo ""

# Check 2: Memory registration
echo "2. Memory Registration:"
echo "-------------------------------------------"
REGISTER_COUNT=$(grep -c "cuFileBuffer\|Register\|Pinned" "$LATEST_LOG" 2>/dev/null || echo "0")
echo "Memory registration operations: $REGISTER_COUNT"
if grep -q "registration failed\|register.*error" "$LATEST_LOG" 2>/dev/null; then
    echo "❌ Memory registration failures detected!"
    grep -i "registration.*fail\|register.*error" "$LATEST_LOG" | head -5
else
    echo "✓ No obvious registration errors"
fi
echo ""

# Check 3: Compatibility mode fallback
echo "3. Compat Mode Fallback Check:"
echo "-------------------------------------------"
if grep -q "falling back\|compat.*mode.*enabled\|using.*compat" "$LATEST_LOG" 2>/dev/null; then
    echo "⚠ WARNING: System fell back to compat mode!"
    echo ""
    echo "Reasons found:"
    grep -i "falling back\|compat.*mode.*enabled\|using.*compat\|reason" "$LATEST_LOG" | head -10
else
    echo "✓ No compat mode fallback detected"
fi
echo ""

# Check 4: GDS operations
echo "4. GDS Direct I/O Operations:"
echo "-------------------------------------------"
GDS_READ=$(grep -c "gds.*read\|direct.*read\|cuFileRead" "$LATEST_LOG" 2>/dev/null || echo "0")
GDS_WRITE=$(grep -c "gds.*write\|direct.*write\|cuFileWrite" "$LATEST_LOG" 2>/dev/null || echo "0")
echo "GDS read operations: $GDS_READ"
echo "GDS write operations: $GDS_WRITE"
echo ""

# Check 5: Errors and warnings
echo "5. Errors and Warnings:"
echo "-------------------------------------------"
ERROR_COUNT=$(grep -ic "error\|fail\|invalid" "$LATEST_LOG" 2>/dev/null || echo "0")
WARN_COUNT=$(grep -ic "warning\|warn" "$LATEST_LOG" 2>/dev/null || echo "0")
echo "Errors: $ERROR_COUNT"
echo "Warnings: $WARN_COUNT"

if [ "$ERROR_COUNT" -gt 0 ]; then
    echo ""
    echo "Recent errors:"
    grep -i "error\|fail" "$LATEST_LOG" | tail -10
fi

if [ "$WARN_COUNT" -gt 0 ]; then
    echo ""
    echo "Recent warnings:"
    grep -i "warning\|warn" "$LATEST_LOG" | tail -10
fi
echo ""

# Check 6: Buffer alignment
echo "6. Buffer Alignment Issues:"
echo "-------------------------------------------"
if grep -q "alignment\|aligned\|unaligned" "$LATEST_LOG" 2>/dev/null; then
    echo "Alignment-related messages:"
    grep -i "alignment\|aligned\|unaligned" "$LATEST_LOG" | head -5
else
    echo "✓ No alignment issues detected"
fi
echo ""

# Check 7: Performance hints
echo "7. Performance-Related Messages:"
echo "-------------------------------------------"
if grep -q "performance\|slow\|degraded\|bounce.*buffer" "$LATEST_LOG" 2>/dev/null; then
    grep -i "performance\|slow\|degraded\|bounce.*buffer" "$LATEST_LOG" | head -10
else
    echo "✓ No performance warnings"
fi
echo ""

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "Full logs available at:"
for log in $LOGS; do
    echo "  - $log"
done
echo ""
echo "To view full log:"
echo "  less $LATEST_LOG"
echo ""
echo "To search for specific errors:"
echo "  grep -i 'search_term' $LATEST_LOG"
echo ""
echo "To extract statistics:"
echo "  grep -i 'stat\|performance\|bandwidth' $LATEST_LOG"
echo "=========================================="

