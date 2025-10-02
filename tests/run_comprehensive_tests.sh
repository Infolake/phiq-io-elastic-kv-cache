#!/bin/bash
# =============================================================================
# PHIQ Elastic KV Cache - Comprehensive Test Suite
# Validates performance across different configurations
# =============================================================================

echo "PHIQ Elastic KV Cache - Comprehensive Test Suite"
echo "================================================"

# Configuration
OUTPUT_DIR="test_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${OUTPUT_DIR}/comprehensive_results_${TIMESTAMP}.json"

# Create output directory
mkdir -p $OUTPUT_DIR

# Check if binary exists
if [ ! -f "./elastic_kv_cli" ]; then
    echo "Error: elastic_kv_cli not found. Please build the project first."
    exit 1
fi

echo "Starting comprehensive benchmark suite..."
echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Test 1: Baseline Performance
echo "Test 1: Baseline Performance Validation"
echo "----------------------------------------"
./elastic_kv_cli \
    --seq=1024 \
    --heads=16 \
    --dim=64 \
    --compress=1 \
    --reps=100 \
    --warmup=20 \
    --json > "${OUTPUT_DIR}/test1_baseline.json"

if [ $? -eq 0 ]; then
    echo "[OK] Baseline test completed"
else
    echo "[ERROR] Baseline test failed"
    exit 1
fi

# Test 2: Elastic 2x Compression
echo ""
echo "Test 2: Elastic 2x Compression"
echo "-------------------------------"
./elastic_kv_cli \
    --seq=1024 \
    --heads=16 \
    --dim=64 \
    --compress=2 \
    --reps=100 \
    --warmup=20 \
    --paired-baseline \
    --json > "${OUTPUT_DIR}/test2_elastic_2x.json"

if [ $? -eq 0 ]; then
    echo "[OK] Elastic 2x test completed"
else
    echo "[ERROR] Elastic 2x test failed"
fi

# Test 3: High Compression (4x)
echo ""
echo "Test 3: High Compression (4x)"
echo "------------------------------"
./elastic_kv_cli \
    --seq=2048 \
    --heads=32 \
    --dim=64 \
    --compress=4 \
    --reps=50 \
    --warmup=20 \
    --paired-baseline \
    --json > "${OUTPUT_DIR}/test3_elastic_4x.json"

if [ $? -eq 0 ]; then
    echo "[OK] High compression test completed"
else
    echo "[ERROR] High compression test failed"
fi

# Test 4: Inference Cycle Simulation
echo ""
echo "Test 4: Inference Cycle Simulation"
echo "-----------------------------------"
./elastic_kv_cli \
    --seq=1024 \
    --heads=16 \
    --dim=64 \
    --compress=2 \
    --reps=50 \
    --warmup=20 \
    --inference \
    --decode_tokens=64 \
    --paired-baseline \
    --json > "${OUTPUT_DIR}/test4_inference_cycle.json"

if [ $? -eq 0 ]; then
    echo "[OK] Inference cycle test completed"
else
    echo "[ERROR] Inference cycle test failed"
fi

# Test 5: Large Model Simulation
echo ""
echo "Test 5: Large Model Simulation"
echo "-------------------------------"
./elastic_kv_cli \
    --seq=4096 \
    --heads=32 \
    --dim=128 \
    --compress=4 \
    --reps=30 \
    --warmup=10 \
    --paired-baseline \
    --json > "${OUTPUT_DIR}/test5_large_model.json"

if [ $? -eq 0 ]; then
    echo "[OK] Large model test completed"
else
    echo "[ERROR] Large model test failed"
fi

# Test 6: Precision Test (High iterations)
echo ""
echo "Test 6: Precision Test (High iterations)"
echo "-----------------------------------------"
./elastic_kv_cli \
    --seq=1024 \
    --heads=16 \
    --dim=64 \
    --compress=2 \
    --reps=500 \
    --warmup=100 \
    --inner_loops=128 \
    --truncate=5 \
    --paired-baseline \
    --json > "${OUTPUT_DIR}/test6_precision.json"

if [ $? -eq 0 ]; then
    echo "[OK] Precision test completed"
else
    echo "[ERROR] Precision test failed"
fi

# Combine results into comprehensive report
echo ""
echo "Generating comprehensive report..."

cat > "$RESULTS_FILE" << EOF
{
  "test_suite": "phiq_elastic_kv_comprehensive",
  "timestamp": "$TIMESTAMP",
  "tests_completed": [
    {
      "name": "baseline_performance",
      "file": "test1_baseline.json",
      "description": "Baseline performance validation"
    },
    {
      "name": "elastic_2x_compression",
      "file": "test2_elastic_2x.json",
      "description": "2x compression with paired baseline"
    },
    {
      "name": "high_compression_4x",
      "file": "test3_elastic_4x.json",
      "description": "4x compression large sequence"
    },
    {
      "name": "inference_cycle",
      "file": "test4_inference_cycle.json",
      "description": "Real-world inference simulation"
    },
    {
      "name": "large_model_simulation",
      "file": "test5_large_model.json",
      "description": "Large model configuration test"
    },
    {
      "name": "precision_test",
      "file": "test6_precision.json",
      "description": "High-precision measurement validation"
    }
  ],
  "summary": {
    "total_tests": 6,
    "output_directory": "$OUTPUT_DIR",
    "platform": "$(uname -s)",
    "gpu_info": "Run nvidia-smi for GPU details"
  }
}
EOF

echo ""
echo "=============================================="
echo "Comprehensive Test Suite Completed!"
echo "=============================================="
echo "Results directory: $OUTPUT_DIR"
echo "Comprehensive report: $RESULTS_FILE"
echo ""
echo "Individual test results:"
ls -la ${OUTPUT_DIR}/test*.json
echo ""

# Quick analysis
echo "Quick Analysis:"
echo "---------------"

if [ -f "${OUTPUT_DIR}/test2_elastic_2x.json" ]; then
    SPEEDUP=$(grep '"speedup_vs_baseline"' "${OUTPUT_DIR}/test2_elastic_2x.json" | head -1 | grep -o '[0-9]\+\.[0-9]\+')
    CV=$(grep '"coefficient_of_variation"' "${OUTPUT_DIR}/test2_elastic_2x.json" | head -1 | grep -o '[0-9]\+\.[0-9]\+')

    if [ ! -z "$SPEEDUP" ] && [ ! -z "$CV" ]; then
        echo "Elastic 2x Performance:"
        echo "  Speedup: ${SPEEDUP}x"
        echo "  CV: ${CV} ($(echo "$CV * 100" | bc -l | cut -d. -f1)%)"

        # Golden ticket analysis
        if (( $(echo "$SPEEDUP >= 2.0" | bc -l) )); then
            echo "  [GOLDEN TICKET] Speedup: ACHIEVED"
        else
            echo "  [NEAR TARGET] Speedup: CLOSE (target: 2.0x)"
        fi

        if (( $(echo "$CV <= 0.01" | bc -l) )); then
            echo "  [GOLDEN TICKET] Precision: ACHIEVED"
        else
            echo "  [GOOD] Precision: GOOD (target: ≤1%)"
        fi
    fi
fi

echo ""
echo "Use the following commands to analyze results:"
echo "  cat $RESULTS_FILE | jq ."
echo "  python analyze_results.py $OUTPUT_DIR"
echo ""