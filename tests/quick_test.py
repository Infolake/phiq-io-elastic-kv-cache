#!/usr/bin/env python3
"""
PHIQ Elastic KV Cache - Quick Performance Test
Simple validation script for CI/CD integration
"""

import subprocess
import json
import sys
import time

def run_benchmark(args, timeout=60):
    """Run benchmark with timeout"""
    try:
        cmd = ['./elastic_kv_cli'] + args
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

        if result.returncode == 0:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON output from benchmark")
                return None
        else:
            print(f"Benchmark failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        print(f"Benchmark timed out after {timeout} seconds")
        return None
    except FileNotFoundError:
        print("Error: elastic_kv_cli not found. Please build the project first.")
        return None

def validate_result(result, min_speedup=1.2, max_cv=0.1):
    """Validate benchmark result meets minimum criteria"""
    if not result or 'results' not in result:
        return False, "Invalid result format"

    r = result['results']

    # Check speedup
    speedup = r.get('speedup_vs_baseline', 0)
    if speedup < min_speedup:
        return False, f"Speedup {speedup:.3f}x below minimum {min_speedup}x"

    # Check coefficient of variation
    cv = r.get('coefficient_of_variation', 1.0)
    if cv > max_cv:
        return False, f"CV {cv:.3f} above maximum {max_cv}"

    # Check basic metrics exist
    required_metrics = ['tokens_per_sec', 'memory_bandwidth_gbs', 'roofline_score']
    for metric in required_metrics:
        if metric not in r or r[metric] <= 0:
            return False, f"Missing or invalid metric: {metric}"

    return True, "All checks passed"

def main():
    print("PHIQ Elastic KV Cache - Quick Performance Test")
    print("=" * 50)

    # Test configurations
    tests = [
        {
            'name': 'Basic Performance',
            'args': ['--seq=1024', '--compress=2', '--reps=20', '--warmup=5', '--json'],
            'min_speedup': 1.2,
            'max_cv': 0.2
        },
        {
            'name': 'Paired Baseline',
            'args': ['--seq=1024', '--compress=2', '--paired-baseline', '--reps=20', '--warmup=5', '--json'],
            'min_speedup': 1.5,
            'max_cv': 0.15
        },
        {
            'name': 'Inference Cycle',
            'args': ['--seq=1024', '--compress=2', '--inference', '--decode_tokens=32', '--reps=10', '--warmup=5', '--json'],
            'min_speedup': 1.3,
            'max_cv': 0.2
        }
    ]

    total_tests = len(tests)
    passed_tests = 0

    for i, test in enumerate(tests, 1):
        print(f"\nTest {i}/{total_tests}: {test['name']}")
        print("-" * 30)

        start_time = time.time()
        result = run_benchmark(test['args'])
        elapsed = time.time() - start_time

        if result:
            valid, message = validate_result(result, test['min_speedup'], test['max_cv'])

            if valid:
                print(f"[PASSED] ({elapsed:.1f}s)")

                # Print key metrics
                r = result['results']
                print(f"   Speedup: {r.get('speedup_vs_baseline', 0):.3f}x")
                print(f"   CV: {r.get('coefficient_of_variation', 0)*100:.1f}%")
                print(f"   Tokens/sec: {r.get('tokens_per_sec', 0):.0f}")
                print(f"   Memory Efficiency: {r.get('memory_efficiency_percent', 0):.1f}%")

                # Check for inference cycle
                if 'inference_cycle' in result and result['inference_cycle']:
                    ic = result['inference_cycle']
                    print(f"   Inference Speedup: {ic.get('speedup_vs_baseline', 0):.3f}x")

                passed_tests += 1
            else:
                print(f"[FAILED] ({elapsed:.1f}s)")
                print(f"   Reason: {message}")
        else:
            print(f"[FAILED] ({elapsed:.1f}s)")
            print("   Reason: Benchmark execution failed")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Tests Passed: {passed_tests}/{total_tests}")

    if passed_tests == total_tests:
        print("Status: [ALL TESTS PASSED]")
        print("The PHIQ Elastic KV Cache is working correctly!")
        sys.exit(0)
    else:
        print("Status: [SOME TESTS FAILED]")
        print("Please check the configuration and try again.")
        sys.exit(1)

if __name__ == '__main__':
    main()