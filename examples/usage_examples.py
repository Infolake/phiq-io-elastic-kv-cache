#!/usr/bin/env python3
"""
ΦQ™ PHIQ.IO Elastic KV Cache - Usage Examples
Author: Dr. Guilherme de Camargo
Organization: PHIQ.IO Quantum Technologies (ΦQ™)
Contact: https://phiq.io | support@phiq.io
© 2025 PHIQ.IO Quantum Technologies. All rights reserved.

Demonstrates CLI options for elastic key-value cache benchmarking
including paired baseline, compression ratios, and inference cycle timing.

Camargo Constant: Δ = φ + π = 4.759627
"""

import subprocess
import json
import os
import time
from pathlib import Path

class ElasticKVExample:
    def __init__(self, binary_path="./elastic_kv_cli"):
        self.binary_path = binary_path
        self.results = []

    def run_example(self, name, args, description):
        """Run a single example and collect results"""
        print(f"\n{'='*60}")
        print(f"EXAMPLE: {name}")
        print(f"{'='*60}")
        print(f"Description: {description}")
        print(f"Command: {self.binary_path} {' '.join(args)}")
        print("-" * 60)

        try:
            # Run with JSON output for parsing
            json_args = args + ['--json'] if '--json' not in args else args
            start_time = time.time()

            result = subprocess.run([self.binary_path] + json_args,
                                  capture_output=True, text=True, timeout=120)

            elapsed = time.time() - start_time

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    self.results.append({
                        'name': name,
                        'description': description,
                        'args': args,
                        'elapsed_time': elapsed,
                        'data': data
                    })

                    # Print summary
                    if 'results' in data:
                        r = data['results']
                        print(f"[SUCCESS] ({elapsed:.1f}s)")
                        print(f"   Tokens/sec: {r.get('tokens_per_sec', 0):.0f}")
                        print(f"   Speedup: {r.get('speedup_vs_baseline', 0):.3f}x")
                        print(f"   CV: {r.get('coefficient_of_variation', 0)*100:.1f}%")
                        print(f"   Memory Efficiency: {r.get('memory_efficiency_percent', 0):.1f}%")

                        if 'inference_cycle' in data and data['inference_cycle']:
                            ic = data['inference_cycle']
                            print(f"   Inference Speedup: {ic.get('speedup_vs_baseline', 0):.3f}x")

                except json.JSONDecodeError:
                    print(f"[FAILED] Invalid JSON output")
            else:
                print(f"[FAILED] Return code {result.returncode}")
                print(f"Error: {result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"[FAILED] Timeout after 120 seconds")
        except FileNotFoundError:
            print(f"[FAILED] Binary not found at {self.binary_path}")
            print("Please build the project first!")

def main():
    print("PHIQ ELASTIC KV CACHE - USAGE EXAMPLES")
    print("=" * 80)
    print("This script demonstrates various usage patterns and configurations.")
    print("Each example shows different aspects of the elastic KV cache performance.")

    example = ElasticKVExample()

    # Example 1: Basic usage
    example.run_example(
        "Basic Benchmark",
        ["--seq=1024", "--compress=2", "--reps=50"],
        "Simple benchmark with 2x compression on 1024 sequence length"
    )

    # Example 2: Paired baseline comparison
    example.run_example(
        "Paired Baseline Comparison",
        ["--seq=1024", "--compress=2", "--paired-baseline", "--reps=50"],
        "Compare elastic performance against uncompressed baseline"
    )

    # Example 3: High precision measurement
    example.run_example(
        "High Precision Measurement",
        ["--seq=1024", "--compress=2", "--reps=200", "--warmup=50",
         "--inner_loops=128", "--truncate=5"],
        "High-precision benchmark with outlier removal and temporal amplification"
    )

    # Example 4: Inference cycle simulation
    example.run_example(
        "Inference Cycle Simulation",
        ["--seq=1024", "--compress=2", "--inference", "--decode_tokens=64",
         "--paired-baseline"],
        "Real-world inference simulation with sequential decode steps"
    )

    # Example 5: Large model configuration
    example.run_example(
        "Large Model Configuration",
        ["--seq=4096", "--heads=32", "--dim=128", "--compress=4", "--reps=30"],
        "Large model simulation with high compression ratio"
    )

    # Example 6: Memory bandwidth focus
    example.run_example(
        "Memory Bandwidth Analysis",
        ["--seq=2048", "--compress=1", "--reps=100", "--warmup=20"],
        "Focus on memory bandwidth measurement (no compression)"
    )

    # Example 7: Extreme compression
    example.run_example(
        "Extreme Compression Test",
        ["--seq=8192", "--heads=64", "--dim=64", "--compress=8", "--reps=20"],
        "Test extreme compression ratios on large sequences"
    )

    # Example 8: CUDA Graphs disabled
    example.run_example(
        "Without CUDA Graphs",
        ["--seq=1024", "--compress=2", "--no-graphs", "--reps=50"],
        "Benchmark without CUDA Graphs optimization for comparison"
    )

    # Generate summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")

    if example.results:
        print(f"Total Examples: {len(example.results)}")
        print(f"Average Runtime: {sum(r['elapsed_time'] for r in example.results)/len(example.results):.1f}s")

        # Find best performing configurations
        speedups = []
        cvs = []
        memory_effs = []

        print(f"\n{'Configuration':<30} {'Speedup':<10} {'CV%':<8} {'MemEff%':<10} {'Status'}")
        print("-" * 70)

        for result in example.results:
            if 'data' in result and 'results' in result['data']:
                r = result['data']['results']
                speedup = r.get('speedup_vs_baseline', 0)
                cv = r.get('coefficient_of_variation', 0) * 100
                mem_eff = r.get('memory_efficiency_percent', 0)

                speedups.append(speedup)
                cvs.append(cv)
                memory_effs.append(mem_eff)

                # Status determination
                if speedup >= 2.0 and cv <= 1.0:
                    status = "[GOLDEN]"
                elif speedup >= 1.8:
                    status = "[EXCELLENT]"
                elif speedup >= 1.5:
                    status = "[GOOD]"
                else:
                    status = "[FAIR]"

                name = result['name'][:29]
                print(f"{name:<30} {speedup:<10.3f} {cv:<8.1f} {mem_eff:<10.1f} {status}")

        if speedups:
            print(f"\nBest Performance:")
            print(f"  Max Speedup: {max(speedups):.3f}x")
            print(f"  Best CV: {min(cvs):.1f}%")
            print(f"  Best Memory Efficiency: {max(memory_effs):.1f}%")

            # Golden ticket analysis
            golden_count = sum(1 for i, r in enumerate(example.results)
                             if 'data' in r and 'results' in r['data'] and
                             r['data']['results'].get('speedup_vs_baseline', 0) >= 2.0 and
                             r['data']['results'].get('coefficient_of_variation', 1) <= 0.01)

            print(f"\nGolden Ticket Status: {golden_count}/{len(example.results)} configurations achieved full criteria")

    else:
        print("No successful results to analyze.")

    # Usage recommendations
    print(f"\n{'='*80}")
    print("USAGE RECOMMENDATIONS")
    print(f"{'='*80}")
    print("""
1. For Production Deployment:
   Use paired baseline comparison with sufficient iterations:
   ./elastic_kv_cli --seq=1024 --compress=2 --paired-baseline --reps=100 --json

2. For Development/Testing:
   Use quick tests with lower iteration counts:
   ./elastic_kv_cli --seq=1024 --compress=2 --reps=20 --warmup=5

3. For High-Precision Benchmarking:
   Use inner loops and trimmed mean:
   ./elastic_kv_cli --seq=1024 --compress=2 --reps=200 --inner_loops=64 --truncate=5

4. For Real-World Performance:
   Always include inference cycle simulation:
   ./elastic_kv_cli --seq=1024 --compress=2 --inference --decode_tokens=64 --paired-baseline

5. For Large Model Simulation:
   Scale up sequence length and compression:
   ./elastic_kv_cli --seq=4096 --heads=32 --dim=128 --compress=4

Remember: Higher compression ratios work better with larger sequence lengths!
""")

if __name__ == '__main__':
    main()