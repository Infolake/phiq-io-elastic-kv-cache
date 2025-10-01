#!/usr/bin/env python3
"""
PHIQ Elastic KV Cache - Results Analysis Tool
Analyzes benchmark results and generates performance reports
"""

import json
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

def load_json_result(filepath: str) -> Optional[Dict]:
    """Load and parse JSON benchmark result"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load {filepath}: {e}")
        return None

def analyze_performance(result: Dict) -> Dict:
    """Analyze performance metrics from benchmark result"""
    analysis = {}

    if 'results' in result:
        r = result['results']

        # Basic metrics
        analysis['tokens_per_sec'] = r.get('tokens_per_sec', 0)
        analysis['speedup_vs_baseline'] = r.get('speedup_vs_baseline', 0)
        analysis['cv_percent'] = r.get('coefficient_of_variation', 0) * 100
        analysis['memory_efficiency'] = r.get('memory_efficiency_percent', 0)
        analysis['roofline_score'] = r.get('roofline_score', 0)

        # Golden ticket criteria
        analysis['golden_ticket_speedup'] = analysis['speedup_vs_baseline'] >= 2.0
        analysis['golden_ticket_precision'] = analysis['cv_percent'] <= 1.0
        analysis['golden_ticket_overall'] = (analysis['golden_ticket_speedup'] and
                                           analysis['golden_ticket_precision'])

        # Performance classification
        speedup = analysis['speedup_vs_baseline']
        if speedup >= 2.0:
            analysis['performance_class'] = 'Golden Ticket'
        elif speedup >= 1.8:
            analysis['performance_class'] = 'Excellent'
        elif speedup >= 1.5:
            analysis['performance_class'] = 'Good'
        elif speedup >= 1.2:
            analysis['performance_class'] = 'Minor Improvement'
        else:
            analysis['performance_class'] = 'Baseline'

    # Inference cycle analysis
    if 'inference_cycle' in result and result['inference_cycle']:
        ic = result['inference_cycle']
        analysis['inference_speedup'] = ic.get('speedup_vs_baseline', 0)
        analysis['inference_tokens_per_sec'] = ic.get('elastic_tokens_per_sec', 0)
        analysis['inference_baseline_tokens_per_sec'] = ic.get('baseline_tokens_per_sec', 0)
        analysis['real_world_golden_ticket'] = analysis['inference_speedup'] >= 2.0

    return analysis

def generate_summary_report(results_dir: str) -> Dict:
    """Generate comprehensive summary report"""
    results_path = Path(results_dir)
    json_files = list(results_path.glob('*.json'))

    summary = {
        'total_tests': len(json_files),
        'tests': [],
        'overall_analysis': {},
        'recommendations': []
    }

    all_speedups = []
    all_cvs = []
    all_memory_efficiencies = []

    for json_file in json_files:
        result = load_json_result(str(json_file))
        if result:
            analysis = analyze_performance(result)

            test_summary = {
                'filename': json_file.name,
                'configuration': result.get('configuration', {}),
                'analysis': analysis
            }
            summary['tests'].append(test_summary)

            # Collect metrics for overall analysis
            if analysis.get('speedup_vs_baseline', 0) > 0:
                all_speedups.append(analysis['speedup_vs_baseline'])
            if analysis.get('cv_percent', 0) > 0:
                all_cvs.append(analysis['cv_percent'])
            if analysis.get('memory_efficiency', 0) > 0:
                all_memory_efficiencies.append(analysis['memory_efficiency'])

    # Overall analysis
    if all_speedups:
        summary['overall_analysis']['average_speedup'] = np.mean(all_speedups)
        summary['overall_analysis']['max_speedup'] = np.max(all_speedups)
        summary['overall_analysis']['min_speedup'] = np.min(all_speedups)

    if all_cvs:
        summary['overall_analysis']['average_cv'] = np.mean(all_cvs)
        summary['overall_analysis']['best_cv'] = np.min(all_cvs)
        summary['overall_analysis']['worst_cv'] = np.max(all_cvs)

    if all_memory_efficiencies:
        summary['overall_analysis']['average_memory_efficiency'] = np.mean(all_memory_efficiencies)
        summary['overall_analysis']['best_memory_efficiency'] = np.max(all_memory_efficiencies)

    # Generate recommendations
    if summary['overall_analysis'].get('average_cv', 100) > 5:
        summary['recommendations'].append(
            "High coefficient of variation detected. Consider increasing warmup iterations and inner loops."
        )

    if summary['overall_analysis'].get('average_speedup', 0) < 1.5:
        summary['recommendations'].append(
            "Low speedup detected. Try higher compression ratios or larger sequence lengths."
        )

    if summary['overall_analysis'].get('average_memory_efficiency', 0) < 60:
        summary['recommendations'].append(
            "Low memory efficiency. Check GPU utilization and consider optimizing memory access patterns."
        )

    return summary

def print_detailed_report(summary: Dict):
    """Print detailed analysis report"""
    print("="*80)
    print("PHIQ ELASTIC KV CACHE - PERFORMANCE ANALYSIS REPORT")
    print("="*80)
    print()

    # Overall statistics
    print("📊 OVERALL STATISTICS")
    print("-" * 40)
    overall = summary['overall_analysis']

    if 'average_speedup' in overall:
        print(f"Average Speedup: {overall['average_speedup']:.3f}x")
        print(f"Max Speedup: {overall['max_speedup']:.3f}x")
        print(f"Min Speedup: {overall['min_speedup']:.3f}x")

    if 'average_cv' in overall:
        print(f"Average CV: {overall['average_cv']:.2f}%")
        print(f"Best CV: {overall['best_cv']:.2f}%")
        print(f"Worst CV: {overall['worst_cv']:.2f}%")

    if 'average_memory_efficiency' in overall:
        print(f"Average Memory Efficiency: {overall['average_memory_efficiency']:.1f}%")
        print(f"Best Memory Efficiency: {overall['best_memory_efficiency']:.1f}%")

    print()

    # Individual test results
    print("🔬 INDIVIDUAL TEST RESULTS")
    print("-" * 40)

    for test in summary['tests']:
        analysis = test['analysis']
        config = test['configuration']

        print(f"\n📁 {test['filename']}")
        print(f"   Configuration: seq={config.get('seq_len', 'N/A')}, "
              f"heads={config.get('heads', 'N/A')}, "
              f"dim={config.get('head_dim', 'N/A')}, "
              f"compress={config.get('compression', 'N/A')}")

        if 'tokens_per_sec' in analysis:
            print(f"   Tokens/sec: {analysis['tokens_per_sec']:.1f}")

        if 'speedup_vs_baseline' in analysis:
            speedup_icon = "🏆" if analysis['golden_ticket_speedup'] else "⚠️"
            print(f"   Speedup: {analysis['speedup_vs_baseline']:.3f}x {speedup_icon}")

        if 'cv_percent' in analysis:
            cv_icon = "🏆" if analysis['golden_ticket_precision'] else "⚠️"
            print(f"   CV: {analysis['cv_percent']:.2f}% {cv_icon}")

        if 'performance_class' in analysis:
            print(f"   Classification: {analysis['performance_class']}")

        if 'inference_speedup' in analysis:
            print(f"   Inference Cycle Speedup: {analysis['inference_speedup']:.3f}x")

        # Golden ticket status
        if analysis.get('golden_ticket_overall', False):
            print("   Status: 🏆 GOLDEN TICKET ACHIEVED")
        elif analysis.get('real_world_golden_ticket', False):
            print("   Status: 🏆 REAL-WORLD GOLDEN TICKET")
        elif analysis.get('speedup_vs_baseline', 0) >= 1.8:
            print("   Status: ✅ EXCELLENT PERFORMANCE")
        else:
            print("   Status: ⚠️ NEEDS OPTIMIZATION")

    print()

    # Recommendations
    if summary['recommendations']:
        print("💡 RECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(summary['recommendations'], 1):
            print(f"{i}. {rec}")
        print()

    # Golden ticket summary
    golden_tickets = sum(1 for test in summary['tests']
                        if test['analysis'].get('golden_ticket_overall', False))
    real_world_golden = sum(1 for test in summary['tests']
                           if test['analysis'].get('real_world_golden_ticket', False))

    print("🏆 GOLDEN TICKET SUMMARY")
    print("-" * 40)
    print(f"Full Golden Tickets: {golden_tickets}/{summary['total_tests']}")
    print(f"Real-world Golden Tickets: {real_world_golden}/{summary['total_tests']}")

    if golden_tickets > 0 or real_world_golden > 0:
        print("Status: PRODUCTION READY ✅")
    else:
        print("Status: NEEDS OPTIMIZATION ⚠️")

    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Analyze PHIQ Elastic KV Cache benchmark results')
    parser.add_argument('results_dir', help='Directory containing JSON result files')
    parser.add_argument('--output', '-o', help='Output file for JSON summary')
    parser.add_argument('--json-only', action='store_true', help='Output only JSON summary')

    args = parser.parse_args()

    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' not found")
        sys.exit(1)

    # Generate summary
    summary = generate_summary_report(args.results_dir)

    # Output JSON summary if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {args.output}")

    # Print detailed report unless JSON-only mode
    if not args.json_only:
        print_detailed_report(summary)
    else:
        print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()