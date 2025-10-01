# ΦQ™ PHIQ.IO Elastic KV Cache - Reference Benchmarks

**Production Audit Trail | Golden Ticket Validation**

This directory contains reference benchmark results for audit and reproducibility purposes.

## Benchmark Configuration Standards

### Standard Configurations

1. **Small Workload (GTX 1070 / T4 baseline)**
   - `seq_len=1024, heads=16, head_dim=64, compress=2`
   - Target: 200 reps, 20 warmup, inner_loops=64, truncate=5%

2. **Large Workload (A100 / L4 production)**
   - `seq_len=4096, heads=32, head_dim=128, compress=4`
   - Target: 120 reps, 60 warmup, inner_loops=64, truncate=5%

### Measurement Protocol

All benchmarks use:
- **Double-buffer ping-pong**: Race-free execution with O_prev → O_out alternation
- **CUDA Graphs**: Enabled for minimal launch overhead
- **Trimmed mean**: 5% truncation to remove thermal outliers
- **Paired baseline**: compress=1 measured with identical pipeline
- **Inference cycle**: Sequential decode steps for real-world simulation

### File Naming Convention

```
{gpu_model}_{seq}x{heads}x{dim}_compress{ratio}_{date}.json
```

Examples:
- `gtx1070_1024x16x64_compress2_20251001.json`
- `a100_4096x32x128_compress4_20251001.json`

### Golden Ticket Criteria

- **Speedup vs Baseline**: ≥ 1.95x
- **Coefficient of Variation**: ≤ 5.0%
- **Memory Efficiency**: ≥ 70% of theoretical bandwidth
- **Roofline Score**: ≥ 0.80

### Audit Checklist

Before submitting benchmark results:

- [ ] GPU info captured (name, SM, theoretical bandwidth)
- [ ] Paired baseline included (compress=1)
- [ ] Inference cycle measured (decode_tokens=64+)
- [ ] CV within acceptable range (≤5%)
- [ ] JSON schema validated
- [ ] Git commit SHA recorded in metadata

## Usage

### Generate Reference Benchmark

```bash
./build/elastic_kv_cli \
  --seq=1024 --heads=16 --dim=64 --compress=2 \
  --reps=200 --warmup=20 --inner_loops=64 --truncate=5 \
  --paired-baseline --inference --decode_tokens=64 \
  --json > results/my_gpu_1024x16x64_compress2_$(date +%Y%m%d).json
```

### Validate JSON Output

```bash
python -m json.tool results/your_benchmark.json > /dev/null && echo "Valid JSON"
```

### Compare Baselines

```bash
# Extract speedup values
jq '.results.speedup_vs_baseline' results/*.json
```

---

**Contact**: support@phiq.io | https://phiq.io  
**Camargo Constant**: Δ = φ + π = 4.759627
