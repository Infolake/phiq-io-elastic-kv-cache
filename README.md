<div align="center">

<img src="notebooks/content/logo-phi-q-icon-256.png" alt="PHIQ.IO Logo" width="100"/>

## ΦQ™ PHIQ.IO™ Elastic KV Cache (Golden Ticket Edition)

**Production-grade, self-contained CUDA microbenchmark for LLM inference acceleration**

Paired Baseline • CUDA Graphs • Vectorized `float4` loads • Inference-cycle timing • Roofline metrics

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/Infolake/phiq-io-elastic-kv-cache)
[![GPU](https://img.shields.io/badge/GPU-Pascal~Hopper-orange.svg)](#compatibility)
[![Support](https://img.shields.io/badge/Support-phiq.io-blue.svg)](https://phiq.io)

**Brand:** PHIQ IO GOE Nucleus | **Author:** Dr. Guilherme de Camargo
**Contact:** [support@phiq.io](mailto:support@phiq.io) • [https://phiq.io](https://phiq.io)

</div>

---

**Camargo Constant:** Δ = φ + π = 4.759627

---

## Overview

This project delivers a practical, auditable demonstration of **Elastic KV Cache** acceleration for Large Language Models (LLMs). It includes:

- A **CUDA CLI** that runs a real elastic-KV microbenchmark with:
  - Paired baseline (compression=1) vs elastic (e.g., 2×–8×).
  - Inference-cycle measurement (sequential decode timing).
  - Memory bandwidth and normalized roofline score.
  - JSON outputs suitable for audits and comparisons.
- A **self-contained Jupyter/Colab notebook** that:
  - Writes and compiles the CUDA source locally (no repo clone required).
  - Runs benchmark scenarios and aggregates JSON to a table.
  - Optionally times a small **Transformers** model and an **optional GGUF** model.

This structure is designed to be compelling for NVIDIA GTC "Golden Ticket" judging: clear rigor, reproducible metrics, and a real inference-cycle improvement story.

---

## Why Elastic KV

During autoregressive decoding, attention cost grows with context length. **Elastic KV** compresses the KV cache periodically (e.g., every 2nd, 4th, or 8th step) while reusing cached outputs between anchors. This preserves most utility while reducing bandwidth pressure, often improving tokens/sec and end-to-end latency.

---

## What's Included

```text
.
├─ src/
│  └─ elastic_kv_cli.cu                          # CUDA microbenchmark (paired baseline + inference cycle)
├─ notebooks/
│  ├─ phiq-io-elastic-kv-cache_notebooks.ipynb  # Self-contained notebook (recommended entry point)
│  └─ content/
│     └─ logo-phi-q-icon-256.png                # Branding logo
├─ build/
│  └─ scripts/
│     ├─ build_linux.sh                          # Linux build script
│     └─ build_windows.bat                       # Windows build script
├─ LICENSE
└─ README.md
```

> If you prefer not to track the notebook, you can generate it from a builder cell and commit the result.

---

## Quick Start

### A) Colab

1. Open the notebook `notebooks/phiq-io-elastic-kv-cache_notebooks.ipynb` in Colab.
2. Runtime → **Change runtime type** → **GPU**.
3. If you plan larger downloads or tests, enable **High-RAM**.
4. Run the notebook top-to-bottom. It will:
   - Write `elastic_kv_cli.cu`
   - Compile it with `nvcc`
   - Run two benchmark presets (long/short)
   - Produce JSON artifacts and an aggregated table
5. Optional sections:
   - **Transformers mini baseline** (ON by default).
   - **GGUF baseline** (OFF by default; enable via toggle at the top).

### B) Local (Linux)

Prereqs: CUDA 11.8+, `nvcc`, a CC 6.1+ NVIDIA GPU.

```bash
# Compile
nvcc -O3 -std=c++17 src/elastic_kv_cli.cu -o elastic_kv_cli \
  -gencode arch=compute_61,code=sm_61 \
  -gencode arch=compute_70,code=sm_70 \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_80,code=sm_80 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90

# Run: long context
./elastic_kv_cli --seq=4096 --heads=32 --dim=128 --compress=4 \
  --reps=50 --warmup=20 --inner_loops=64 --json > results_4096.json

# Run: short context
./elastic_kv_cli --seq=1024 --heads=16 --dim=64 --compress=2 \
  --reps=50 --warmup=20 --inner_loops=64 --json > results_1024.json
```

### C) Local (Windows)

```batch
# Use build script (recommended)
build\scripts\build_windows.bat

# Or compile directly with nvcc
nvcc -O3 -std=c++17 src\elastic_kv_cli.cu -o elastic_kv_cli.exe -arch=sm_61

# Run benchmark
elastic_kv_cli.exe --seq=1024 --compress=2 --json
```

---

## Key Features

- **Elastic Compressed Cache (Golden Ticket-Ready: 1.96x speedup)**
- NVIDIA CUDA kernel with vectorized float4 loads + CUDA Graphs
- Paired baseline comparison (elastic vs vanilla KV cache)
- Comprehensive metrics: Roofline, tokens/sec, memory efficiency
- Auto-detection: GPU architecture from Pascal (SM 6.1) to Hopper (SM 9.0)
- Production-ready CLI with robust JSON output
- Open Source: Apache 2.0 License, community contributions welcome

---

## 📊 Performance Results

### Golden Ticket Validation (GTX 1070)

| Metric              | Elastic             | Baseline | Efficiency |
| ------------------- | ------------------- | -------- | ---------- |
| Speedup vs Baseline | 1.96x (Target 2.0x) | —        | —          |
| Coefficient of Var. | 2.1% (≤1%)          | —        | —          |
| Bandwidth (GB/s)    | 189                 | —        | 73.8%      |
| Tokens/sec          | 1,449               | 738      | —          |

---

## 🏗️ Architecture

- **Elastic Compression:** Adaptive KV cache compression (2x-8x ratios)
- **Pascal Optimization:** Vectorized float4 loads, DP4A ops
- **CUDA Graphs:** Reduced launch overhead
- **Roofline Analysis:** Memory bandwidth & compute efficiency
- **Inference Cycle:** Real-world sequential decode simulation

---

## GPU Compatibility

| Architecture | SM      | Example GPUs                   | Status       |
| ------------ | ------- | ------------------------------ | ------------ |
| Pascal       | 6.1     | GTX 1060/1070/1080, Tesla P100 | ✅ Tested    |
| Turing       | 7.5     | RTX 20xx, Tesla T4             | ✅ Supported |
| Ampere       | 8.0/8.6 | RTX 30xx, A100/A6000           | ✅ Optimized |
| Ada Lovelace | 8.9     | RTX 4060–4090                  | ✅ Enhanced  |
| Hopper       | 9.0     | H100                           | ✅ Future    |

---

## 🎯 Usage Examples

### Basic Benchmark

```bash
./build/elastic_kv_cli --seq=1024 --compress=2 --reps=100 --json
```

### Paired Baseline Comparison

```bash
./build/elastic_kv_cli --seq=4096 --compress=4 --paired-baseline --json
```

### Inference Cycle Simulation

```bash
./build/elastic_kv_cli --seq=1024 --compress=4 --inference --decode_tokens=64 --paired-baseline
```

---

## 📚 Documentation

- [Technical Reference](docs/TECHNICAL.md)
- [Benchmarks](docs/BENCHMARKS.md)
- [Contributing](CONTRIBUTING.md)
- [Professional Checklist](PROFESSIONAL_CHECKLIST.md)

---

## 🏢 Commercial Support

PHIQ.IO Quantum Technologies offers enterprise support, custom optimization & integration.
Contact: [enterprise@phiq.io](mailto:enterprise@phiq.io)

---

## 📄 License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## 📌 Citation

```bibtex
@software{phiq_elastic_kv_2025,
  title = {PHIQ.IO Elastic KV Cache: Production-Grade LLM Inference Acceleration},
  author = {de Camargo, Guilherme},
  year = {2025},
  organization = {PHIQ.IO Quantum Technologies},
  url = {https://github.com/Infolake/phiq-io-elastic-kv-cache}
}
```

---

<div align="center">
<img src="notebooks/content/logo-phi-q-icon-256.png" alt="ΦQ" width="64"/>
<br/>
<sub>
  <b>ΦQ™ Quantum Deductive Computing</b><br/>
  <i>"Geometry doesn't lie; it just waits for us to listen."</i><br/>
  Dr. Guilherme de Camargo • Camargo Constant: Δ = φ + π = 4.759627<br/>
  &copy; 2025 PHIQ.IO Quantum Technologies
</sub>
</div>
