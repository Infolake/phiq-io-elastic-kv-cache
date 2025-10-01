<div align="center">

<img src="notebooks/content/logo-phi-q-icon-256.png" alt="PHIQ.IO Logo" width="140"/>

# ΦQ™ PHIQ.IO Elastic KV Cache

**High-Performance Elastic Key-Value Cache for Large Language Models**

**Production-Grade LLM Inference Acceleration**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows-lightgrey.svg)](https://github.com/Infolake/phiq-io-elastic-kv-cache)
[![GPU](https://img.shields.io/badge/GPU-Pascal~Hopper-orange.svg)](#gpu-compatibility)
[![Support](https://img.shields.io/badge/Support-phiq.io-blue.svg)](https://phiq.io)

<small>
PHIQ.IO Quantum Technologies • GOE Nucleus Edition
Developed by Dr. Guilherme de Camargo
</small>

</div>

---

## Camargo Constant

> **Δ = φ + π = 4.759627**  
> _(Golden Ratio + Pi: geometric harmony in entropy optimization)_

---

## 🚀 Quick Start

### Requirements

- NVIDIA GPU with Compute Capability 6.1+ _(Pascal and above)_
- CUDA 11.8 or higher
- CMake 3.18+
- C++17 compatible compiler

### Build & Run

```bash
git clone https://github.com/Infolake/phiq-io-elastic-kv-cache.git
cd phiq-io-elastic-kv-cache
./build.sh

# Quick benchmark
./build/elastic_kv_cli --seq=1024 --compress=2 --json
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
