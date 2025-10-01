# ΦQ™ PHIQ.IO Elastic KV Cache - Notebooks

**Production-grade Jupyter notebooks for demonstration and benchmarking**

---

## 📚 Available Notebooks

### 🎯 **PHIQ_Elastic_KV_GTC_Autocontained.ipynb** (RECOMMENDED)

**Self-contained notebook for GTC 2025 submission and live demos**

- ✅ **Fully portable** - Embeds CUDA source code (no repo clone needed)
- ✅ **Multi-architecture** - Compiles for Pascal (SM 6.1) through Hopper (SM 9.0)
- ✅ **Golden Ticket benchmarks** - 4096×32×128 and 1024×16×64 configs
- ✅ **Production kernel** - Race-free double-buffer + ping-pong CUDA Graphs
- ✅ **Optional baselines** - Transformers (ON) and GGUF (OFF) for comparison
- ✅ **Audit-ready output** - JSON artifacts with full metrics
- ✅ **Social media generator** - Twitter/X and LinkedIn posts ready

**Use Cases:**

- GTC 2025 submission
- Live demonstrations to judges/investors
- Academic presentations
- Quick onboarding for new developers

**How to run:**

1. Upload to Google Colab
2. Runtime → Change runtime type → GPU (T4/L4/A100)
3. Run all cells top-to-bottom
4. Results appear automatically with Golden Ticket analysis

**Runtime:** ~5-7 minutes (with Transformers baseline), ~3-4 minutes (CUDA only)

---

### 📊 **phiq-io-elastic-kv-cache_notebooks_PH.ipynb** (INTERNAL)

**Development notebook with placeholder logos (for private repo testing)**

- Same content as GTC notebook
- Uses placeholder URLs while repo is private
- Will be removed after repo goes public

**Status:** ⚠️ Temporary - delete after repo goes public

---

## 🚀 Quick Start

### For GTC Judges / First-time Users:

```bash
# 1) Download the notebook
wget https://raw.githubusercontent.com/Infolake/phiq-io-elastic-kv-cache/master/notebooks/phiq-io-elastic-kv-cache_notebooks.ipynb
# 2) Upload to Google Colab
# Go to https://colab.research.google.com/
# File → Upload notebook → Select the downloaded file

# 3) Select GPU runtime
# Runtime → Change runtime type → Hardware accelerator: GPU

# 4) Run all cells
# Runtime → Run all (Ctrl+F9)
```

### For Developers (Local):

```bash
# Clone repository
git clone https://github.com/Infolake/phiq-io-elastic-kv-cache.git
cd phiq-io-elastic-kv-cache

# Open with Jupyter
jupyter notebook notebooks/PHIQ_Elastic_KV_GTC_Autocontained.ipynb
```

---

## 📋 Notebook Contents

All production notebooks include:

### 1. **Setup & Configuration**

- GPU detection (`nvidia-smi`, `nvcc`)
- Optional HuggingFace login (for Transformers baseline)
- Control flags (ENABLE_GGUF, ENABLE_TRANSFORMERS_MINI)

### 2. **CUDA Source (Embedded)**

- Complete production kernel (~600 lines)
- Double-buffer race-free implementation
- Ping-pong CUDA Graphs
- Vectorized `float4` loads

### 3. **Multi-Arch Compilation**

- Pascal (SM 6.1), Volta (SM 7.0), Turing (SM 7.5)
- Ampere (SM 8.0, 8.6), Ada (SM 8.9), Hopper (SM 9.0)
- Flags: `-O3 --use_fast_math -lineinfo`

### 4. **Golden Ticket Benchmarks**

- **Config 1:** 4096×32×128, compress=4 (long context)
- **Config 2:** 1024×16×64, compress=2 (standard)
- Paired baseline comparison
- Inference cycle timing (64 decode tokens)
- Statistical CV < 5%

### 5. **Optional Baselines**

- **Transformers:** TinyLlama-1.1B-Chat (FP16, fast)
- **GGUF:** llama-cpp-python (heavier, optional)

### 6. **Results Aggregation**

- Pandas DataFrame with all metrics
- Golden Ticket validation analysis
- Automatic verdict (✅ / ⭐ / ✓)

### 7. **Technical Explanation**

- Problem: Memory bottleneck in LLMs
- Solution: Elastic KV Cache architecture
- Why it matters (developers + researchers)

### 8. **Social Media Generator**

- Twitter/X post (280 chars, #NVIDIAGTC @NVIDIAGTC)
- LinkedIn post (full technical details)
- Auto-save to `social_media_content.txt`

---

## 🎯 Golden Ticket Criteria

Notebooks automatically validate against these thresholds:

| Metric                       | Target | Golden Ticket |
| ---------------------------- | ------ | ------------- |
| **Speedup vs Baseline**      | ≥1.95x | ✅            |
| **Coefficient of Variation** | ≤0.05  | ✅            |
| **Memory Efficiency**        | ≥70%   | ✅            |
| **Roofline Score**           | ≥0.80  | ⭐            |

**Current Achievement:**

- ✅ 1.96x speedup
- ✅ <5% CV (audit-ready)
- ✅ 73.8% memory efficiency
- ⭐ 0.82 roofline score

---

## 🔧 Customization

### Adjust Benchmark Configuration

Edit cell #7 (Controls):

```python
# Change model size
TRANSFORMERS_MODEL = "microsoft/phi-2"  # Larger model

# More decode tokens
DECODE_TOKENS = 256

# Enable GGUF baseline
ENABLE_GGUF = True
GGUF_REPO = "TheBloke/Llama-2-7B-Chat-GGUF"
GGUF_FILE = "llama-2-7b-chat.Q4_K_M.gguf"
```

### Modify Benchmark Parameters

Edit cell #13 (Run benchmarks):

```bash
# Longer context
./elastic_kv_cli --seq=8192 --heads=64 --dim=128 --compress=8 \
  --reps=100 --warmup=50 --inner_loops=128
```

---

## 📊 Expected Outputs

### JSON Artifacts:

- `results_4096_golden_ticket.json` - Long context benchmark
- `results_1024_standard.json` - Standard benchmark
- `transformers_baseline.json` - Optional HF reference
- `gguf_baseline.json` - Optional llama.cpp reference

### Social Media:

- `social_media_content.txt` - Copy-paste ready posts

### Notebook Output:

- Pandas DataFrame with comparative metrics
- Golden Ticket analysis with automatic verdict
- GPU information and build configuration

---

## ⚠️ Troubleshooting

### "nvcc not found"

- **Colab:** Runtime → Change runtime type → GPU
- **Local:** Install CUDA Toolkit 11.8+

### "Out of memory"

- **Colab:** Runtime → Change runtime type → High-RAM: ON
- Reduce `DECODE_TOKENS` to 64 or 32

### "Module not found: transformers"

- Cell #15 auto-installs packages
- Manually run: `!pip install transformers torch`

### Compilation errors

- Check GPU architecture matches `-gencode` flags
- For older GPUs (Maxwell/Kepler): remove SM 8.x and 9.0

---

## 📚 Additional Resources

- **Main README:** `../README.md`
- **Usage Guide:** `../USAGE_GUIDE.md`
- **Scientific Paper:** `../SCIENTIFIC_PAPER.md`
- **Source Code:** `../src/elastic_kv_cli.cu`
- **Build Guide:** `../BUILD_GUIDE.md`

---

## 🏆 Citation

If you use these notebooks in your research, please cite:

```bibtex
@software{phiq_elastic_kv_2025,
  author = {Camargo, Guilherme de},
  title = {PHIQ.IO Elastic KV Cache: Race-Free LLM Inference Acceleration},
  year = {2025},
  organization = {PHIQ.IO Quantum Technologies},
  note = {Camargo Constant: Δ = φ + π = 4.759627}
}
```

---

<div align="center">
<b>ΦQ™ Quantum Deductive Computing</b><br/>
<i>"Geometry doesn't lie; it just waits for us to listen."</i><br/>
Dr. Guilherme de Camargo • Camargo Constant: Δ = φ + π = 4.759627<br/>
© 2025 PHIQ.IO Quantum Technologies
</div>
