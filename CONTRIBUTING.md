<div align="center">

![ΦQ Logo](docs/assets/logo-phi-q-icon-100.png)

# ΦQ™ PHIQ.IO Elastic KV Cache — Contributing Guide

**Author:** Dr. Guilherme de Camargo | **Organization:** PHIQ.IO Quantum Technologies (ΦQ™)
**Contact:** https://phiq.io | support@phiq.io

Development Guidelines • Community Standards

**Camargo Constant:** Δ = φ + π = 4.759627

</div>

---

# Contributing to PHIQ Elastic KV Cache

We welcome contributions to the PHIQ Elastic KV Cache project! This document provides guidelines for contributing to make the process smooth and effective for everyone.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and professional in all interactions.

## How to Contribute

### Reporting Issues

1. **Search existing issues** first to avoid duplicates
2. **Use issue templates** when creating new issues
3. **Provide detailed information** including:
   - GPU model and driver version
   - CUDA toolkit version
   - Operating system
   - Steps to reproduce
   - Expected vs. actual behavior

### Submitting Code Changes

1. **Fork the repository** and create a feature branch
2. **Make your changes** following our coding standards
3. **Test thoroughly** using the provided test suite
4. **Update documentation** if needed
5. **Submit a pull request** with a clear description

### Development Workflow

```bash
# 1. Fork and clone the repository
git clone https://github.com/yourusername/elastic-kv-cache.git
cd elastic-kv-cache

# 2. Create a feature branch
git checkout -b feature/your-feature-name

# 3. Make your changes
# ... edit files ...

# 4. Build and test
mkdir build && cd build
cmake ..
make -j$(nproc)
./elastic_kv_cli --seq=1024 --compress=2 --reps=20

# 5. Run test suite
cd ../tests
python quick_test.py

# 6. Commit and push
git add .
git commit -m "Add feature: description of your changes"
git push origin feature/your-feature-name

# 7. Create pull request on GitHub
```

## Coding Standards

### C++/CUDA Code

- **Follow Google C++ Style Guide** with CUDA-specific adaptations
- **Use meaningful variable names** and comments
- **Optimize for Pascal architecture** (GTX 1070 target)
- **Include error checking** for all CUDA calls
- **Document kernel parameters** and performance characteristics

### Example Code Style

```cuda
// Good: Clear naming and error checking
__global__ void __launch_bounds__(OPTIMAL_BLOCK_SIZE)
elastic_attention_pascal_optimized(
    const float4* __restrict__ Q,    // Query tensor (vectorized)
    const float4* __restrict__ K,    // Key tensor (vectorized)
    const float4* __restrict__ V,    // Value tensor (vectorized)
    float4* __restrict__ O,          // Output tensor (vectorized)
    int seq_len,                     // Sequence length
    int num_heads,                   // Number of attention heads
    int head_dim_vec,               // Head dimension (vectorized)
    int compression_factor,          // Elastic compression ratio
    float scale_factor              // Attention scaling factor
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= seq_len * num_heads * head_dim_vec) return;

    // Implementation...
}

// Good: Error checking
CUDA_CHECK(cudaMalloc(&d_buffer, size));
```

### Python Code

- **Follow PEP 8** style guidelines
- **Use type hints** where appropriate
- **Include docstrings** for functions and classes
- **Handle errors gracefully** with appropriate error messages

### Documentation

- **Update README.md** for user-facing changes
- **Update TECHNICAL.md** for implementation details
- **Update BENCHMARKS.md** for performance-related changes
- **Include examples** for new features

## Testing Requirements

### Before Submitting

1. **Build successfully** on target platforms
2. **Pass all existing tests**
3. **Add tests for new features**
4. **Verify performance** doesn't regress

### Test Commands

```bash
# Quick validation
python tests/quick_test.py

# Comprehensive test suite
bash tests/run_comprehensive_tests.sh

# Performance regression test
python tests/analyze_results.py test_results/
```

### Performance Standards

New contributions should maintain or improve:

- **Speedup vs. baseline ≥ 1.5x** (minimum acceptable)
- **Coefficient of variation ≤ 5%** (measurement precision)
- **Memory efficiency ≥ 60%** (on GTX 1070)

## Optimization Guidelines

### Pascal-Specific Optimizations

1. **Use vectorized loads** (float4) when possible
2. **Optimize for 256 threads per block**
3. **Minimize register usage** (target ≤64 registers)
4. **Use shared memory efficiently** (49KB limit)
5. **Leverage tensor core operations** where available

### Performance Best Practices

```cuda
// Preferred: Vectorized access
float4 data = input[tid];
output[tid] = make_float4(data.x + 1.0f, data.y + 1.0f,
                         data.z + 1.0f, data.w + 1.0f);

// Avoid: Scalar access
for (int i = 0; i < 4; ++i) {
    output[tid * 4 + i] = input[tid * 4 + i] + 1.0f;
}
```

## Documentation Standards

### Code Comments

```cuda
/**
 * Elastic attention kernel optimized for Pascal architecture
 *
 * Implements compression anchoring with configurable ratios.
 * Uses float4 vectorization for optimal memory bandwidth.
 *
 * @param Q Query tensor (seq_len * heads * head_dim_vec)
 * @param K Key tensor (same layout as Q)
 * @param V Value tensor (same layout as Q)
 * @param O Output tensor (same layout as Q)
 * @param compression_factor Compression ratio (1=no compression)
 *
 * Performance: ~189 GB/s on GTX 1070 (73.8% efficiency)
 * Register usage: ~25 registers per thread
 */
__global__ void elastic_attention_pascal_optimized(/* params */);
```

### Commit Messages

Use conventional commit format:

```
feat: add support for dynamic compression ratios
fix: resolve memory alignment issue in Pascal kernel
docs: update benchmark methodology documentation
perf: optimize register usage in attention kernel
test: add regression tests for large sequence lengths
```

## Review Process

### Pull Request Requirements

- [ ] **Clear description** of changes and motivation
- [ ] **Performance impact analysis** included
- [ ] **Tests added/updated** for new functionality
- [ ] **Documentation updated** as needed
- [ ] **No breaking changes** without version bump
- [ ] **Passes CI/CD checks**

### Review Criteria

1. **Correctness**: Does the code work as intended?
2. **Performance**: Maintains or improves benchmark results?
3. **Style**: Follows coding standards and conventions?
4. **Testing**: Adequate test coverage for changes?
5. **Documentation**: Clear and accurate documentation?

## Release Process

### Version Numbering

We follow semantic versioning:

- **Major** (X.0.0): Breaking changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes, backward compatible

### Release Checklist

- [ ] All tests pass on supported platforms
- [ ] Performance benchmarks meet standards
- [ ] Documentation is up to date
- [ ] CHANGELOG.md is updated
- [ ] Version numbers are bumped appropriately

## Getting Help

### Community Support

- **GitHub Discussions**: General questions and discussions
- **GitHub Issues**: Bug reports and feature requests
- **Documentation**: Comprehensive guides in `docs/` directory

### Contact

- **Technical Issues**: Create GitHub issue with detailed reproduction steps
- **Performance Questions**: Include benchmark results and GPU information
- **Commercial Support**: Contact enterprise@phiq.io

## Recognition

Contributors will be recognized in:

- **CONTRIBUTORS.md** file
- **Release notes** for significant contributions
- **Documentation** for major features

Thank you for contributing to PHIQ Elastic KV Cache! 🚀

---

**PHIQ IO GOE Nucleus**
_Accelerating the future of Large Language Models_
