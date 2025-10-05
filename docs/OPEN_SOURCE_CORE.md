# Elastic KV Core - Minimal Open-Source Implementation

## Overview

`elastic_kv_core.cu` is a minimal, open-source implementation of the Elastic KV Cache optimized for Google Colab GPUs (T4, V100, A100). It provides a clean C API for easy integration into other projects.

## Key Features

- **Clean C API**: Easy to integrate with Python (via ctypes) or other languages
- **Minimal Dependencies**: Only requires CUDA Runtime and CUB (included in CUDA Toolkit)
- **Colab-Optimized**: Designed for T4, V100, and A100 GPUs
- **Production-Ready**: Includes proper error handling and validation
- **Benchmarking Tools**: Built-in memory bandwidth and compute throughput tests

## Compilation

### Simple Shared Library

```bash
nvcc -O3 -std=c++17 -arch=sm_75 -shared -o libelastic_kv_core.so src/elastic_kv_core.cu
```

### Multi-Architecture Build (Recommended)

```bash
nvcc -O3 -std=c++17 \
     -gencode arch=compute_70,code=sm_70 \  # V100
     -gencode arch=compute_75,code=sm_75 \  # T4
     -gencode arch=compute_80,code=sm_80 \  # A100
     -shared -o libelastic_kv_core.so \
     src/elastic_kv_core.cu
```

### Using CMake

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make elastic_kv_core
```

## API Reference

### Core Compression Function

```c
cudaError_t ekv_compress(
    const float* kv_cache,        // Input KV cache data
    float* compressed_cache,      // Output compressed data
    int* indices,                 // Optional: indices of kept elements (can be NULL)
    float threshold,              // Compression threshold
    int seq_length,               // Sequence length
    int hidden_size,              // Hidden dimension size
    int* compressed_size,         // Output: number of elements kept
    int repetitions,              // Number of benchmark iterations
    int warmup_iterations,        // Warmup iterations
    float* avg_ms,                // Output: average time in milliseconds
    float* std_ms                 // Output: standard deviation in milliseconds
);
```

### Memory Bandwidth Benchmark

```c
cudaError_t ekv_mem_bandwidth(
    float* input,                 // Input buffer
    float* output,                // Output buffer
    int size,                     // Buffer size in elements
    int iterations,               // Number of iterations
    float* gbps                   // Output: bandwidth in GB/s
);
```

### Batched Memory Bandwidth (TDR-Safe)

```c
cudaError_t ekv_mem_bandwidth_batched(
    float* input,                 // Input buffer
    float* output,                // Output buffer
    int size,                     // Buffer size in elements
    int iters_per_launch,         // Iterations per kernel launch
    int num_launches,             // Number of kernel launches
    float* gbps                   // Output: bandwidth in GB/s
);
```

### Compute Throughput Benchmark

```c
cudaError_t ekv_compute_throughput(
    float* data,                  // Data buffer
    int size,                     // Buffer size in elements
    int ops_per_element,          // Operations per element
    float* gflops                 // Output: throughput in GFLOPS
);
```

### Inference Cycle Comparison

```c
cudaError_t ekv_inference_cycle(
    const float* kv_cache,        // Input KV cache data
    float* compressed_cache,      // Temporary buffer
    int* indices,                 // Optional: indices buffer (can be NULL)
    int seq_length,               // Sequence length
    int hidden_size,              // Hidden dimension size
    int decode_steps,             // Number of decode steps
    float threshold_elastic,      // Compression threshold for elastic version
    float* baseline_total_ms,     // Output: baseline time (no compression)
    float* elastic_total_ms       // Output: elastic time (with compression)
);
```

### GPU Information

```c
cudaError_t ekv_get_gpu_info(
    char* device_name,            // Output: device name buffer
    int* compute_major,           // Output: compute capability major version
    int* compute_minor,           // Output: compute capability minor version
    size_t* total_memory,         // Output: total GPU memory in bytes
    int max_name_length           // Size of device_name buffer
);
```

## Python Integration Example

```python
import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./libelastic_kv_core.so')

# Define function signatures
lib.ekv_compress.argtypes = [
    ctypes.POINTER(ctypes.c_float),  # kv_cache
    ctypes.POINTER(ctypes.c_float),  # compressed_cache
    ctypes.POINTER(ctypes.c_int),    # indices
    ctypes.c_float,                  # threshold
    ctypes.c_int,                    # seq_length
    ctypes.c_int,                    # hidden_size
    ctypes.POINTER(ctypes.c_int),    # compressed_size
    ctypes.c_int,                    # repetitions
    ctypes.c_int,                    # warmup_iterations
    ctypes.POINTER(ctypes.c_float),  # avg_ms
    ctypes.POINTER(ctypes.c_float),  # std_ms
]
lib.ekv_compress.restype = ctypes.c_int

# Example usage
seq_len = 1024
hidden = 64
data = np.random.randn(seq_len * hidden).astype(np.float32)
compressed = np.zeros(seq_len * hidden, dtype=np.float32)
compressed_size = ctypes.c_int(0)
avg_ms = ctypes.c_float(0.0)
std_ms = ctypes.c_float(0.0)

# Allocate GPU memory (using pycuda or similar)
# ... GPU memory allocation code here ...

# Call the function
result = lib.ekv_compress(
    data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    compressed.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    None,  # No indices
    0.1,   # threshold
    seq_len,
    hidden,
    ctypes.byref(compressed_size),
    100,   # repetitions
    20,    # warmup_iterations
    ctypes.byref(avg_ms),
    ctypes.byref(std_ms)
)

if result == 0:  # cudaSuccess
    print(f"Compression completed in {avg_ms.value:.2f} ± {std_ms.value:.2f} ms")
    print(f"Compressed size: {compressed_size.value} elements")
```

## Architecture Detection

The library automatically detects the GPU architecture at runtime:

- **COLAB_GPU_A100**: Compute capability 8.0+ (A100, H100)
- **COLAB_GPU_T4**: Compute capability 7.5 (T4)
- **COLAB_GPU_V100**: Compute capability 7.0 (V100)

## Performance Optimization

### Block Size

The default block size is 256 threads. You can customize it by defining `EKV_BLOCK_SIZE` before compilation:

```bash
nvcc -DEKV_BLOCK_SIZE=512 -O3 -std=c++17 -arch=sm_75 -shared -o libelastic_kv_core.so src/elastic_kv_core.cu
```

### Grid Size Limiting

To prevent Colab timeouts, the grid size is automatically limited to 2048 blocks. This can be adjusted in the source code if needed.

## Differences from Production Version

The minimal open-source implementation differs from `elastic_kv_cli.cu` in the following ways:

1. **Simplified API**: Clean C API instead of complex CLI with JSON output
2. **No CUDA Graphs**: Direct kernel launches for simplicity
3. **No Vectorized Loads**: Uses standard float instead of float4 for simplicity
4. **Colab-Focused**: Optimized for common Colab GPUs (T4, V100, A100)
5. **Minimal Features**: Core compression functionality without advanced features

## License

Apache-2.0 License

## Brand

PHIQ IO GOE Nucleus Open-Core

---

**Camargo Constant:** Δ = φ + π = 4.759627
