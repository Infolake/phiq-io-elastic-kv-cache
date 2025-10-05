// ELASTIC KV CORE - Open Source Minimal Implementation
// Optimized for Colab GPUs (T4, V100, A100) with clean C API
// License: Apache-2.0 | Brand: PHIQ IO GOE Nucleus Open-Core
// Compile: nvcc -O3 -std=c++17 -arch=sm_75 -shared -o libelastic_kv_core.so

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <functional>

#define CUDA_CHECK(x) do{auto err=(x); if(err!=cudaSuccess){ \
  printf("CUDA Error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); return err; }}while(0)

#ifndef EKV_BLOCK_SIZE
#define EKV_BLOCK_SIZE 256
#endif

// GPU Architecture Detection for Colab
#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ >= 800  // A100
#define COLAB_GPU_A100
#elif __CUDA_ARCH__ >= 750  // T4
#define COLAB_GPU_T4
#elif __CUDA_ARCH__ >= 700  // V100
#define COLAB_GPU_V100
#endif
#endif

// Core compression kernel - simplified for open source
__global__ void __launch_bounds__(EKV_BLOCK_SIZE)
elastic_kv_compress_core(
    const float* __restrict__ kv_cache,
    float* __restrict__ compressed_cache,
    int* __restrict__ indices,     // may be nullptr
    const float threshold,
    const int total_elements,
    int* __restrict__ compressed_size
){
    namespace cg = cooperative_groups;
    auto block = cg::this_thread_block();

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Phase 0: local predicate evaluation
    int pred = 0;
    float val = 0.0f;
    if (idx < total_elements) {
        val = kv_cache[idx];
        pred = (fabsf(val) > threshold) ? 1 : 0;
    }

    // Phase 1: block-wide reduction using CUB
    typedef cub::BlockReduce<int, EKV_BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage red_storage;
    int block_total = BlockReduce(red_storage).Sum(pred);

    // Reserve global memory space
    __shared__ int base;
    if (threadIdx.x == 0) {
        base = (block_total > 0) ? atomicAdd(compressed_size, block_total) : 0;
    }
    block.sync();

    // Phase 2: exclusive scan for local positioning
    typedef cub::BlockScan<int, EKV_BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage scan_storage;
    int offset;
    BlockScan(scan_storage).ExclusiveSum(pred, offset);

    // Phase 3: write compressed data
    if (pred) {
        const int write_idx = base + offset;
        compressed_cache[write_idx] = val;
        if (indices) indices[write_idx] = idx;
    }
}

// Memory bandwidth benchmark kernel
__global__ void __launch_bounds__(EKV_BLOCK_SIZE)
memory_bandwidth_benchmark(
    float* input,
    float* output,
    int size,
    int iterations
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int iter = 0; iter < iterations; ++iter) {
        for (int i = tid; i < size; i += stride) {
            float v = input[i];
            output[i] = v * 1.01f;  // Simple read + write operation
        }
    }
}

// Compute throughput benchmark kernel
__global__ void __launch_bounds__(EKV_BLOCK_SIZE)
compute_throughput_benchmark(
    float* data,
    int size,
    int ops_per_element
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < size; i += stride) {
        float v = data[i];

        #pragma unroll 4
        for (int op = 0; op < ops_per_element; ++op) {
            v = fmaf(v, 1.1f, 0.1f);           // FMA operation
            v = __sinf(v * 0.1f);              // Transcendental
            v = sqrtf(fabsf(v) + 1e-6f);       // Square root
        }

        data[i] = v;
    }
}

// Timing helper using CUDA events
static float time_kernel_ms(std::function<void(cudaStream_t)> kernel_func, cudaStream_t stream) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    kernel_func(stream);
    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

extern "C" {

// Result structure for metrics
struct ekv_metrics_t {
    float avg_ms;
    float std_ms;
    float cv;         // coefficient of variation (std/avg)
    float gbps;       // for memory benchmark
    float gflops;     // for compute benchmark
};

// Core compression function with statistics
cudaError_t ekv_compress(
    const float* kv_cache,
    float* compressed_cache,
    int* indices,
    float threshold,
    int seq_length,
    int hidden_size,
    int* compressed_size,
    int repetitions,
    int warmup_iterations,
    float* avg_ms,
    float* std_ms
){
    // Input validation
    if (!kv_cache || !compressed_cache || !compressed_size || !avg_ms || !std_ms) {
        return cudaErrorInvalidValue;
    }

    CUDA_CHECK(cudaMemset(compressed_size, 0, sizeof(int)));

    const long long total = (long long)seq_length * hidden_size;
    if (total > INT_MAX) {
        printf("Error: total_elements overflow\n");
        return cudaErrorInvalidValue;
    }

    const int total_elements = (int)total;
    const int block_size = EKV_BLOCK_SIZE;
    int grid_size = (total_elements + block_size - 1) / block_size;
    grid_size = std::min(grid_size, 2048);  // Prevent Colab timeouts

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup iterations
    for (int i = 0; i < warmup_iterations; ++i) {
        CUDA_CHECK(cudaMemsetAsync(compressed_size, 0, sizeof(int), stream));
        elastic_kv_compress_core<<<grid_size, block_size, 0, stream>>>(
            kv_cache, compressed_cache, indices, threshold, total_elements, compressed_size);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Benchmark iterations
    std::vector<float> samples;
    samples.reserve(repetitions);

    for (int i = 0; i < repetitions; ++i) {
        CUDA_CHECK(cudaMemsetAsync(compressed_size, 0, sizeof(int), stream));

        float ms = time_kernel_ms([&](cudaStream_t s) -> void {
            elastic_kv_compress_core<<<grid_size, block_size, 0, s>>>(
                kv_cache, compressed_cache, indices, threshold, total_elements, compressed_size);
        }, stream);

        samples.push_back(ms);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    // Calculate statistics
    double sum = 0, sum_sq = 0;
    for (float v : samples) {
        sum += v;
        sum_sq += v * v;
    }

    double mean = sum / samples.size();
    double variance = sum_sq / samples.size() - mean * mean;
    double std_dev = variance > 0 ? std::sqrt(variance) : 0;

    *avg_ms = (float)mean;
    *std_ms = (float)std_dev;

    return cudaSuccess;
}

// Memory bandwidth benchmark
cudaError_t ekv_mem_bandwidth(
    float* input,
    float* output,
    int size,
    int iterations,
    float* gbps
){
    if (!input || !output || !gbps) {
        return cudaErrorInvalidValue;
    }

    const int block_size = EKV_BLOCK_SIZE;
    int grid_size = std::min((size + block_size - 1) / block_size, 1024);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float ms = time_kernel_ms([&](cudaStream_t s) {
        memory_bandwidth_benchmark<<<grid_size, block_size, 0, s>>>(
            input, output, size, iterations);
        // Don't use CUDA_CHECK here as it returns from the lambda
    }, stream);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaStreamDestroy(stream));

    // Calculate bandwidth: 2 * size * sizeof(float) * iterations / time
    const double bytes_transferred = 2.0 * (double)size * sizeof(float) * iterations;
    *gbps = (float)(bytes_transferred / (ms * 1e6));  // Convert to GB/s

    return cudaSuccess;
}

// Batched memory bandwidth benchmark (TDR-safe)
cudaError_t ekv_mem_bandwidth_batched(
    float* input,
    float* output,
    int size,
    int iters_per_launch,
    int num_launches,
    float* gbps
){
    if (!input || !output || !gbps || num_launches <= 0) {
        return cudaErrorInvalidValue;
    }

    const int block_size = EKV_BLOCK_SIZE;
    int grid_size = std::min((size + block_size - 1) / block_size, 1024);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    double total_ms = 0.0;
    for (int k = 0; k < num_launches; ++k) {
        float ms = time_kernel_ms([&](cudaStream_t s) -> void {
            memory_bandwidth_benchmark<<<grid_size, block_size, 0, s>>>(
                input, output, size, iters_per_launch);
        }, stream);

        // Check for kernel launch errors after each launch
        cudaError_t kernel_err = cudaGetLastError();
        if (kernel_err != cudaSuccess) {
            CUDA_CHECK(cudaStreamDestroy(stream));
            return kernel_err;
        }

        total_ms += ms;
    }

    CUDA_CHECK(cudaStreamDestroy(stream));

    // Calculate bandwidth: 2 * size * sizeof(float) * iters_per_launch * num_launches / total_time
    // Factor of 2 accounts for both read and write operations
    const double bytes_transferred = 2.0 * (double)size * sizeof(float) * (double)iters_per_launch * (double)num_launches;
    
    // Convert bytes to GB (decimal: 1 GB = 10^9 bytes) and ms to seconds
    // bandwidth (GB/s) = bytes / (ms * 1e6) = bytes / (ms * 1e-3 * 1e9)
    *gbps = static_cast<float>(bytes_transferred / (total_ms * 1e6));

    return cudaSuccess;
}

// Compute throughput benchmark
cudaError_t ekv_compute_throughput(
    float* data,
    int size,
    int ops_per_element,
    float* gflops
){
    if (!data || !gflops) {
        return cudaErrorInvalidValue;
    }

    const int block_size = EKV_BLOCK_SIZE;
    int grid_size = std::min((size + block_size - 1) / block_size, 1024);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float ms = time_kernel_ms([&](cudaStream_t s) -> void {
        compute_throughput_benchmark<<<grid_size, block_size, 0, s>>>(
            data, size, ops_per_element);
    }, stream);

    CUDA_CHECK(cudaStreamDestroy(stream));

    // Estimate FLOPS: 4 operations per iteration * ops_per_element * size
    const size_t total_ops = 4ULL * ops_per_element * size;
    *gflops = (float)(total_ops / (ms * 1e6));  // Convert to GFLOPS

    return cudaSuccess;
}

// Inference Cycle: baseline (no compression) vs elastic (with compression)
cudaError_t ekv_inference_cycle(
    const float* kv_cache,
    float* compressed_cache,
    int* indices,
    int seq_length,
    int hidden_size,
    int decode_steps,
    float threshold_elastic,
    float* baseline_total_ms,
    float* elastic_total_ms
){
    if (!kv_cache || !compressed_cache || !baseline_total_ms || !elastic_total_ms) {
        return cudaErrorInvalidValue;
    }

    const long long total = (long long)seq_length * hidden_size;
    if (total > INT_MAX) {
        printf("Error: total_elements overflow\n");
        return cudaErrorInvalidValue;
    }

    const int total_elements = (int)total;
    const int block_size = EKV_BLOCK_SIZE;
    int grid_size = std::min((total_elements + block_size - 1) / block_size, 2048);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    int* d_size;
    CUDA_CHECK(cudaMalloc(&d_size, sizeof(int)));

    // Baseline: threshold = INFINITY (no compression)
    float baseline_ms = time_kernel_ms([&](cudaStream_t s) -> void {
        for (int step = 0; step < decode_steps; ++step) {
            cudaMemsetAsync(d_size, 0, sizeof(int), s);
            elastic_kv_compress_core<<<grid_size, block_size, 0, s>>>(
                kv_cache, compressed_cache, indices, std::numeric_limits<float>::infinity(), total_elements, d_size);
        }
    }, stream);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Elastic: real threshold (with compression)
    float elastic_ms = time_kernel_ms([&](cudaStream_t s) -> void {
        for (int step = 0; step < decode_steps; ++step) {
            cudaMemsetAsync(d_size, 0, sizeof(int), s);
            elastic_kv_compress_core<<<grid_size, block_size, 0, s>>>(
                kv_cache, compressed_cache, indices, threshold_elastic, total_elements, d_size);
        }
    }, stream);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    *baseline_total_ms = baseline_ms;
    *elastic_total_ms = elastic_ms;

    CUDA_CHECK(cudaFree(d_size));
    CUDA_CHECK(cudaStreamDestroy(stream));

    return cudaSuccess;
}

// Get GPU information
cudaError_t ekv_get_gpu_info(
    char* device_name,
    int* compute_major,
    int* compute_minor,
    size_t* total_memory,
    int max_name_length
){
    if (!device_name || !compute_major || !compute_minor || !total_memory) {
        return cudaErrorInvalidValue;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    strncpy(device_name, prop.name, max_name_length - 1);
    device_name[max_name_length - 1] = '\0';

    *compute_major = prop.major;
    *compute_minor = prop.minor;
    *total_memory = prop.totalGlobalMem;

    return cudaSuccess;
}

} // extern "C"
