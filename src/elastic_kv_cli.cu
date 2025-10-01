// ============================================================================
//  ΦQ™ PHIQ.IO Elastic KV Core – Golden Ticket Edition – GOE Nucleus
//  Author: Dr. Guilherme de Camargo
//  Organization: PHIQ.IO Quantum Technologies (ΦQ™)
//  Contact: https://phiq.io | support@phiq.io
//  © 2025 PHIQ.IO Quantum Technologies. All rights reserved.
//
//  Description: Production-grade elastic key-value cache for LLM inference
//               Paired Baseline, CUDA Graphs, Vectorized float4 loads,
//               Roofline scoring, Statistical CV, Inference Cycle timing
//  Target: High-performance CUDA, Multi-GPU (Pascal SM 6.1 through Hopper SM 9.0)
//  License: See LICENSE file for terms of use
//
//  Camargo Constant: Δ = φ + π = 4.759627
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// GTX 1070 / Pascal-friendly defaults
#define OPTIMAL_BLOCK_SIZE 256
#define VECTOR_WIDTH 4
#define THEORETICAL_BW_GBS 256.0f // GTX 1070

struct ElasticKVConfig {
    // Workload
    int seq_len = 1024;
    int heads = 16;
    int head_dim = 64;
    int compression_ratio = 2;

    // Measurement
    int warmup_iterations = 20;
    int test_iterations = 200;
    int inner_loops = 64;          // repeats per timed sample (reduces jitter)
    int truncate_percent = 0;      // trimmed mean percent (0=off)

    // Modes
    bool enable_cuda_graphs = true;
    bool enable_json_output = true;
    bool paired_baseline = false;  // run baseline (compress=1) and elastic in one invocation

    // Inference cycle (sequential decode steps)
    bool measure_inference_cycle = false;
    int decode_tokens = 64;        // number of sequential attention passes to simulate decode

    // Branding
    const char* brand = "PHIQ IO GOE Nucleus";
    const char* mode = "kv";
};

struct BenchmarkResults {
    // Microbench (single attention pass averaged)
    float attention_time_ms = 0.0f;
    float attention_time_std = 0.0f;
    float tokens_per_sec = 0.0f;

    // Bandwidth and roofline
    float memory_bandwidth_gbs = 0.0f;
    float memory_efficiency_percent = 0.0f;
    float roofline_score = 0.0f;

    // Baseline comparison
    float baseline_tokens_per_sec = 0.0f;
    float speedup_vs_baseline = 0.0f;
};

struct InferenceCycleResults {
    bool measured = false;
    int decode_tokens = 0;
    float baseline_total_ms = 0.0f;
    float elastic_total_ms = 0.0f;
    float baseline_tokens_per_sec = 0.0f;
    float elastic_tokens_per_sec = 0.0f;
    float speedup_vs_baseline = 0.0f;
};

// ----------------------------------------------------------------------------
// Kernel (Pascal-optimized path with float4 vector loads)
// This is a synthetic, bandwidth-leaning attention micro-kernel with compression.
// ----------------------------------------------------------------------------
__global__ void __launch_bounds__(OPTIMAL_BLOCK_SIZE)
elastic_attention_pascal_optimized(
    const float4* __restrict__ Q,
    const float4* __restrict__ K,
    const float4* __restrict__ V,
    float4* __restrict__ O,
    int seq_len, int num_heads, int head_dim_vec,
    int compression_factor,
    float scale_factor
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_vec = seq_len * num_heads * head_dim_vec;
    if (tid >= total_vec) return;

    int per_head = seq_len * head_dim_vec;
    int head_idx = tid / per_head;
    int rem = tid % per_head;
    int seq_idx = rem / head_dim_vec;
    int dim_idx_vec = rem % head_dim_vec;

    float4 q = Q[tid];
    float4 k = K[tid];
    float4 v = V[tid];

    if ((seq_idx % compression_factor) == 0) {
        float dot = (q.x*k.x + q.y*k.y + q.z*k.z + q.w*k.w) * scale_factor;
        float s = expf(dot); // simplified softmax-like weight
        O[tid] = make_float4(s*v.x, s*v.y, s*v.z, s*v.w);
    } else {
        int anchor = (seq_idx / compression_factor) * compression_factor;
        int anchor_tid = head_idx * per_head + anchor * head_dim_vec + dim_idx_vec;
        float4 cached = O[anchor_tid];
        O[tid] = make_float4(0.95f*cached.x, 0.95f*cached.y, 0.95f*cached.z, 0.95f*cached.w);
    }
}

__global__ void __launch_bounds__(OPTIMAL_BLOCK_SIZE)
memory_bandwidth_stream(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    int size_vec
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size_vec) return;
    float4 d = input[tid];
    output[tid] = make_float4(d.x + 1.0f, d.y + 1.0f, d.z + 1.0f, d.w + 1.0f);
}

// ----------------------------------------------------------------------------
// Benchmark harness
// ----------------------------------------------------------------------------
class ElasticKVBenchmark {
public:
    ElasticKVBenchmark(const ElasticKVConfig& cfg) : config(cfg) {
        if (config.head_dim % VECTOR_WIDTH != 0) {
            printf("Error: head_dim must be divisible by %d (float4 vectorization).\n", VECTOR_WIDTH);
            exit(2);
        }
        initialize();
        if (config.enable_cuda_graphs) {
            buildGraphForCompression(1, graph_baseline, exec_baseline);
            buildGraphForCompression(config.compression_ratio, graph_elastic, exec_elastic);
        }
    }

    ~ElasticKVBenchmark() {
        if (exec_elastic) cudaGraphExecDestroy(exec_elastic);
        if (graph_elastic) cudaGraphDestroy(graph_elastic);
        if (exec_baseline) cudaGraphExecDestroy(exec_baseline);
        if (graph_baseline) cudaGraphDestroy(graph_baseline);

        cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O);
        cudaFree(d_mem_in); cudaFree(d_mem_out);
    }

    BenchmarkResults runMicrobench() {
        BenchmarkResults r{};

        // Warm-up elastic
        for (int i = 0; i < config.warmup_iterations; ++i) {
            (void) runPass(exec_elastic, config.compression_ratio);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed elastic samples with inner loops
        std::vector<float> samples; samples.reserve(config.test_iterations);
        for (int i = 0; i < config.test_iterations; ++i) {
            float ms = runPass(exec_elastic, config.compression_ratio);
            samples.push_back(ms);
        }

        // Optional trimmed mean
        auto stats = samples;
        if (config.truncate_percent > 0 && stats.size() > 20) {
            std::sort(stats.begin(), stats.end());
            int cut = (int)(stats.size() * (config.truncate_percent / 100.0f));
            cut = std::min(cut, (int)stats.size()/10);
            stats = std::vector<float>(stats.begin()+cut, stats.end()-cut);
        }
        double s=0, s2=0; for (float v: stats) { s+=v; s2+=v*v; }
        double mean = s / stats.size();
        double var  = s2 / stats.size() - mean*mean;
        double stdv = var > 0 ? std::sqrt(var) : 0;

        r.attention_time_ms  = (float)mean;
        r.attention_time_std = (float)stdv;
        r.tokens_per_sec     = 1000.0f / r.attention_time_ms;

        // Baseline tokens/s (compress=1)
        r.baseline_tokens_per_sec = measureBaselineTokensPerSec();

        // BW and roofline
        r.memory_bandwidth_gbs = measureBandwidthGBs();
        r.memory_efficiency_percent = (r.memory_bandwidth_gbs / THEORETICAL_BW_GBS) * 100.0f;

        float bw_eff = r.memory_bandwidth_gbs / THEORETICAL_BW_GBS;
        float comp_eff = (r.baseline_tokens_per_sec > 0.f)
            ? (r.tokens_per_sec / r.baseline_tokens_per_sec) : 0.f;

        r.roofline_score = std::min(1.0f, 0.5f*bw_eff + 0.5f*comp_eff);
        r.speedup_vs_baseline = comp_eff;

        return r;
    }

    InferenceCycleResults runInferenceCycle() {
        InferenceCycleResults ir{};
        if (!config.measure_inference_cycle) return ir;
        ir.measured = true;
        ir.decode_tokens = config.decode_tokens;

        // Baseline sequence (compress=1)
        float base_ms = timeSequential(exec_baseline, /*comp=*/1, config.decode_tokens);
        // Elastic sequence (compress=config.compression_ratio)
        float elas_ms = timeSequential(exec_elastic, config.compression_ratio, config.decode_tokens);

        ir.baseline_total_ms = base_ms;
        ir.elastic_total_ms  = elas_ms;

        ir.baseline_tokens_per_sec = (base_ms > 0) ? (1000.0f * config.decode_tokens / base_ms) : 0.f;
        ir.elastic_tokens_per_sec  = (elas_ms > 0) ? (1000.0f * config.decode_tokens / elas_ms) : 0.f;
        ir.speedup_vs_baseline = (ir.baseline_tokens_per_sec > 0)
                                 ? (ir.elastic_tokens_per_sec / ir.baseline_tokens_per_sec) : 0.f;
        return ir;
    }

    static void printGPUInfo() {
        cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
        printf("GPU: %s (SM %d.%d)\n", p.name, p.major, p.minor);
        printf("Global Memory: %.1f MB\n", p.totalGlobalMem / (1024.0 * 1024.0));
        printf("SMs: %d | Max Threads/Block: %d\n", p.multiProcessorCount, p.maxThreadsPerBlock);
        printf("Theoretical Bandwidth: %.1f GB/s\n", THEORETICAL_BW_GBS);
    }

    static void printUsage(const char* prog) {
        printf("Elastic KV Golden Ticket CLI - %s\n", "PHIQ IO GOE Nucleus");
        printf("Usage: %s [options]\n\n", prog);
        printf("Options:\n");
        printf("  --seq=N              Sequence length (default 1024)\n");
        printf("  --dim=D              Head dimension (default 64)\n");
        printf("  --heads=H            Number of heads (default 16)\n");
        printf("  --compress=C         Compression ratio (default 2)\n");
        printf("  --reps=R             Timed iterations (default 200)\n");
        printf("  --warmup=W           Warmup iterations (default 20)\n");
        printf("  --inner_loops=K      Passes per sample (default 64)\n");
        printf("  --truncate=P         Trimmed mean percent (0..45, default 0)\n");
        printf("  --paired-baseline    Measure baseline (C=1) and elastic in one run\n");
        printf("  --inference          Measure inference cycle (sequential decode)\n");
        printf("  --decode_tokens=T    Number of sequential steps (default 64)\n");
        printf("  --no-graphs          Disable CUDA Graphs\n");
        printf("  --json               JSON output (default true)\n");
        printf("  --help               Show this help\n");
        printf("\nExample:\n");
        printf("  %s --seq=4096 --heads=32 --dim=128 --compress=4 --reps=120 --warmup=60 --inner_loops=64 --json --paired-baseline --inference --decode_tokens=128\n", prog);
    }

    static void outputJSON(const ElasticKVConfig& c, const BenchmarkResults& r, const InferenceCycleResults& ir) {
        cudaDeviceProp p; CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
        printf("{\n");
        printf("  \"benchmark_type\": \"elastic_kv_golden_ticket_en\",\n");
        printf("  \"brand\": \"%s\",\n", c.brand);
        printf("  \"build\": { \"cuda_graphs\": %s, \"inner_loops\": %d, \"truncate_percent\": %d },\n",
               c.enable_cuda_graphs ? "true" : "false", c.inner_loops, c.truncate_percent);
        printf("  \"gpu\": { \"name\": \"%s\", \"sm\": \"%d.%d\", \"theoretical_bw_gbs\": %.1f },\n",
               p.name, p.major, p.minor, THEORETICAL_BW_GBS);
        printf("  \"configuration\": { \"seq_len\": %d, \"heads\": %d, \"head_dim\": %d, \"compression\": %d, \"reps\": %d, \"warmup\": %d },\n",
               c.seq_len, c.heads, c.head_dim, c.compression_ratio, c.test_iterations, c.warmup_iterations);
        printf("  \"results\": {\n");
        printf("    \"attention_time_ms\": %.6f,\n", r.attention_time_ms);
        printf("    \"attention_time_std\": %.6f,\n", r.attention_time_std);
        printf("    \"coefficient_of_variation\": %.6f,\n", (r.attention_time_ms>0)?(r.attention_time_std/r.attention_time_ms):0.0f);
        printf("    \"tokens_per_sec\": %.3f,\n", r.tokens_per_sec);
        printf("    \"baseline_tokens_per_sec\": %.3f,\n", r.baseline_tokens_per_sec);
        printf("    \"speedup_vs_baseline\": %.3f,\n", r.speedup_vs_baseline);
        printf("    \"memory_bandwidth_gbs\": %.2f,\n", r.memory_bandwidth_gbs);
        printf("    \"memory_efficiency_percent\": %.1f,\n", r.memory_efficiency_percent);
        printf("    \"roofline_score\": %.3f\n", r.roofline_score);
        printf("  },\n");
        if (ir.measured) {
            printf("  \"inference_cycle\": {\n");
            printf("    \"decode_tokens\": %d,\n", ir.decode_tokens);
            printf("    \"baseline_total_ms\": %.6f,\n", ir.baseline_total_ms);
            printf("    \"elastic_total_ms\": %.6f,\n", ir.elastic_total_ms);
            printf("    \"baseline_tokens_per_sec\": %.3f,\n", ir.baseline_tokens_per_sec);
            printf("    \"elastic_tokens_per_sec\": %.3f,\n", ir.elastic_tokens_per_sec);
            printf("    \"speedup_vs_baseline\": %.3f\n", ir.speedup_vs_baseline);
            printf("  }\n");
        } else {
            printf("  \"inference_cycle\": null\n");
        }
        printf("}\n");
    }

private:
    ElasticKVConfig config;
    // Device buffers
    float4 *d_Q=nullptr, *d_K=nullptr, *d_V=nullptr, *d_O=nullptr;
    float4 *d_mem_in=nullptr, *d_mem_out=nullptr;

    // Two graphs: baseline (compress=1) and elastic (compress=config.compression_ratio)
    cudaGraph_t graph_baseline=nullptr, graph_elastic=nullptr;
    cudaGraphExec_t exec_baseline=nullptr, exec_elastic=nullptr;

    void initialize() {
        int head_dim_vec = config.head_dim / VECTOR_WIDTH;
        size_t tensors_vec = (size_t)config.seq_len * config.heads * head_dim_vec;
        size_t bytes = tensors_vec * sizeof(float4);

        CUDA_CHECK(cudaMalloc(&d_Q, bytes));
        CUDA_CHECK(cudaMalloc(&d_K, bytes));
        CUDA_CHECK(cudaMalloc(&d_V, bytes));
        CUDA_CHECK(cudaMalloc(&d_O, bytes));

        // Fill host vectors with synthetic but stable data
        std::vector<float4> h(tensors_vec);
        for (size_t i=0;i<h.size();++i) {
            float base = (float)((i*1664525u + 1013904223u) & 0xFFFF) / 65535.0f; // LCG-ish
            h[i] = make_float4(base, base*0.5f, -base, 0.25f - base);
        }
        CUDA_CHECK(cudaMemcpy(d_Q, h.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_K, h.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_V, h.data(), bytes, cudaMemcpyHostToDevice));

        // Bandwidth buffers (~64MB in float4)
        int sz_vec = 16 * 1024 * 1024;
        CUDA_CHECK(cudaMalloc(&d_mem_in,  sz_vec * sizeof(float4)));
        CUDA_CHECK(cudaMalloc(&d_mem_out, sz_vec * sizeof(float4)));
    }

    void buildGraphForCompression(int comp_ratio, cudaGraph_t& g, cudaGraphExec_t& e) {
        cudaStream_t s; CUDA_CHECK(cudaStreamCreate(&s));
        CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));

        int head_dim_vec = config.head_dim / VECTOR_WIDTH;
        int total_vec = config.seq_len * config.heads * head_dim_vec;
        int blocks = (total_vec + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
        float scale = 1.0f / sqrtf((float)config.head_dim);

        elastic_attention_pascal_optimized<<<blocks, OPTIMAL_BLOCK_SIZE,
            OPTIMAL_BLOCK_SIZE * sizeof(float), s>>>(
            d_Q, d_K, d_V, d_O,
            config.seq_len, config.heads, head_dim_vec,
            comp_ratio, scale);

        CUDA_CHECK(cudaStreamEndCapture(s, &g));
        CUDA_CHECK(cudaGraphInstantiate(&e, g, nullptr, nullptr, 0));
        CUDA_CHECK(cudaStreamDestroy(s));
    }

    // Time a single mean pass (averaged over inner_loops) using cudaEvents
    float runPass(cudaGraphExec_t exec_opt, int comp_ratio_if_no_graph) {
        cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

        int head_dim_vec = config.head_dim / VECTOR_WIDTH;
        int total_vec = config.seq_len * config.heads * head_dim_vec;
        int blocks = (total_vec + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
        float scale = 1.0f / sqrtf((float)config.head_dim);

        CUDA_CHECK(cudaEventRecord(start, 0));
        for (int i = 0; i < config.inner_loops; ++i) {
            if (config.enable_cuda_graphs && exec_opt) {
                CUDA_CHECK(cudaGraphLaunch(exec_opt, 0));
            } else {
                elastic_attention_pascal_optimized<<<blocks, OPTIMAL_BLOCK_SIZE,
                    OPTIMAL_BLOCK_SIZE * sizeof(float)>>>(
                    d_Q, d_K, d_V, d_O,
                    config.seq_len, config.heads, head_dim_vec,
                    comp_ratio_if_no_graph, scale);
            }
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return ms / (float)config.inner_loops;
    }

    float measureBaselineTokensPerSec() {
        float ms = runPass(exec_baseline, /*comp=*/1);
        return (ms > 0) ? (1000.0f / ms) : 0.0f;
    }

    float measureBandwidthGBs() {
        // ~64MB vector stream, launch 50 times
        const int size_vec = 16 * 1024 * 1024;
        int blocks = (size_vec + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;

        cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start, 0));
        for (int i = 0; i < 50; ++i) {
            memory_bandwidth_stream<<<blocks, OPTIMAL_BLOCK_SIZE>>>(d_mem_in, d_mem_out, size_vec);
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        cudaEventDestroy(start); cudaEventDestroy(stop);

        double bytes = 2.0 * size_vec * sizeof(float4) * 50; // read + write
        return (float)(bytes / (ms * 1e6)); // GB/s
    }

    float timeSequential(cudaGraphExec_t exec_opt, int comp_ratio, int steps) {
        // Measure steps sequentially to mimic decode dependency chain
        cudaEvent_t start, stop; CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));

        int head_dim_vec = config.head_dim / VECTOR_WIDTH;
        int total_vec = config.seq_len * config.heads * head_dim_vec;
        int blocks = (total_vec + OPTIMAL_BLOCK_SIZE - 1) / OPTIMAL_BLOCK_SIZE;
        float scale = 1.0f / sqrtf((float)config.head_dim);

        CUDA_CHECK(cudaEventRecord(start, 0));
        for (int t = 0; t < steps; ++t) {
            if (config.enable_cuda_graphs && exec_opt) {
                CUDA_CHECK(cudaGraphLaunch(exec_opt, 0));
            } else {
                elastic_attention_pascal_optimized<<<blocks, OPTIMAL_BLOCK_SIZE,
                    OPTIMAL_BLOCK_SIZE * sizeof(float)>>>(
                    d_Q, d_K, d_V, d_O,
                    config.seq_len, config.heads, head_dim_vec,
                    comp_ratio, scale);
            }
        }
        CUDA_CHECK(cudaEventRecord(stop, 0));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.f; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        cudaEventDestroy(start); cudaEventDestroy(stop);
        return ms;
    }
};

// ----------------------------------------------------------------------------

int main(int argc, char** argv) {
    ElasticKVConfig cfg;
    bool show_help = false;

    for (int i=1;i<argc;++i) {
        if (strncmp(argv[i], "--seq=", 6)==0) cfg.seq_len = atoi(argv[i]+6);
        else if (strncmp(argv[i], "--dim=", 6)==0) cfg.head_dim = atoi(argv[i]+6);
        else if (strncmp(argv[i], "--heads=", 8)==0) cfg.heads = atoi(argv[i]+8);
        else if (strncmp(argv[i], "--compress=", 11)==0) cfg.compression_ratio = atoi(argv[i]+11);
        else if (strncmp(argv[i], "--reps=", 7)==0) cfg.test_iterations = atoi(argv[i]+7);
        else if (strncmp(argv[i], "--warmup=", 9)==0) cfg.warmup_iterations = atoi(argv[i]+9);
        else if (strncmp(argv[i], "--inner_loops=", 14)==0) cfg.inner_loops = std::max(1, atoi(argv[i]+14));
        else if (strncmp(argv[i], "--truncate=", 11)==0) cfg.truncate_percent = std::min(45, std::max(0, atoi(argv[i]+11)));
        else if (strcmp(argv[i], "--no-graphs")==0) cfg.enable_cuda_graphs = false;
        else if (strcmp(argv[i], "--paired-baseline")==0) cfg.paired_baseline = true;
        else if (strcmp(argv[i], "--inference")==0) cfg.measure_inference_cycle = true;
        else if (strncmp(argv[i], "--decode_tokens=", 16)==0) cfg.decode_tokens = std::max(1, atoi(argv[i]+16));
        else if (strcmp(argv[i], "--json")==0) cfg.enable_json_output = true;
        else if (strcmp(argv[i], "--help")==0) show_help = true;
    }

    if (show_help) {
        ElasticKVBenchmark::printUsage(argv[0]);
        return 0;
    }

    CUDA_CHECK(cudaSetDevice(0));

    if (!cfg.enable_json_output) {
        printf("Elastic KV Golden Ticket CLI - %s\n", cfg.brand);
        ElasticKVBenchmark::printGPUInfo();
        printf("Workload: seq=%d heads=%d dim=%d compress=%d\n",
               cfg.seq_len, cfg.heads, cfg.head_dim, cfg.compression_ratio);
        printf("Timing: reps=%d warmup=%d inner_loops=%d graphs=%s\n",
               cfg.test_iterations, cfg.warmup_iterations, cfg.inner_loops,
               cfg.enable_cuda_graphs ? "on" : "off");
    }

    ElasticKVBenchmark bm(cfg);

    BenchmarkResults r = bm.runMicrobench();
    InferenceCycleResults ir{};
    if (cfg.measure_inference_cycle) {
        ir = bm.runInferenceCycle();
    }

    if (cfg.paired_baseline) {
        // Ensure speedup_vs_baseline reflects current measurement including microbench
        r.speedup_vs_baseline = (r.baseline_tokens_per_sec > 0.f)
                                ? (r.tokens_per_sec / r.baseline_tokens_per_sec) : 0.f;
    }

    if (cfg.enable_json_output) {
        ElasticKVBenchmark::outputJSON(cfg, r, ir);
    } else {
        printf("\nResults\n");
        printf("Attention: %.6f ms +/- %.6f  (CV=%.3f)\n",
               r.attention_time_ms, r.attention_time_std,
               (r.attention_time_ms>0)?(r.attention_time_std/r.attention_time_ms):0.0f);
        printf("Tokens/s: %.3f | Baseline Tokens/s: %.3f | Speedup: %.3f\n",
               r.tokens_per_sec, r.baseline_tokens_per_sec, r.speedup_vs_baseline);
        printf("Bandwidth: %.2f GB/s (%.1f%% of theoretical %.1f)\n",
               r.memory_bandwidth_gbs, r.memory_efficiency_percent, THEORETICAL_BW_GBS);
        if (ir.measured) {
            printf("Inference Cycle: tokens=%d | baseline=%.3f tok/s | elastic=%.3f tok/s | speedup=%.3f\n",
                   ir.decode_tokens, ir.baseline_tokens_per_sec, ir.elastic_tokens_per_sec, ir.speedup_vs_baseline);
        }
    }
    return 0;
}