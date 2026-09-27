// Roofline denominators for the RTX 3060 (GA106, sm_86), measured on this card.
//
// These numbers are the yardsticks the model-level results are compared against:
//   1. read bandwidth  -> the decode floor (weights are streamed once per token)
//   2. launch overhead -> how much of a token can be launch-bound
//   3. dp4a peak       -> the prefill ceiling (the ternary path is int8 4-way dot)
//   4. fp32/fp16x2     -> the CUDA-core compute reference
//   5. FWHT cost       -> the Hadamard pass at the three real activation widths
//
// Nothing here needs the model, so it can be run while the weights download.
//
// Build: nvcc -O3 -arch=sm_86 -o out/roofline tools/microbench/roofline.cu
// Run:   ./out/roofline
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)

static double now_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

struct DevInfo { int sms; int clock_khz; int cc_major, cc_minor; size_t mem_total; };
static DevInfo dev_info() {
    cudaDeviceProp p{};
    CUDA_CHECK(cudaGetDeviceProperties(&p, 0));
    return { p.multiProcessorCount, p.clockRate, p.major, p.minor, p.totalGlobalMem };
}

// ---------------------------------------------------------------- 1. bandwidth
// Streaming read with float4, reduced so the compiler cannot drop the loads.
__global__ void read_bw_kernel(const float4 * __restrict__ src, size_t n4, float * __restrict__ sink) {
    float acc = 0.f;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        const float4 v = __ldg(&src[i]);
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 1234567.f) { *sink = acc; }
}

static void bench_bandwidth(size_t bytes) {
    const size_t n4 = bytes / sizeof(float4);
    float4 * buf = nullptr;
    float * sink = nullptr;
    CUDA_CHECK(cudaMalloc(&buf, n4 * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));
    CUDA_CHECK(cudaMemset(buf, 1, n4 * sizeof(float4)));

    const int block = 256;
    const int grid  = 2048;
    for (int i = 0; i < 3; ++i) read_bw_kernel<<<grid, block>>>(buf, n4, sink);
    CUDA_CHECK(cudaDeviceSynchronize());

    const double t0 = now_ms();
    const int reps = 10;
    for (int i = 0; i < reps; ++i) read_bw_kernel<<<grid, block>>>(buf, n4, sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t1 = now_ms();

    const double gb = (double)bytes * reps / 1e9;
    const double sec = (t1 - t0) / 1e3;
    printf("read bandwidth      : %8.2f GB/s   (%.2f GiB buffer, %d reps)\n",
           gb / sec, bytes / 1073741824.0, reps);
    CUDA_CHECK(cudaFree(buf));
    CUDA_CHECK(cudaFree(sink));
}

// ------------------------------------------------------------ 2. launch overhead
__global__ void empty_kernel(int * p) { if (p && threadIdx.x == 99999) *p = 1; }

static void bench_launch_overhead() {
    int * p = nullptr;
    const int n = 20000;
    for (int i = 0; i < 1000; ++i) empty_kernel<<<1, 32>>>(p);
    CUDA_CHECK(cudaDeviceSynchronize());

    const double t0 = now_ms();
    for (int i = 0; i < n; ++i) empty_kernel<<<1, 32>>>(p);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t1 = now_ms();
    printf("empty launch        : %8.3f us/launch (%d launches, back-to-back)\n",
           (t1 - t0) * 1e3 / n, n);

    // A decode-shaped launch: 256 threads, modest grid, real dependency on prior work.
    const double t2 = now_ms();
    for (int i = 0; i < n; ++i) read_bw_kernel<<<28, 256>>>((const float4 *) nullptr, 0, (float *) p);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t3 = now_ms();
    printf("null-work launch    : %8.3f us/launch (%d launches, 28 blocks x 256)\n",
           (t3 - t2) * 1e3 / n, n);
}

// -------------------------------------------------------------------- 3. dp4a
// 8 independent accumulator chains: dp4a has ~4-6 cycle latency, so 4 chains can
// leave the pipe underfilled and understate the real issue-limited ceiling.
__global__ void dp4a_kernel(const int * __restrict__ a, const int * __restrict__ b, int * __restrict__ out, int iters) {
    int acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0, acc4 = 0, acc5 = 0, acc6 = 0, acc7 = 0;
    const int x = a[threadIdx.x & 31];
    const int y = b[threadIdx.x & 31];
    for (int i = 0; i < iters; ++i) {
        acc0 = __dp4a(x, y, acc0);
        acc1 = __dp4a(x, y, acc1);
        acc2 = __dp4a(x, y, acc2);
        acc3 = __dp4a(x, y, acc3);
        acc4 = __dp4a(x, y, acc4);
        acc5 = __dp4a(x, y, acc5);
        acc6 = __dp4a(x, y, acc6);
        acc7 = __dp4a(x, y, acc7);
    }
    if (acc0 == 0x7fffffff) out[threadIdx.x] = acc1 + acc2 + acc3 + acc4 + acc5 + acc6 + acc7;
}

static void bench_dp4a(const DevInfo & d) {
    int *a, *b, *out;
    CUDA_CHECK(cudaMalloc(&a, 32 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&b, 32 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&out, 256 * sizeof(int)));
    CUDA_CHECK(cudaMemset(a, 1, 32 * sizeof(int)));
    CUDA_CHECK(cudaMemset(b, 1, 32 * sizeof(int)));

    const int iters = 100000;
    const int sms = d.sms;
    dim3 grid(sms * 8), block(256);
    dp4a_kernel<<<grid, block>>>(a, b, out, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    const double t0 = now_ms();
    const int reps = 5;
    for (int i = 0; i < reps; ++i) dp4a_kernel<<<grid, block>>>(a, b, out, iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t1 = now_ms();

    // threads * iters * 4 dp4a ops * 4 MACs * 2 FLOP
    const double threads = (double)grid.x * block.x;
    const double ops = threads * iters * 8.0 * reps;
    const double macs = ops * 4.0;
    printf("dp4a peak           : %8.1f TOPS(int8)  = %.0f GMAC/s  (%d SMs x 8 blocks)\n",
           macs * 2 / ((t1 - t0) / 1e3) / 1e12, macs / ((t1 - t0) / 1e3) / 1e9, sms);
    CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b)); CUDA_CHECK(cudaFree(out));
}

// ------------------------------------------------------- 4. CUDA-core compute
__global__ void fma32_kernel(float * __restrict__ out, int iters) {
    float a0 = threadIdx.x * 1e-8f, a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
    const float b = 1.0000001f, c = 1e-9f;
    for (int i = 0; i < iters; ++i) {
        a0 = __fmaf_rn(a0, b, c);
        a1 = __fmaf_rn(a1, b, c);
        a2 = __fmaf_rn(a2, b, c);
        a3 = __fmaf_rn(a3, b, c);
    }
    if (a0 == 12345.f) out[threadIdx.x] = a1 + a2 + a3;
}

__global__ void fma16x2_kernel(float * __restrict__ out, int iters) {
    __half2 a0 = __float2half2_rn(threadIdx.x * 1e-8f), a1 = a0, a2 = a0, a3 = a0;
    const __half2 b = __float2half2_rn(1.0000001f), c = __float2half2_rn(1e-9f);
    for (int i = 0; i < iters; ++i) {
        a0 = __hfma2(a0, b, c);
        a1 = __hfma2(a1, b, c);
        a2 = __hfma2(a2, b, c);
        a3 = __hfma2(a3, b, c);
    }
    if (__low2float(a0) == 12345.f) out[threadIdx.x] = __low2float(a1) + __low2float(a2) + __low2float(a3);
}

template <typename K>
static void bench_fma(const char * name, K kernel, const DevInfo & d, double flops_per_op, int iters) {
    float * out;
    CUDA_CHECK(cudaMalloc(&out, 256 * sizeof(float)));
    dim3 grid(d.sms * 8), block(256);
    kernel<<<grid, block>>>(out, iters);
    CUDA_CHECK(cudaDeviceSynchronize());

    const double t0 = now_ms();
    const int reps = 5;
    for (int i = 0; i < reps; ++i) kernel<<<grid, block>>>(out, iters);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t1 = now_ms();

    const double ops = (double)grid.x * block.x * iters * 4.0 * reps;
    printf("%-20s: %8.2f TFLOPS   (%.0f GHz x %d SMs x 128 lanes)\n",
           name, ops * flops_per_op / ((t1 - t0) / 1e3) / 1e12, d.clock_khz / 1e6, d.sms);
    CUDA_CHECK(cudaFree(out));
}

// ----------------------------------------------------------------- 5. FWHT
// Verbatim port of fwht_cuda_block<1024, 256, T, true> from the fork's
// ggml/src/ggml-cuda/fwht.cu, with the two ggml helpers stubbed. This is the
// default path at the model's Hadamard block size (1024).
#define FWHT_N 1024
#define FWHT_NT 256
#define PHYS_WARP 32

template <typename T>
__device__ __forceinline__ float fwht_load_stub(const T value) { return (float) value; }
template <>
__device__ __forceinline__ float fwht_load_stub<half>(const half value) { return __half2float(value); }

template <typename T>
__global__ __launch_bounds__(FWHT_NT, 1) void fwht_block_kernel(
        const T * __restrict__ src, float * __restrict__ dst, const int64_t n_rows,
        const float scale, const float * __restrict__ signs, const int n_blk) {
    constexpr int N  = FWHT_N;
    constexpr int NT = FWHT_NT;
    constexpr int NE = N / NT;

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) return;

    src += r * N;
    dst += r * N;

    const int tid  = threadIdx.x;
    const int lane = tid % PHYS_WARP;
    const float * signs_row = signs ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = fwht_load_stub(src[i * NT + tid]) * scale;
        if (signs) reg[i] *= signs_row[i * NT + tid];
    }
#pragma unroll
    for (int h = 1; h < PHYS_WARP; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, PHYS_WARP);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
#pragma unroll
    for (int h = PHYS_WARP; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; ++j) s[j * NT + tid] = reg[j];
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; ++j) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < NE; ++i) dst[i * NT + tid] = reg[i];
}

// rows = tokens * (width / 1024): one block per 1024-wide row, as the fork launches it.
static void bench_fwht(int64_t rows, const char * label) {
    const float * src = nullptr;
    float * dst = nullptr;
    CUDA_CHECK(cudaMalloc((void **) &src, rows * FWHT_N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dst, rows * FWHT_N * sizeof(float)));

    const int block = FWHT_NT;
    for (int i = 0; i < 3; ++i) fwht_block_kernel<float><<<(unsigned) rows, block>>>(src, dst, rows, 0.03125f, nullptr, 1);
    CUDA_CHECK(cudaDeviceSynchronize());

    const double t0 = now_ms();
    const int reps = 50;
    for (int i = 0; i < reps; ++i) fwht_block_kernel<float><<<(unsigned) rows, block>>>(src, dst, rows, 0.03125f, nullptr, 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double t1 = now_ms();

    const double bytes = (double) rows * FWHT_N * sizeof(float) * 2 * reps;
    printf("fwht %-16s: %8.2f us/call  %7.2f GB/s  (%lld blocks)\n",
           label, (t1 - t0) * 1e3 / reps, bytes / ((t1 - t0) / 1e3) / 1e9, (long long) rows);
    CUDA_CHECK(cudaFree((void *) src));
    CUDA_CHECK(cudaFree(dst));
}

int main() {
    const DevInfo d = dev_info();
    printf("device: %d SMs, sm_%d%d, %.0f MHz, %.1f GiB VRAM\n\n",
           d.sms, d.cc_major, d.cc_minor, d.clock_khz / 1000.0, d.mem_total / 1073741824.0);

    bench_bandwidth(4ull * 1024 * 1024 * 1024);
    bench_launch_overhead();
    bench_dp4a(d);
    bench_fma("fp32 fma (cuda core)", fma32_kernel, d, 2.0, 100000);
    bench_fma("fp16x2 fma (cuda core)", fma16x2_kernel, d, 4.0, 100000);

    // FWHT at the three activation widths the model actually rotates
    // (prism.hadamard.sign_widths = [5120, 6144, 17408]), each a multiple of 1024.
    printf("\nFWHT (block 1024, one block per row, as the fork launches it)\n");
    bench_fwht(1 * 5,  "decode w=5120");
    bench_fwht(1 * 17, "decode w=17408");
    bench_fwht(512 * 5,  "pp512 w=5120");
    bench_fwht(512 * 17, "pp512 w=17408");
    return 0;
}
