// GEMV microbenchmark for the shapes Bonsai 2 27B actually uses on this card.
//
// Question it answers: the ternary GEMV (mul_mat_vec_q) reaches 84.3% of the measured
// 300 GB/s streaming ceiling. Is the missing 15.7% the memory access pattern (fixable by
// changing the launch configuration) or the unpack+dp4a arithmetic (not fixable that way)?
//
// Two kernels over the same PQ2_0-layout weights:
//   load_only : same access pattern, arithmetic stripped (sum the loaded words)
//   full      : the fork's real inner loop (__byte_perm unpack + dp4a)
// If load_only also lands near 253 GB/s the pattern is the limit and the (nwarps,
// rows_per_block) sweep is worth acting on. If load_only reaches ~300 GB/s, the
// arithmetic is the limit and no launch configuration will recover it.
//
// Build: nvcc -O3 -arch=sm_86 -o out/gemv tools/microbench/gemv.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

// PQ2_0: 128 elements per group, one fp16 scale + 2 bits per element = 34 bytes.
// K must be a multiple of 128. bytes_per_row = K/128*34.
static constexpr int GROUP = 128;
static constexpr int BYTES_PER_GROUP = 2 + GROUP / 4;

__device__ __forceinline__ int unpack_lo(int q) {
    const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
    const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
    return __byte_perm(qe, qo, 0x5140);
}
__device__ __forceinline__ int unpack_hi(int q) {
    const int qe = __byte_perm(0x020100FF, 0x020100FF, q >> 0);
    const int qo = __byte_perm(0x020100FF, 0x020100FF, q >> 2);
    return __byte_perm(qe, qo, 0x7362);
}

// One block handles RPB output rows of K elements; NWARPS warps split K.
template <int NWARPS, int RPB, bool FULL>
__global__ __launch_bounds__(NWARPS * 32, 1) void gemv_kernel(
        const uint8_t * __restrict__ w,   // rows of K elements in PQ2_0 layout
        const int8_t  * __restrict__ act, // K int8 activations (shared by all rows)
        float * __restrict__ out, int K, long n_rows) {
    constexpr int THREADS = NWARPS * 32;
    __shared__ float red[RPB][NWARPS];

    const long row0 = (long) blockIdx.x * RPB;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    // words per row: 4 bytes each = 16 elements
    const int words_per_row = K / 16;
    // each thread walks words with a warp-strided pattern, one row at a time
    const int words_per_warp = (words_per_row + NWARPS - 1) / NWARPS;

    for (int r = 0; r < RPB; ++r) {
        const long row = row0 + r;
        if (row >= n_rows) break;
        const uint8_t * __restrict__ wr = w + row * (long)(K / GROUP) * BYTES_PER_GROUP;

        int sumi = 0;
        const int w0 = warp * words_per_warp;
        const int w1 = min(w0 + words_per_warp, words_per_row);
        if (FULL) {
            for (int i = w0 + lane; i < w1; i += 32) {
                const int q = ((const int *) wr)[i];       // 4 bytes = 16 trits
                const int e = i * 16;
                const int u = ((const int *) (act + e))[0];
                const int v = ((const int *) (act + e))[4];
                sumi = __dp4a(u, unpack_lo(q), sumi);
                sumi = __dp4a(v, unpack_hi(q), sumi);
            }
        } else {
            for (int i = w0 + lane; i < w1; i += 32) {
                sumi += ((const int *) wr)[i];
            }
        }
        // warp reduction
        for (int o = 16; o > 0; o /= 2) sumi += __shfl_down_sync(0xFFFFFFFF, sumi, o);
        if (lane == 0) red[r][warp] = (float) sumi;
        __syncthreads();
        if (tid == 0) {
            float s = 0.f;
            for (int i = 0; i < NWARPS; ++i) s += red[r][i];
            if (row < n_rows) out[row] = s;
        }
        __syncthreads();
    }
}

static double now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

template <int NWARPS, int RPB, bool FULL>
static void run(const char * label, const uint8_t * w, const int8_t * act, float * out,
                int K, long n_rows, double bytes) {
    dim3 grid((unsigned) ((n_rows + RPB - 1) / RPB));
    for (int i = 0; i < 2; ++i) gemv_kernel<NWARPS, RPB, FULL><<<grid, NWARPS * 32>>>(w, act, out, K, n_rows);
    CHECK(cudaDeviceSynchronize());
    const double t0 = now_ms();
    const int reps = 20;
    for (int i = 0; i < reps; ++i) gemv_kernel<NWARPS, RPB, FULL><<<grid, NWARPS * 32>>>(w, act, out, K, n_rows);
    CHECK(cudaDeviceSynchronize());
    const double ms = (now_ms() - t0) / reps;
    printf("%-26s %7.2f ms  %7.1f GB/s  (nwarps=%d rpb=%d, %ld blocks)\n",
           label, ms, bytes / (ms / 1e3) / 1e9, NWARPS, RPB, (long) ((n_rows + RPB - 1) / RPB));
}

// Pure streaming read with the GEMV's access geometry: each block walks one row of
// bytes_per_row, blocks stride across rows. Isolates DRAM efficiency at 1360-byte rows
// from everything else the GEMV does.
__global__ __launch_bounds__(128, 1) void row_stream_kernel(
        const uint4 * __restrict__ w, int u4_per_row, long n_rows, int * __restrict__ sink) {
    const long row = blockIdx.x;
    if (row >= n_rows) return;
    const uint4 * __restrict__ r = w + row * (long) u4_per_row;
    unsigned acc = 0;
    for (int i = threadIdx.x; i < u4_per_row; i += blockDim.x) {
        const uint4 v = __ldg(&r[i]);
        acc += v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xDEADBEEFu) sink[threadIdx.x] = (int) acc;
}

static void bench_row_stream(int u4_per_row, const char * label) {
    const long n_rows = 200000;
    const double bytes = (double) n_rows * u4_per_row * 16;
    uint4 * w = nullptr; int * sink = nullptr;
    CHECK(cudaMalloc(&w, (size_t) bytes));
    CHECK(cudaMalloc(&sink, 128 * sizeof(int)));
    CHECK(cudaMemset(w, 0x5A, (size_t) bytes));
    for (int i = 0; i < 2; ++i) row_stream_kernel<<<(unsigned) n_rows, 128>>>(w, u4_per_row, n_rows, sink);
    CHECK(cudaDeviceSynchronize());
    const double t0 = now_ms();
    const int reps = 10;
    for (int i = 0; i < reps; ++i) row_stream_kernel<<<(unsigned) n_rows, 128>>>(w, u4_per_row, n_rows, sink);
    CHECK(cudaDeviceSynchronize());
    const double ms = (now_ms() - t0) / reps;
    printf("%-38s %7.2f ms  %7.1f GB/s  (row = %d B)\n",
           label, ms, bytes / (ms / 1e3) / 1e9, u4_per_row * 16);
    CHECK(cudaFree(w)); CHECK(cudaFree(sink));
}

int main() {
    // The dominant shapes in the model: ffn_gate/up are [5120, 17408] and ffn_down is
    // [17408, 5120]. Use K=5120 with many rows so the read is large enough to stream.
    const int K = 5120;
    const long n_rows = 200000;                     // ~1.36 GB at 1360 B/row
    const double bytes = (double) n_rows * (K / GROUP) * BYTES_PER_GROUP;

    uint8_t * w = nullptr; int8_t * act = nullptr; float * out = nullptr;
    CHECK(cudaMalloc(&w, (size_t) bytes));
    CHECK(cudaMalloc(&act, K));
    CHECK(cudaMalloc(&out, n_rows * sizeof(float)));
    CHECK(cudaMemset(w, 0x5A, (size_t) bytes));
    CHECK(cudaMemset(act, 1, K));

    printf("GEMV shape: K=%d, rows=%ld, %.2f GB of weights\n\n", K, n_rows, bytes / 1e9);
    printf("-- pure streaming at GEMV row geometry (is DRAM efficiency the limit?) --\n");
    bench_row_stream(85,  "row = 1360 B (PQ2_0 K=5120)");
    bench_row_stream(290, "row = 4640 B (PQ2_0 K=17408)");
    bench_row_stream(1024,"row = 16384 B (large contiguous)");
    bench_row_stream(16,  "row = 256 B (very short)");
    printf("\n");
    printf("-- memory only (access pattern) --\n");
    run<4, 1, false>("load_only  nwarps=4 rpb=1", w, act, out, K, n_rows, bytes);
    run<2, 1, false>("load_only  nwarps=2 rpb=1", w, act, out, K, n_rows, bytes);
    run<8, 1, false>("load_only  nwarps=8 rpb=1", w, act, out, K, n_rows, bytes);
    run<4, 2, false>("load_only  nwarps=4 rpb=2", w, act, out, K, n_rows, bytes);
    run<4, 4, false>("load_only  nwarps=4 rpb=4", w, act, out, K, n_rows, bytes);
    run<1, 1, false>("load_only  nwarps=1 rpb=1", w, act, out, K, n_rows, bytes);

    printf("\n-- full inner loop (unpack + dp4a, as the fork runs it) --\n");
    run<4, 1, true>("full       nwarps=4 rpb=1", w, act, out, K, n_rows, bytes);
    run<2, 1, true>("full       nwarps=2 rpb=1", w, act, out, K, n_rows, bytes);
    run<2, 2, true>("full       nwarps=2 rpb=2", w, act, out, K, n_rows, bytes);
    run<2, 4, true>("full       nwarps=2 rpb=4", w, act, out, K, n_rows, bytes);
    run<1, 1, true>("full       nwarps=1 rpb=1", w, act, out, K, n_rows, bytes);
    run<1, 2, true>("full       nwarps=1 rpb=2", w, act, out, K, n_rows, bytes);
    run<8, 1, true>("full       nwarps=8 rpb=1", w, act, out, K, n_rows, bytes);

    CHECK(cudaFree(w)); CHECK(cudaFree(act)); CHECK(cudaFree(out));
    return 0;
}
