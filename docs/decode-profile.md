# Decode profile: where the 34.45 ms per token goes

Measured with Nsight Systems on this machine. This closes the open question from
`docs/roofline-rtx3060.md`, which had roughly 9 ms of each decode token unaccounted for
after weight streaming, the Hadamard passes and KV traffic.

## Two limitations, not one

**Second limitation (added 2026-09-28).** The breakdown below comes from a `llama-bench`
decode run, i.e. the `ncols_dst = 4` batched shape that
`docs/gemv-benchmark-artifact.md` later showed is not what a served request uses. Read the
shares as a ranking within that capture, **not** as a measured served-token budget.

## Method and its one limitation

```
nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  -o results/nsys-d8 ./bin/cuda/llama-bench -m <PQ2_0> -ngl 99 -fa on -p 0 -n 8 -d 0 -r 1
nsys export --type sqlite -o results/nsys-d8.sqlite results/nsys-d8.nsys-rep
```

nsys 2023.4.4 against driver 595.91.07 (a ~2 year gap) **silently dropped most kernel
records**: the trace's CUDA runtime span is 2690 ms while its kernel span is only 168 ms,
so 2 complete decode passes were captured out of 64 requested. There is no warning in the
log. Do not read the totals as a full-run total.

The capture is still usable because the passes are identical at depth 0, and because the
per-pass figure checks out independently: **68.9 ms of kernel time over 2 passes =
34.45 ms/pass, against a measured 34.22 ms/token wall clock** (29.22 t/s from
`llama-bench -p 0 -n 64 -d 0`). The pass count is fixed by `gated_delta_net` appearing
exactly 48 times per pass (one per linear-attention layer), which is independent of the
matmul count.

## The decomposition

| Kernel | ms/token | share | calls/token |
|---|---:|---:|---:|
| `mul_mat_vec_q` (ternary GEMV) | 28.45 | 82.6% | 361 |
| `k_get_rows_float_vec` | 0.98 | 2.8% | 48 |
| `gated_delta_net_cuda` | 0.92 | 2.7% | 48 |
| `rms_norm_f32` | 0.80 | 2.3% | 209 |
| `scale_f32` | 0.52 | 1.5% | 48 |
| `quantize_q8_1` | 0.52 | 1.5% | 361 |
| `fwht_cuda_block` (Hadamard) | 0.52 | 1.5% | 258 |
| `mul_mat_vec_f` | 0.38 | 1.1% | 96 |
| `cpy_scalar` | 0.28 | 0.8% | 112 |
| `flash_attn_ext_f16` | 0.20 | 0.6% | 16 |
| about ten more types | ~0.38 | 1.1% | |
| **total** | **34.45** | 100% | ~1967 launches |

Call counts per token are slightly under-counted (361 GEMV calls against the 401 rotated
weights in the model) because the second captured pass is truncated. The time shares are
unaffected to within a percent.

## Three conclusions

**1. Decode is not launch-*enqueue*-bound, but it is launch-latency sensitive.** 34.45 ms
of GPU kernel time against 34.22 ms of wall clock means the GPU is saturated, so CPU-side
enqueue is hidden. That does **not** mean launch handling is free: CUDA graphs are active
during decode and are worth **5.3%**, measured by toggling the fork's own switch
(`GGML_CUDA_DISABLE_GRAPHS=1`: tg128 30.14 -> 28.55, d8192 28.47 -> 27.08). An earlier
version of this document concluded from the saturation argument that CUDA graphs were
"not worth pursuing"; that conclusion was wrong and is corrected here. See ADR 0005, which
also records why three indirect forensic attempts gave the wrong answer before the switch
was found.

**2. The ternary GEMV is DRAM-saturated in the served shape.** This section originally
claimed 84.3% of a 300 GB/s roofline and a 15.7% prize. Two corrections apply: the 84.3%
was measured in `llama-bench`'s shape, not the served one (`docs/gemv-benchmark-artifact.md`),
and the applicable ceiling for the model's row-structured reads is ~275 GB/s, not 300
(ADR 0006). In the served shape the dominant weight shape reads 277.7 GB/s, which is 101% of
that ceiling. **The prize does not exist.** The inner loop already uses `dp4a` with
`__byte_perm` unpacking, L1 hit rate is 96-98%, and occupancy is 59-72%.

**3. The non-GEMV 17.4% is fragmented.** No single kernel exceeds 2.8%, and the biggest
group is normalisation and quantisation plumbing: `rms_norm_f32` (209 calls) +
`scale_f32` (48) + `quantize_q8_1` (361) + `cpy_scalar` (112) = 2.12 ms/token across 730
launches. These are memory-bound small kernels whose cost is dominated by launch latency
rather than throughput; they are the one place where fusion could pay, because unlike the
GEMV they are *not* bandwidth-saturated.

Note that `quantize_q8_1` runs 361 times per token, once per GEMV, even though the
`hadamard_memo` in `llama-graph.cpp` means many GEMVs share the same rotated activation.
Quantising a shared activation once instead of per consumer is a concrete, bounded target.

## What this rules out, and what it invites

| Idea | Verdict |
|---|---|
| CUDA graphs | **Already active and worth 5.3%** (ADR 0005). Nothing left to gain by enabling them |
| Rewriting the GEMV inner loop | At most 13%, and it is already `dp4a`-based |
| Fusing the Hadamard pass | Inert: 0.52 ms/token total (1.5%) |
| GEMV tuning for cache/occupancy | **Swept and closed: +2.0%.** `nwarps=2` beats `nwarps=4` for PQ2_0 at decode, shipped as `patches/0001`; the rest of the 15.7% is not a configuration change (ADR 0004) |
| Fusing norm/quantise plumbing | The largest remaining non-GEMV target, ~2 ms/token across 730 launches |
| Speculative decoding | **Done, and it dwarfs all of the above: 4.9-6.0x.** See ADR 0003 |
