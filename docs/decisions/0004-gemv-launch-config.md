# ADR 0004: Tuning the ternary GEMV launch configuration, and what the microbench got wrong

- Status: **accepted** (measured 2026-09-27), +2.0% decode, patch kept
- Date: 2026-09-27
- Context: RTX 3060 12 GB, `PQ2_0`, decode is 82.6% ternary GEMV

## The target

`docs/decode-profile.md` established that decode is 34.45 ms/token with the GPU
saturated, and that `mul_mat_vec_q` (the ternary GEMV) is 28.45 ms of it, streaming
7.19 GiB at **253 GB/s against the measured 300 GB/s ceiling, i.e. 84.3%**. That gap was
the largest remaining decode target. The inner loop is already `dp4a` + `__byte_perm`, so
the question was whether the *launch configuration* is wrong for a 28-SM part.

Relevant code: `mmvq.cu:calc_nwarps`. There is no Ampere entry in `mmvq_parameter_table_id`,
so sm_86 falls through to `MMVQ_PARAMETERS_GENERIC`, which returns **4 warps** for
`ncols_dst == 1`. That is the decode case.

## Hypothesis and microbenchmark

If the GEMV were limited by its memory access pattern, fewer warps per row (longer
contiguous runs per warp, cheaper cross-warp reduce) should read faster.
`tools/microbench/gemv.cu` reproduces the real shapes (K=5120, PQ2_0 layout) in two
variants: `load_only` (same access pattern, arithmetic stripped) and `full` (the fork's
unpack + `dp4a`).

| config | load_only | full |
|---|---:|---:|
| nwarps=4 rpb=1 (**the fork's**) | 247.9 GB/s | 235.7 GB/s |
| **nwarps=2 rpb=1** | **281.7 GB/s** | **282.6 GB/s** |
| nwarps=2 rpb=2 / rpb=4 | - | 280.5 / 280.0 |
| nwarps=1 rpb=1 | 239.8 | 260.3 |
| nwarps=8 rpb=1 | 159.6 | 154.4 |

Two readings: the arithmetic is nearly free (`load_only` 247.9 vs `full` 235.7, about 5%),
and **`nwarps=2` looks like a 20% win over the fork's `nwarps=4`**. The prediction for the
model was therefore +11% on the GEMV, roughly +8% on the token.

## The real A/B, and the correction

Same toolchain, same commit, patch applied and reverted, `REPS=3`, `tg128` only
(`results/bench-bin-PQ2_0-kf16vf16-20260927-174517.log` unpatched,
`...-174702.log` patched):

| build | tg128 | tg @ d8192 |
|---|---:|---:|
| unpatched source build | 29.59 ± 0.03 | 28.01 ± 0.04 |
| **+ nwarps=2 for PQ2_0** | **30.19 ± 0.03** | **28.49 ± 0.03** |
| | **+2.0%** | **+1.7%** |

(Cross-check on the method itself: the unpatched source build measured 29.59 against the
pinned prebuilt's 29.55, so same-toolchain A/B is reliable to about 0.5%.)

**The microbench overstated the gain by about 10x.** The reason is visible in its own
numbers: at `nwarps=4` the real kernel sustains 253 GB/s while the microbench's simplified
loop sustains only 236-248 GB/s. The real kernel's K-loop structure is already better than
the one written to represent it, so it had less to gain from a configuration change, and
the sweep measured a deficiency that mostly does not exist in the real kernel.

The lesson generalises and is recorded deliberately: **a launch-configuration sweep in a
simplified standalone kernel does not transfer. Only an in-model A/B counts.** The earlier
roofline microbenchmarks were trustworthy because they measured *ceilings* (bandwidth,
`dp4a`), not configurations.

## Decision

Keep the patch (`patches/0001-mmvq-nwarps-pq2_0-decode.patch`, 5 lines). It is
reproducible, `scripts/build.sh` applies it automatically, and it is numerically clean:
`scripts/parity-check.sh` reports perplexity 5.8953 vs 5.8953 and identical greedy tokens
against the pinned binary (`results/parity-20260927-174751/`). A reliable 2% for a
configuration constant, with no risk, is worth carrying.

## Consequences

- Decode: +2.0% at depth 0, +1.7% at 8K. Under speculation this is the same relative gain.
- **The GEMV's remaining ~16% is not reachable by launch configuration.** The `nwarps`
  axis is swept and closed. What is left is the access pattern itself (1360-byte rows,
  one row per block) and L2/DRAM behaviour, which is a kernel rewrite, not a constant.
- Item (b) of the optimisation objective is therefore closed on the configuration axis.
  The other half of (b) - the ~2.1 ms of norm/quantise plumbing across 730 launches per
  token - is not a configuration change either: it needs graph-level fusion (for example
  quantising a shared rotated activation once instead of once per consumer), and it is
  the next candidate if kernel work continues.
