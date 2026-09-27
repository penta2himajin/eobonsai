# GEMV decode: occupancy was not the limiter, L1 traffic is

Follow-up to ADR 0006's correction, which put the byte-weighted GEMV at 84-89% of the
~275 GB/s ceiling with an 8-13% gap concentrated in the 5120/6144/10240-row shapes.

## Experiment 1: raise occupancy. Mechanism worked, throughput did not.

The counter evidence pointed at occupancy. Every non-fused ternary decode shape compiled to
**56 registers per thread**, which caps residency at 9 blocks of 128 threads on sm_86, i.e.
~61% occupancy - and DRAM throughput sat at the same ~61%. The fused gate/up instantiation
compiled to **48 registers** and reached 72% occupancy and 79% DRAM, so occupancy looked like
the cause.

`__launch_bounds__`'s second argument was raised for the ternary decode path
(`MMVQ_DECODE_MIN_BLOCKS`). The mechanism did what it was asked:

| | registers | achieved occupancy | DRAM % (5120-row, K=17408) |
|---|---:|---:|---:|
| unpatched | 56 | 61% | 61.0 |
| minBlocks=10 | **48** | **67-75%** | **72.0** |

And end to end, same session, `llama-cli` served shape, three repetitions each:

| build | gen t/s | mean |
|---|---|---:|
| **unpatched** | 29.3, 29.6, 29.5 | **29.47** |
| minBlocks=8 | 28.9, 28.8, 28.5 | 28.73 |
| minBlocks=10 | 29.7, 28.9, 29.2 | 29.27 |
| minBlocks=12 | 29.3, 29.3, 29.2 | 29.27 |

**The patch is neutral at best and slightly negative overall. Reverted.**

This is worth stating plainly: a counter improved by 11 percentage points (DRAM 61 -> 72 on
the target shape, occupancy 61 -> 72) and the token rate did not move. Counter improvement is
not throughput improvement. That is the ADR 0004 and ADR 0005 lesson again, now at the level
of a hardware metric rather than a kernel or a trace.

The same-session requirement also showed itself: an earlier session measured the unpatched
baseline at 28.8 t/s, this one at 29.47 t/s. Cross-session comparisons of ~2% are worthless.

## Experiment 2: the actual limiter is L1, not DRAM

The counters that explain the gap were not DRAM:

| shape | DRAM % | **L1 (l1tex) %** |
|---|---:|---:|
| 5120 rows | 61.0 | **74.2** |
| 6144 rows | 60.8 | 74.0 |
| 10240 rows | 60.1 | **76.3** |
| 12288 rows | 60.2 | **77.5** |
| 17408 rows (fused gate/up) | 79.4 | 73.3 |

Every non-fused shape has **L1 more loaded than DRAM**, by 13-17 points. Only the fused shape
inverts this, and it is the one that reaches 79% DRAM and 278 GB/s. L1 hit rate on these
shapes is 95.8-97.6%.

### Why L1 traffic is ~4x DRAM traffic

Per output row the kernel reads:

- one weight row: `K x 2.125 / 8` bytes, streamed from DRAM and never reused
- the **whole quantised activation**: `K` int8 values plus scale blocks, reused for every row

For `K = 5120` and 5120 output rows that is 7 MB of weight traffic from DRAM against
5120 x ~5.4 KB = **~28 MB of activation traffic served from L1**. The activation is re-read
once per row, so L1 carries roughly four times what DRAM does, and it saturates first. The
97% L1 hit rate is exactly the signature of this: the hits are the activation, not the
weights.

### The fix this implies, and why it is not a one-liner

L1 traffic can only fall if the activation is loaded **once and reused across many rows** -
an activation-stationary blocking. The current kernel is weight-stationary per block: one
block owns `rows_per_cuda_block` rows (1 for the decode path) and walks all of K for each.
Swapping the loop nest so a block loads an activation tile into registers and sweeps it
against many rows' weights would cut activation reads by the blocking factor.

`rows_per_cuda_block = 2` does **not** achieve this: the kernel's row loop re-reads the
activation pointer for each row, so the L1 traffic per row is unchanged. It only shrinks the
grid. That is consistent with the standalone microbenchmark's earlier rpb=2 result being
worse (206 vs 226 GB/s), although that microbench is not a reliable guide (ADR 0004).

A real activation-stationary rewrite touches a template-heavy kernel with fusion paths
(`has_fusion`, `has_gate`) and would need the full parity and served-shape A/B gate. Expected
ceiling: the gap between L1 at ~75% and DRAM at ~60% bounds what is recoverable, so at most
roughly 10-15% of GEMV time, i.e. under 10% of a decoded token.

## What this round established

| lever | result |
|---|---|
| n-gram parameter tuning | **+40%** on context-reusing work (138 -> 224 t/s), shipped |
| `nwarps` (ADR 0004) | neutral in the served shape, reverted |
| launch configuration | swept, closed |
| occupancy via `__launch_bounds__` | mechanism worked, throughput not; reverted |
| L1 traffic | **identified as the limiter**, needs an activation-stationary rewrite |

Served-shape decode for reference: **28.5-29.5 t/s** without speculation (the range is
session drift, which is why all A/Bs here are same-session), and **224 t/s** with the tuned
n-gram speculation on context-reusing work.

Evidence: `results/ncu-served-deep.csv`, `results/ncu-minblocks10.csv`.
