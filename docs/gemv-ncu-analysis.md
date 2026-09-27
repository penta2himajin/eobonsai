# Decode GEMV: what ncu adds beyond nsys

`docs/decode-profile.md` established the *shape* of the decode budget with nsys: 82.6% in
`mul_mat_vec_q`, and the remainder fragmented. nsys reports durations, which was enough to
rank targets but not to explain them. With hardware counters now available (the machine was
rebooted with `NVreg_RestrictProfilingToAdminUsers=0`, so `ncu` no longer needs root), each
GEMV shape can be asked *why* it is slow.

## Method

```
ncu --target-processes all --kernel-name regex:mul_mat_vec_q --launch-count 40 --launch-skip 700 \
  --metrics gpu__time_duration.sum,dram__bytes.sum,dram__throughput...,lts__t_sector_hit_rate.pct,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,launch__registers_per_thread \
  bin/cuda/llama-cli -m <PQ2_0> -ngl 99 -fa on -st --no-warmup -n 3 -p Hello --spec-type none
```

Raw output: `results/ncu-gemv-shapes.csv`.

## Result: the GEMV is not one thing

40 launches, grouped by grid size, which is the row count of the weight being read:

| grid (rows) | launches | us/launch | MB/launch | **GB/s** | L2 hit | long_scoreboard |
|---:|---:|---:|---:|---:|---:|---:|
| 2560 | 13 | **190.8** | 18.35 | **96.2** | **61.2%** | **2.81** |
| 8704 | 12 | 201.1 | 25.00 | 124.3 | 25.8% | 0.28 |
| 6144 | 1 | 144.7 | 18.32 | 126.6 | 26.5% | 0.31 |
| 5120 | 6 | 121.5 | 15.08 | 124.1 | 26.1% | 0.31 |
| 3072 | 6 | 76.1 | 9.49 | 124.7 | 28.1% | 0.36 |
| 512 | 2 | 18.1 | 2.02 | 111.6 | 38.1% | 0.91 |

Three findings, none of which nsys timing alone could produce:

**1. No individual GEMV comes close to the bandwidth ceiling.** Every shape sits at
**96-127 GB/s, i.e. 32-42% of the measured 300 GB/s**. The global 84.3% figure is an
emergent property of many asymmetric kernels overlapping: while one kernel stalls on DRAM,
others issue. This means the per-kernel roofline is not the right model and "get the GEMV to
300 GB/s" was never a well-posed target.

**2. There are two distinct regimes, and one of them is latency-bound.** At `grid=2560` the
kernel hits 96 GB/s with a 61% L2 hit rate and a `long_scoreboard` stall ratio of **2.81**
(nearly 10x the 0.28-0.36 of every other shape). `long_scoreboard` is a wait on a global
memory dependency, so this kernel is waiting on DRAM latency rather than saturating
bandwidth. The other shapes sit at 26% L2 and ~0.3, which is the healthy bandwidth-limited
profile. A launch-configuration change cannot fix a latency-bound kernel; it needs either
more bytes in flight per thread (wider loads, more independent loads in flight) or the
working set to fit better.

**3. `grid=2560` is a split of a larger matrix, and it is 40% of the GEMV time.** The model
has no 2560-row weight: `ffn_down` is 5120 rows, `attn_qkv` is 10240, `ffn_gate/up` is 17408,
`attn_gate`/`ssm_out`/attention projections are 6144. A grid of exactly half suggests the
matmul is being split, and the split half is the slowest kernel per launch after 8704 while
carrying a pathological L2 and stall profile. 2560-row launches accumulate **2502 us of the
6259 us total (40%)**.

Note also `launch__registers_per_thread = 89` for every shape, which caps occupancy; worth
noting but not yet shown to be the cause.

## What this changes

The earlier framing was "the GEMV is at 84.3% of the roofline, so at most 15.7% is
recoverable". That is now known to be the wrong question:

- The *aggregate* 84.3% is not a per-kernel efficiency that can be raised; it is what
  overlapping kernels produce.
- The actionable target is the **latency-bound 2560-row launches**, 40% of GEMV time, where
  the kernel is idle waiting on memory rather than short of bandwidth.
- Everything else is at 32-42% of peak per kernel and is bandwidth-sharing behaviour, not a
  fixable defect.

## Honest status

This is a diagnosis, not yet a speedup. No kernel change has been made on the basis of it.
The remaining decode-side work is genuinely kernel engineering now: increase memory-level
parallelism in the latency-bound shape (wider per-thread loads, several independent loads in
flight, or restructuring to avoid the split), which needs in-model A/B as ADR 0004 requires,
because the standalone microbench predicted +20% and delivered +2% last time.

Given that decode is already 4.9-6.0x faster under speculation (ADR 0003), the expected
value here is modest: at most ~13% of the non-speculative GEMV time, which is ~5% of a
speculated token.
