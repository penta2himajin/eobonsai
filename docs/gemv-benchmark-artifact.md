# The ncu investigation was chasing a benchmark artifact

This document closes the GEMV line of work with a correction. It is the fourth prediction
in this project that measurement overturned, and the most expensive one to have missed,
so the reasoning failure is recorded in full.

## What happened

`docs/gemv-ncu-analysis.md` and `docs/gemv-split-rootcause.md` analysed a GEMV shape with
`grid = 2560`, 61% L2 hit rate, `long_scoreboard = 2.81`, and ~40% of GEMV time. The second
document correctly root-caused the *mechanism*: `ncols_dst = 4` forces
`rows_per_block = 2`, halving the grid.

What neither document checked is whether that shape exists in the workload anyone runs.
It does not.

## The measurement that settles it

Same ncu configuration, same model, two drivers:

| | `llama-bench -p 0 -n 64` | `llama-cli -n 6` |
|---|---|---|
| `ncols_dst` observed | **4** | **1** |
| grids seen | 512, 2560, 3072, 5120, 6144, 8704 | 1024, 5120, 6144, 10240, 12288, 17408 |
| grids vs real weight rows | each is exactly half | **each is exactly equal** |
| effective bandwidth | **116.5 GB/s** | **228.7 GB/s** |
| L2 hit rate | 26-58% | 3.6-5.4% |
| `long_scoreboard` | 0.00 | 1.6-3.8 |

`llama-bench` generates tokens in batches of 4, so `src1->ne[1] = 4`, so `ncols_dst = 4`.
`llama-cli` generating one token at a time gives `ncols_dst = 1`, and the grid becomes the
weight's true row count (matching the GGUF enumeration in `gemv-split-rootcause.md` exactly:
1024 / 5120 / 6144 / 10240 / 12288 / 17408).

Two conclusions, both uncomfortable:

**1. The "84.3% of the bandwidth roofline" figure describes the benchmark shape, not the
served shape.** In the real one-token-at-a-time path the GEMV reaches **228.7 GB/s = 76%**
of the 300 GB/s ceiling, and its L2 hit rate is 3.6-5.4%, meaning it streams almost
entirely from DRAM. That is a healthy, bandwidth-bound kernel. There was no latency-bound
pathology to fix, and the "40% of GEMV time in a slow shape" finding was measuring
`llama-bench`'s batching, not the model.

**2. RETRACTED: `llama-bench` does not understate served decode throughput.** The benchmark's 4-column batches
cost 116.5 GB/s against 228.7 GB/s for the streaming path: the benchmark is roughly **2x
less bandwidth-efficient** than the workload it is used to represent. So `llama-bench`'s
`tg128` is not a faithful proxy for single-user streaming, which is the target workload of
this entire project.

End to end the claim is false: the `llama-bench` baseline is 29.55 t/s and the served-shape
CLI baseline is 28.8 t/s, so the benchmark is slightly **higher**, not a lower bound.
Per-kernel bandwidth cannot establish end-to-end ordering because the two shapes also differ
in non-GEMV work. The shape diagnosis above stands; this corollary is withdrawn
(`docs/review-findings-sol.md`).

This second point also means **the reported baseline and the ADR 0004 nwarps gain are
measured in the wrong shape.** They are internally consistent (both patched and unpatched
builds were measured the same way, so the +2.0% comparison stands), but their absolute
values should not be read as served throughput. A separate measured fact supports this:
`llama-cli` reported 27.7 t/s generation on the same model where `llama-bench` reported
29.55 t/s, and the server's own counters reported 29.1 t/s.

## Why the mistake was easy to make, and what would have prevented it

ncu profiles whatever runs. `llama-bench` was chosen as the harness early because it
isolates prefill from decode, records machine state, and is reproducible - all good
properties, and none of them is "drives the kernel shapes the server uses".

The missing step was trivial: **check which kernel instantiations the profiled run actually
produced, and compare the grid sizes against the model's real tensor shapes.** The grid-to-
weight mapping was available from the GGUF all along. The ncu analysis even printed the
grids; nobody asked whether 2560 is a shape the model has. It is not.

The general rule this project keeps rediscovering, now in a third form: *verify that the
thing being measured is the thing being optimised.* ADR 0004 was "the microbench is not the
kernel"; ADR 0005 was "the trace is not the mechanism"; this is "the benchmark is not the
workload".

## What stands, and what does not

| Claim | Status |
|---|---|
| `ncols_dst = 4` halves the grid | **Stands** (`calc_rows_per_block`, `mmvq.cu:543`) |
| No 2560-row weight exists in the model | **Stands** (GGUF enumeration) |
| `ggml_cuda_op_mul_mat_vec_q` is dead code | **Stands** (whole-tree search) |
| GEMV is latency-bound at 96 GB/s and needs fixing | **Retracted.** That is the benchmark shape. In the served shape it is 228.7 GB/s and healthy |
| The 84.3% aggregate roofline figure | **Reframed.** Valid for `llama-bench` tg128, not for served decode |
| `nwarps = 2` for PQ2_0 decode (+2.0%) | **Stands** as a like-for-like A/B within the benchmark shape; whether it helps the served shape is **unmeasured** |

## Next actions this forces

1. **Re-measure the served shape.** `scripts/bench.sh` should gain a mode that measures
   single-token streaming (`llama-cli` or the server), so future numbers describe the
   workload. Until then, treat `llama-bench tg` as a lower bound.
2. **Re-validate ADR 0004 in the served shape.** The `nwarps = 2` patch was tuned and
   verified under `ncols_dst = 4`. Under `ncols_dst = 1` the kernel takes a different code
   path, and `calc_nwarps` returns 2 for `ncols_dst == 1` by default in the *generic* table
   only because of the patch - the unpatched generic table returns 4 for all of
   `ncols_dst` 1-4. So the patch does apply to the served path too, but its effect there has
   not been measured.
3. **Do not resume kernel work on the "latency-bound shape".** It does not exist in
   production. The remaining honest target in the served shape is the 24% gap between
   228.7 GB/s and 300 GB/s, which is ordinary bandwidth efficiency work.

## The nwarps patch measured in the served shape

Item 2 above was run. Same prompt, `-n 200`, `--spec-type none`, `llama-cli`, three
repetitions each, patch applied then reverted with a rebuild between:

| shape | unpatched | patched |
|---|---|---|
| served (`ncols_dst = 1`, `llama-cli`) | **29.0, 29.0, 29.1 t/s** | 28.8, 28.9, 28.9 t/s |
| benchmark (`ncols_dst = 4`, `llama-bench tg128`) | 29.59 t/s | 30.19 t/s |

### Correction: the patch is neutral in the served shape, not a regression

The first pass of this table showed 29.0-29.1 t/s unpatched against 28.8-28.9 patched and
was written up as a -0.6% regression. Re-measured under identical conditions, that was
measurement noise, not an effect:

| build | served-shape repetitions |
|---|---|
| unpatched source build | 28.0, 28.8, 28.8 t/s |
| pinned prebuilt (never patched) | 28.7, 28.8 t/s |
| patched source build | 28.8, 28.9, 28.9 t/s |

All of it sits in 28.0-28.9, and the pinned prebuilt - which never had the patch - measures
the same 28.8. A separate check confirms clocks are not the confounder: the GPU ramps to
1957-1980 MHz during generation (max is 2160) and drops when idle.

**Conclusion: the patch is neutral in the served shape** (-0.6% was noise) while being a
+2.0% gain in the benchmark shape. It has therefore been removed: carrying a kernel change
that buys nothing for the target workload, and whose acceptance test was run in a shape the
product does not use, is not justified. The removal is verified - the unpatched build
measures 28.8 t/s, identical to the patched build.

The broader point from ADR 0004 still holds and this episode sharpens it: the A/B that
justified the patch ran in `llama-bench`'s shape, not the served shape. A +2.0% result in
the wrong shape was worth less than a 0.0% result in the right one.

Measured served-shape baseline for future work: **28.8 t/s** (`llama-cli`, `-n 200`,
`--spec-type none`, three repetitions, `results/bench-bin-PQ2_0-kf16vf16-20260927-234156.log`).
