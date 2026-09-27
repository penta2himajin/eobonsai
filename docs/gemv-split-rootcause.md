# Decode GEMV: root cause of the split, and what it means for optimization

Follow-up to `docs/gemv-ncu-analysis.md`, which found that one GEMV shape profile
(`grid=2560`) was latency-bound, carried 61% L2 hit rate, and accounted for ~40% of GEMV
time. This document identifies why, and corrects a wrong guess made along the way.

## Wrong guess first, for the record

The initial reading was "grid=2560 is the *split half* of a 5120-row matrix", inferred from
the microbenchmark's own 2.16 GB of weight traffic divided by its 200,000 rows. That was
arithmetic on the wrong data: the microbench's `bytes/row` is a constant it chose, not a
property of the model.

A second guess blamed a row-split code path. `ggml_cuda_op_mul_mat_vec_q`
(`mmvq.cu:1479`) takes `row_low`/`row_high` and would do exactly this split, but a
whole-tree search shows it is **dead code**: defined and declared, never called. The live
path passes `ne01` (the full row count) at `mmvq.cu:1474`. So the split had to come from
elsewhere.

## Root cause

The instantiations ncu captured are all
`mul_mat_vec_q<(ggml_type)142, (int)4, ...>` - that is `type = PQ2_0`, **`ncols_dst = 4`**.

`ncols_dst` is the number of destination columns the kernel processes at once, and 4 is
`MMVQ_MAX_BATCH_SIZE`, the cap. With `ncols_dst = 4`, `calc_rows_per_block` returns **2**
(`mmvq.cu:543-549`), so the grid becomes `nrows_x / 2`:

| ncu grid observed | x2 | matching weight (from the GGUF) | count |
|---:|---:|---|---:|
| 512 | 1024 | `attn_k` / `attn_v` | 32 |
| 2560 | 5120 | `ffn_down` / `ssm_out` | 128 |
| 3072 | 6144 | `attn_gate` / attention projections | 48 |
| 5120 | 10240 | `attn_qkv` | 48 |
| 6144 | 12288 | `attn_q` (full-attention layers) | 16 |
| 8704 | 17408 | `ffn_gate` / `ffn_up` | 128 |

Enumerated directly from the GGUF, the ternary weights have row counts of
1024 / 5120 / 6144 / 10240 / 12288 / 17408 / 248320 and **no 2560-row weight exists**. Every
observed grid is exactly half a real weight's row count, and every one of the six matches a
real weight. The mapping is therefore exact, not circumstantial.

So decode is running the multi-column MMVQ path with the maximum batch of 4, even though a
single user generates one token at a time.

## Why this matters, and why it is not obviously a bug

This is not simply "a split is bad". There is a real trade:

- **Cost**: `ncols_dst = 4` uses 2 rows per block and a different reduction path, and the
  grid halves. The measured 2560-row shape is the worst offender: 96 GB/s against 124 GB/s
  for the same family, with a 61% L2 hit rate and `long_scoreboard` 2.81 against 0.28-0.36.
- **Benefit**: processing 4 columns per pass means the *weights are read once for four
  columns*. For a bandwidth-bound GEMV that is the whole point, and it is the same mechanism
  that makes speculative decoding pay off.

Which effect wins is an empirical question, and `llama-bench -p 0 -n 128` presumably already
has 4-token batches available to it (the benchmark generates tokens in a batch), so the
choice may be correct for the benchmark and wrong for single-token streaming. That
distinction has **not** been measured yet and is the next experiment, not a conclusion.

## What this does not change

The corrected picture from the ncu analysis stands: per-kernel bandwidth is 96-127 GB/s
against a 300 GB/s ceiling, i.e. every shape is far from its own roofline, and the aggregate
84.3% is emergent from overlapping kernels rather than a per-kernel efficiency. What changed
is the explanation of *why* one shape is worse: it is a configuration choice
(`ncols_dst = 4`), not a mysterious split.

## Next experiment, stated before running it

Force the single-column path for PQ2_0 decode and A/B it in-model, as ADR 0004 requires.
If `ncols_dst = 1` is faster for streaming, the change is a one-line configuration decision;
if it is slower, then the 4-column path is correctly chosen and the latency-bound shape is
simply the price of reading the weights once for four columns, which is a good trade.
Either result is informative and both are cheap to obtain.

The standalone-microbench trap applies in full here: the microbench measured a *different*
config sweep (`nwarps`) on a simplified loop and overstated the gain 10x. Only the in-model
A/B counts.
