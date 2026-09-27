# RTX 3060 roofline (measured)

All numbers below were measured on this machine, not taken from spec sheets or
community tables. They are the denominators used to judge whether a model-level
result is good or bad, and to decide where optimization effort can pay off.

- GPU: NVIDIA GeForce RTX 3060 12 GB (GA106, sm_86, 28 SMs, 1837 MHz sustained)
- Driver 595.91.07, CUDA 12.4, `third_party/llama.cpp` at `adfffbe`
- Harness: `tools/microbench/roofline.cu`, run via `scripts/roofline.sh`
- Raw output: `results/roofline-*.txt`

> **Correction (ADR 0006):** the 298-307 GB/s figure below is measured on one large
> contiguous buffer. `tools/microbench/gemv.cu` measures the same card reading
> model-shaped rows (1360 B and 4640 B per row) and gets **272-275 GB/s**, flat above ~1 KB
> rows and collapsing to 154 GB/s at 256 B. Use ~275 GB/s when judging the GEMV; the
> contiguous figure overstates the reachable ceiling by about 10% and was the source of a
> nonexistent "24% headroom" target.

## Measured ceilings

| Quantity | Measured | Reference | Note |
|---|---:|---|---|
| Streaming read bandwidth | **298 - 307 GB/s** | 360 GB/s (192-bit GDDR6 spec) | 83-85% of spec. **This is a contiguous-buffer ceiling**; for the model's row-structured reads the applicable ceiling is ~275 GB/s (ADR 0006) |
| Empty kernel launch | **1.6 - 1.8 us** | - | rises to 2.9 us under CPU load (parallel build) |
| Null-work launch (28x256) | **1.6 us** | - | launch cost is grid-size independent at this size |
| dp4a (int8 4-way dot) | **26.0 TOPS(int8)** = 13.0e12 MAC/s | - | 8 independent chains |
| FP32 FMA | **11.7 TFLOPS** = 5.84e12 FMA/s | 12.8 TFLOPS (28 SM x 128 lane x 2 x 1.837 GHz) | 91% of issue-limited peak |
| FP16x2 FMA (`__hfma2`) | **13.1 TFLOPS** | - | only 1.12x FP32 |
| FWHT N=1024, decode (5 blocks) | **2.07 us/call** | - | the model's rotation block size |
| FWHT N=1024, decode (17 blocks) | **2.09 us/call** | - | launch-latency bound, not bandwidth |
| FWHT N=1024, pp512 (2560 blocks) | **66 us** (318 GB/s) | 298-307 GB/s | at the bandwidth ceiling |
| FWHT N=1024, pp512 (8704 blocks) | **240 us** (297 GB/s) | 298-307 GB/s | at the bandwidth ceiling |

### The dp4a result that matters

dp4a delivers **13.0e12 MAC/s against FP32's 5.84e12 FMA/s, i.e. 2.2x, not 4x.**
GA106 has 128 FP32 lanes but only 64 INT32 lanes per SM, so the int8 advantage on
this card is roughly half of what a datacenter part gives. Any plan that assumes a
4x int8 multiplier over-counts the prefill ceiling by about 2x.

## Derived floors for Bonsai 2 27B

Model facts taken from the GGUF header (`tools/gguf-inspect.py`): 64 blocks,
embedding 5120, FFN 17408, 24 heads / 4 KV heads, key = value = 256,
`full_attention_interval = 4` (16 full-attention + 48 linear-attention layers),
Hadamard block 1024, 401 rotated weight tensors.

### Decode (bandwidth-bound)

Weights are streamed once per token.

| Pack | Weight bytes | Floor at 300 GB/s | Ceiling | Practical (82-85% eff.) |
|---|---:|---:|---:|---:|
| PTQ1_0 | 5.95 GB | 19.8 ms | 50 t/s | 40 - 43 t/s |
| PQ2_0 | 7.21 GB | 24.0 ms | 42 t/s | 34 - 36 t/s |

#### Confirmed by measurement

`scripts/bench.sh bin/cuda PQ2_0` with `PHASES=tg`
(`results/bench-cuda-PQ2_0-kvf16-20260927-164528.log`):

| Test | Measured t/s | ms/token | Efficiency vs the 24.0 ms floor |
|---|---:|---:|---:|
| tg128 | 29.55 ± 0.13 | 33.8 | 71% |
| tg128 @ d8192 | 28.04 ± 0.07 | 35.7 | 67% |
| tg128 @ d32768 | 24.09 ± 0.07 | 41.5 | 58% |

**The KV traffic model is confirmed.** The depth deltas against the 64 KiB/token figure:

| Depth | Predicted KV read | Predicted added time | Measured delta | Error |
|---:|---:|---:|---:|---:|
| 8192 | 512 MiB | +1.79 ms | +1.9 ms | +6% |
| 32768 | 2.0 GiB | +7.16 ms | +7.7 ms | +8% |

At 32K depth, KV is **19% of the decode token**; the ADR 0002 prediction that a 4-bit
cache becomes worthwhile above 8-32K depth rests on a model that has now been checked
against this card.

#### The remaining decode gap

At depth 0, decode takes 33.8 ms against a 24.0 ms streaming floor, so **9.8 ms is
neither weight streaming nor KV**. The RTX 3070 (same architecture, 448 GB/s) measured
22.9 ms for the same model, which scales to 29 ms at this card's bandwidth: the 3060 is
about 5 ms worse than bandwidth scaling predicts, consistent with its 28 SMs and 3 MB L2
hurting the non-matmul parts of the step.

That 9.8 ms is the real optimization target, and the candidate list is in the section
below. `scripts/profile-ncu.sh` exists to identify it per kernel rather than by
elimination.

### Prefill (dp4a-bound)

PP512 needs 2 x 26.9e9 x 512 = 27.5 TFLOP = 13.75e12 int8 MACs.

> **Correction (2026-09-28):** this charges all 26.9B parameters to every prompt token. The
> embedding (1.271B) is a row lookup, not a per-token matmul, and the output head (1.271B) is
> computed only for the last token of an ordinary batch. Excluding 2.543B (9.5%) gives
> 2 x 24.36e9 x 512 = 24.9 TFLOP and a bound of roughly **534 t/s**, so the measured
> 493.8 t/s is about **92%** of the bound rather than 102% of a hard ceiling. Attention, the
> norms and imperfect dp4a utilisation consume the remainder, so this does not establish a
> recoverable 8%; it invalidates the "measured above the ceiling, therefore closed" reasoning.
> See `docs/review-findings-sol.md`.

```
13.75e12 MAC / 13.0e12 MAC/s = 1.06 s  ->  483 t/s ceiling
```

Cross-check: the RTX 3070 (46 SMs, same sm_86) measured PP512 = 838.9 t/s in the
community table. Scaled to 28 SMs that is 510 t/s. Two independent routes agree, so
**PQ2_0 prefill is already at the dp4a roofline** and a tile-tuning effort can only
recover the residual 5-15%, not a multiple.

#### Confirmed by measurement

`scripts/bench.sh bin/cuda PQ2_0` with `PHASES=pp`
(`results/bench-cuda-PQ2_0-kvf16-20260927-164343.log`), load average 4.7:

| Test | Measured t/s | Roofline prediction |
|---|---:|---:|
| pp128 | 460.1 ± 18.7 | - |
| pp512 | 493.8 ± 2.5 | 483 (dp4a) / 510 (3070-scaled) |
| pp2048 | 491.3 ± 0.5 | - |
| pp8192 | 474.9 ± 0.2 | - |

The measurement lands between the two predictions, within 2% of the mean. Throughput is
flat from 512 to 8192 tokens, so prefill is throughput-limited rather than
overhead-limited, exactly as the dp4a model says.

**Consequence: PQ2_0 prefill tuning is a dead end.** The only way to prefill materially
faster on this card is to stop doing int8 `dp4a`, for example by feeding the ternary
weights to tensor cores, which needs a different weight packing and is a
re-architecture rather than a tuning exercise.

PTQ1_0 is a different story: on the RTX 3090 Ti the community measured PP512 = 805 for
PTQ1_0 against 1552 for PQ2_0. Both packs perform the same number of int8 MACs, so
PTQ1_0's prefill cannot be dp4a-bound; its dense-trit unpacking is instruction-issue
bound and roughly doubles the cost. See ADR 0001.

### FWHT (the Hadamard rotation) is not a bottleneck

The rotation is dispatched from inside `mul_mat` when the node carries
`GGML_HINT_SRC0_IS_HADAMARD` (`ggml-cuda.cu:1822`), and `hadamard_memo` is keyed on
`(activation, rot)` (`llama-graph.cpp:1556`), so weights that share an activation and
rotation share one transform. The pass count is on the order of 257 per forward step.

| Phase | Cost | Share |
|---|---:|---:|
| decode | 257 x ~2.1 us = **0.54 ms** | ~1.9% of a 29 ms token |
| pp512 | ~0.5 ms | ~0.08% of a 0.6-1.1 s prefill |

FWHT is at the bandwidth ceiling where it is bandwidth-bound, and launch-latency bound
at decode. It is not where the remaining time is.

## What this rules out

- **Rewriting the decode GEMV inner loop.** `vec_dot_pq2_0_q8_1` already uses dp4a plus
  `__byte_perm` unpacking; the decode floor is the 24 ms bandwidth wall, so the whole
  remaining gap is ~5 ms, and it is not in that loop.
- **Expecting a big prefill win from PQ2_0 tile tuning.** 483 t/s ceiling vs a 510 t/s
  cross-check; the headroom is single-digit percent.
- **Fusing the FWHT into the GEMM.** It would remove at most ~0.5 ms of launch latency
  per decode step.

## Where the remaining decode time must be (to be confirmed with ncu)

24 ms of weight streaming against a ~29 ms measured token leaves roughly 5 ms. Candidates,
in rough order of expected size:

1. **KV cache traffic at depth.** 16 full-attention layers x 4 KV heads x 256 x 2 x 2 B =
   64 KiB/token, which matches the model card exactly. At 32K depth that is 2 GiB read per
   token (+5.8 ms, +29%); at 64K it is 4 GiB (+11.6 ms, +58%). A 4-bit KV cache
   (~18 KiB/token) cuts most of that. This is a configuration change, not a kernel.
2. **Launch count.** At 1.6 us per launch, a decode step with 1000-2000 launches spends
   1.6-3.2 ms. Small ops (norms, rope, the 257 FWHT passes) each do less work than their
   own launch cost, so the GPU starves there.
3. **The linear-attention path.** 48 of 64 layers are gated delta net (`ssm-scan.cu`,
   conv1d + a sequential scan), which is not a dp4a matmul and is therefore not covered by
   the roofline above.

The per-kernel time table from `ncu` over one decode step will settle the split.
