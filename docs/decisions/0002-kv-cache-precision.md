# ADR 0002: KV cache precision on a 12 GB card

- Status: **rejected for speed, retained as a capacity tool** (measured 2026-09-27)
- Date: 2026-09-27
- Context: RTX 3060 12 GB, `PQ2_0` weights (7.21 GB), single user, agent work with
  long inputs

## Context

The model supports 262,144 tokens of context, and only 16 of 64 layers are full
attention (`full_attention_interval = 4`), so the KV cache is small for a 27B model.
KV per token, from the GGUF header parameters:

```
16 full-attention layers x 4 KV heads x 256 key dim x 2 (K and V) x 2 bytes = 65,536 B
```

This matches the 64 KiB/token stated on the model card exactly.

## The prediction that was tested

Decode streams the KV cache once per token at the same bandwidth as the weights, so a
4-bit cache (~18 KiB/token) should pay for its dequantization cost above some depth.
The model, using the measured 300 GB/s ceiling and the 24.0 ms weight-streaming floor:

| Depth | FP16 KV read | Added to the floor | 4-bit KV read | Added |
|---:|---:|---:|---:|---:|
| 8K | 0.5 GiB | +1.8 ms | 0.14 GiB | +0.5 ms |
| 32K | 2.0 GiB | +5.8 ms | 0.56 GiB | +1.6 ms |
| 64K | 4.0 GiB | +11.6 ms | 1.1 GiB | +3.2 ms |

Predicted crossover: 4-bit should pull ahead between 8K and 32K and win by roughly 25%
by 64K.

## Measurement: the prediction is wrong

`scripts/bench.sh bin/cuda PQ2_0` with `PHASES=tg`, `REPS=3`
(`results/bench-cuda-PQ2_0-kvf16-20260927-164528.log`,
`results/bench-cuda-PQ2_0-kvq4_0-20260927-165521.log`):

| KV type | d0 | d8192 | d32768 | d65536 |
|---|---:|---:|---:|---:|
| F16 (baseline) | **29.55** | **28.04** | **24.09** | not measured (needs ~11.2 GB) |
| q4_0 / q4_0 | 29.28 | 25.82 | 19.40 | 14.46 |
| q4_0 / f16 (K only) | 17.36 | - | (run abandoned, far slower) | - |

In milliseconds per token the deltas over depth 0 are:

| KV type | +d8192 | +d32768 | +d65536 |
|---|---:|---:|---:|
| F16 | +1.8 ms | +7.7 ms | - |
| q4_0 / q4_0 | +4.6 ms | +17.4 ms | +35.0 ms |

The 4-bit cache adds **more** time at every depth, not less: at 32K its KV cost is
17.4 ms against F16's 7.7 ms, for 3.5x fewer bytes. The bandwidth model behind the
prediction is therefore not what governs this path. The KV cost is dominated by
dequantization work in the attention path, and it grows with depth faster than the
bytes it saves. Quantizing K alone is worse still (17.36 t/s at depth 0), which points
at a mixed K/V cache falling off a specialized kernel.

Note the depth-independent part is unaffected: at depth 0 all three configurations are
within 1% of each other, so the penalty is entirely in the depth term.

## Decision

- **Do not use a quantized KV cache on this card for speed.** FP16 wins at every depth
  measured, and is also the highest-quality option, so the default in
  `config/rt3060.env` stays `KV_TYPE=f16`.
- A 4-bit cache remains the **only** way to exceed roughly 61K tokens of context inside
  12 GB (about 220K tokens). If a workload needs that, accept the ~19% decode penalty at
  32K depth, and calibrate the mean-centering bias first: PrismML's `KV-CACHE.md`
  requires `scripts/make_kv_bias.sh` and states that plain `-ctk q4_0 -ctv q4_0` is not
  an option on quality grounds. That quality check was not run here.
- The 12 GB profile should therefore be explicit about its context ceiling: ~61K tokens
  with FP16 KV at full speed, or more context at reduced speed and unverified quality.

## Consequences

- The earlier claim in the project's own notes that "KV traffic at depth is the biggest
  single decode lever, and 4-bit KV is a speed feature" was wrong and is retracted. The
  depth term is real (19% of the token at 32K) but it cannot be bought back by
  quantizing the cache on this backend.
- This is the second prediction in this project that measurement reversed. Both are
  recorded rather than quietly fixed, because the pattern matters: on this card, models
  built from bandwidth alone mispredict anything that carries dequantization work.
