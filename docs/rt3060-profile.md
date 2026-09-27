# Confirmed profile: Bonsai 2 27B on the RTX 3060 12 GB

This is the configuration to run, with the measurement behind each choice. Every
number is from a file under `results/`; predictions that measurement overturned are
recorded in `docs/decisions/`.

## The profile

```bash
scripts/serve.sh                 # config/rt3060.env
```

| Setting | Value | Why |
|---|---|---|
| pack | `PQ2_0` (7.19 GiB) | `PTQ1_0` decodes 31% slower (ADR 0001) |
| `-ngl` | 99 | everything offloaded; weights are 6,872 MiB |
| `-fa` | on | required by the model's attention path |
| `-c` | 32768 (64K verified) | 64K fits at 11,732 MiB; ~74K is the f16 ceiling |
| KV type | f16 | quantized KV is **slower** at every depth (ADR 0002) |
| `-b` / `-ub` | 2048 / 512 | swept 512-4096: pp512 spans 487-494, all within noise |
| `-np` | 1 | `-np 2` splits the pool to 16,384 tokens per slot and costs VRAM |
| `--reasoning-budget` | 2048 | thinking dominates latency at 29 tok/s |
| `reasoning_effort` | medium | documented as about as accurate at moderate limits |
| mmproj | off GPU | 0.63 GB, only needed for image input |

## Measured performance

Baseline, load average 1.9-2.9, `REPS=3`.

| Phase | Result | Roofline | Files |
|---|---:|---|---|
| pp512 | **493.8 ± 2.5 t/s** | 483 (dp4a) / 510 (3070-scaled) | `results/bench-cuda-PQ2_0-kvf16-20260927-164343.log` |
| pp2048 | 491.3 t/s | - | same |
| pp8192 | 474.9 t/s | - | same |
| tg128 | **29.55 ± 0.13 t/s** | 42 t/s ceiling, 24.0 ms floor | `results/bench-cuda-PQ2_0-kvf16-20260927-164528.log` |
| tg128 @ d8192 | 28.04 t/s | - | same |
| tg128 @ d32768 | **24.09 t/s** | - | same |

Independent confirmation of decode through a different path: `scripts/telemetry.sh`
during a live 300-token request computed **29.1 tok/s** from the server's own counters
and showed the load transient (GPU 97% -> 12%, 175 W -> 46 W, 67 C -> 57 C)
(`results/telemetry-20260927-171527.csv`).

### Prefill is at the hardware ceiling

The card sustains 13.0e12 int8 MAC/s of `dp4a` (measured, `tools/microbench/roofline.cu`)
and PP512 needs 13.75e12 MACs, so the ceiling is 483 t/s. The measurement lands 2% above
that, between it and the 510 t/s obtained by scaling the RTX 3070's community result.
Throughput is flat from 512 to 8192 tokens, so this is throughput-limited, not
overhead-limited. **There is no prefill tuning left in the `PQ2_0` path.**

### Where decode time goes

Per token at depth 0: **33.8 ms measured against a 24.0 ms weight-streaming floor.**

| Component | Time | Source |
|---|---:|---|
| weight streaming (7.19 GiB at 300 GB/s) | 24.0 ms | measured bandwidth |
| FWHT (257 Hadamard passes) | 0.54 ms | measured, 2.1 us each |
| KV read at depth 32K | 7.7 ms | measured depth delta |
| **unaccounted at depth 0** | **~9.3 ms** | remainder |

The unaccounted part is the one open question. `scripts/profile-nsys.sh` and
`scripts/profile-ncu.sh` exist to identify it per kernel; neither has been run yet
(nsys is not installed, and ncu needs root for hardware counters on this machine).

## Measured 12 GB budget

llama.cpp sees 11,904 MiB with the display attached (12,288 MiB total, ~384 MiB for the
desktop).

| Component | Size |
|---|---:|
| weights (`PQ2_0`) | 6,872 MiB |
| compute buffers | 736 MiB at ctx 32768, 764 at 65536 |
| KV (f16) | 64 MiB per 1024 tokens |

Observed totals: **9,656 MiB at ctx 32768** and **11,732 MiB at ctx 65536**, both
verified to load and serve. That leaves about 556 MiB spare at 64K, putting the f16
context ceiling near **74K tokens**.

## Reproduction

```bash
scripts/setup.sh all                        # pinned binary + both packs, hash-checked
scripts/roofline.sh                         # card ceilings
scripts/bench.sh bin/cuda PQ2_0             # both phases, records machine state
scripts/parity-check.sh bin/cuda build/llama-sm86/bin PQ2_0   # build equivalence
scripts/serve.sh start && scripts/telemetry.sh
```

`scripts/parity-check.sh` passes: the sm_86 source build reproduces the pinned binary
exactly (perplexity 5.8953 vs 5.8953, identical greedy tokens) on
`results/parity-20260927-164848/`. Any future kernel change must keep that true.

## Explored and rejected

| Option | Result |
|---|---|
| `PTQ1_0` packing | prefill +4%, decode -24%, effective bandwidth 45% vs 71% |
| 4-bit KV (`q4_0`/`q4_0`) | slower at every depth; -19% at 32K |
| 4-bit K only (`q4_0`/`f16`) | worst of all: 17.36 t/s at depth 0 |
| `-ub` 1024/2048, `-b` 4096 | within 1.5% of default, i.e. noise |
| `-np` 2 | halves per-slot context to 16,384 and raises VRAM |
| tuning `PQ2_0` prefill | at the dp4a roofline already |

## What would still move the needle

1. **Identify the ~9.3 ms.** Only profiling will say whether it is launch overhead
   (~1.6 us x launch count), the gated-delta-net scan path in the 48 linear-attention
   layers, or the full-attention layers. This is the prerequisite for any kernel work.
2. **Tensor cores for prefill.** The only route past 483 t/s, and it needs a different
   weight packing plus mma kernels: a re-architecture, not a tuning pass.
3. **Speculative decoding.** Not a kernel change and not yet tried; the fork ships
   `llama-lookup` (prompt-lookup, no drafter model needed) which suits agentic and code
   workloads where output copies context.
4. **Beyond ~74K context.** Requires a quantized cache and therefore accepting the
   measured speed penalty plus calibrating the mean-centering bias.
