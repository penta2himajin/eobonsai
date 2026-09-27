# ADR 0001: Which Bonsai 2 27B GGUF packing to run on the RTX 3060

- Status: **accepted** (measured on this card, 2026-09-27)
- Date: 2026-09-27
- Context: RTX 3060 12 GB (GA106, sm_86), single user, interactive chat and agent use

## Context

Bonsai 2 27B ships two GGUF packings whose weights are identical; only the packing
differs. All three packings need the PrismML llama.cpp fork: stock llama.cpp rejects
`PQ2_0`/`PTQ1_0` as unknown types, and loads the development `Q2_0` file without a
warning and produces garbage.

| Pack | True bpw | Size | Packing |
|---|---:|---:|---|
| `PTQ1_0` | 1.75 | 5.95 GB | dense trits, 5 elements per byte |
| `PQ2_0` | 2.13 | 7.21 GB | one trit per 2-bit slot |

The model card says `PTQ1_0` is the faster decode on Ada-class cards and the L4, and
that `PQ2_0` is faster at prompt processing everywhere. That guidance does not cover
Ampere consumer parts, which is what this machine has.

## Evidence

1. **Community measurement, RTX 3090 Ti (GA102, sm_86, Ampere), CUDA/Linux**
   (`Bonsai-demo/community-benchmarks/bonsai2/cuda-rtx3090ti-linux.md`):

   | Pack | pp512 | tg128 |
   |---|---:|---:|
   | `PQ2_0` | 1552 | 81.6 |
   | `PTQ1_0` | 805 | 67.7 |

2. **Our measured roofline** (`docs/roofline-rtx3060.md`): prefill for this model needs
   13.75e12 int8 MACs per 512 tokens and the card sustains 13.0e12 MAC/s of dp4a, so the
   PQ2_0 prefill ceiling is 483 t/s. The RTX 3070's measured pp512 of 838.9 t/s scales to
   510 t/s at 28 SMs. Two independent routes agree, so `PQ2_0` prefill sits at the roofline.

3. Both packings execute the same number of int8 MACs. Therefore the 1.93x prefill gap
   in row 1 cannot come from arithmetic throughput; `PTQ1_0` must be spending issue slots
   on unpacking dense trits, and roughly halving effective throughput.

## Measured on this card

`scripts/bench.sh bin/cuda <pack>` with `REPS=3`, load average 1.9-2.9
(`results/bench-cuda-PQ2_0-kvf16-20260927-164528.log`,
`results/bench-cuda-PTQ1_0-kvf16-20260927-165055.log`):

| Pack | pp512 | tg128 | tg @ d8192 | tg @ d32768 | Effective decode bandwidth |
|---|---:|---:|---:|---:|---:|
| `PQ2_0` | 493.8 ± 2.5 | **29.55** ± 0.13 | **28.04** | **24.09** | 213 GB/s = 71% of the 300 GB/s ceiling |
| `PTQ1_0` | **515.8** ± 11.7 | 22.51 ± 0.04 | 21.61 | 19.32 | 134 GB/s = **45%** of the ceiling |

**Evidence 3 above did not generalise, and is corrected here.** On this card `PTQ1_0`
prefill is *faster* than `PQ2_0` (515.8 vs 493.8), not ~2x slower; both packs sit at the
dp4a ceiling, and the 3090 Ti's 2x prefill gap does not reproduce on 28 SMs. What does
reproduce, and much harder, is the decode penalty: `PTQ1_0` streams 17% fewer bytes per
token but takes 31% longer, because its dense-trit unpacking leaves it at 45% of the
achievable bandwidth against `PQ2_0`'s 71%. At 5.95 GB/44.4 ms the card is not
bandwidth-starved at all in that configuration; it is issue-bound.

## Decision

Use **`PQ2_0`** as the primary packing. The deciding factor is decode latency, which is
what a single user experiences, and there `PQ2_0` is 31% faster. Keep `PTQ1_0` only as a
fallback if VRAM ever becomes the binding constraint, accepting that penalty.

## Consequences

- Decode at depth 0: 29.55 t/s (`PQ2_0`) vs 22.51 (`PTQ1_0`). Prefill is a wash.
- `PTQ1_0` saves 1.26 GB, which buys about 20K tokens of FP16 KV context or about 73K
  tokens of 4-bit KV context. With `PQ2_0` plus a 4-bit cache already reaching roughly
  220K tokens (ADR 0002), that extra context is not worth a 31% slower token.
- `PTQ1_0` is not a workaround for prefill on this card. Nothing in the packing choice
  moves prefill away from the dp4a ceiling.
- The `mmproj` vision projector (0.63 GB) should stay off GPU (`--no-mmproj-offload`) when
  context is the priority.
