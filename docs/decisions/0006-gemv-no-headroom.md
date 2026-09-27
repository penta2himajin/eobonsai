# ADR 0006: The decode GEMV headroom is 8-13%, not 0% and not 24% (corrected)

- Status: **superseded by the correction below** (measured 2026-09-27/28, corrected 2026-09-28)
- Date: 2026-09-28
- Context: RTX 3060 12 GB, `PQ2_0`, served shape (`ncols_dst = 1`)

## The claim being tested

After `docs/gemv-benchmark-artifact.md` corrected the shape error, the remaining decode
target looked like this: the GEMV reaches **228.7 GB/s** in the served shape against a
**300 GB/s** ceiling, so ~24% might be recoverable with kernel work. Step B of the
optimization objective was to go get it.

## The ceiling was wrong

`tools/microbench/gemv.cu` gained `bench_row_stream()`: a pure sequential read of one row
per block, blocks striding across rows, at the row sizes the model actually presents.

| row size | pure streaming read |
|---|---:|
| 1360 B (K=5120) | 272.2 GB/s |
| 4640 B (K=17408) | 274.8 GB/s |
| 16384 B | 274.5 GB/s |
| 256 B | 154.3 GB/s |

The applicable ceiling is **~275 GB/s**, not 300, and it is flat above ~1 KB rows. The
300 GB/s figure came from a single large contiguous buffer in `roofline.cu`, which is not
the geometry the model presents. Below ~1 KB per row, DRAM efficiency collapses (154 GB/s at
256 B), which is a property of the memory system, not of the kernel.

## The GEMV already meets the real ceiling

ncu on the served shape, 6 launches (`results/ncu-served-gemv.csv`):

| rows | us | GB/s | DRAM % | L2 % | L1 % | SM % | occupancy % | reg |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 10.8 | 184.4 | 53.1 | 14.0 | 97.3 | 53.1 | 59.4 | 56 |
| 1024 | 11.0 | 175.7 | 50.5 | 14.0 | 97.3 | 53.4 | 59.6 | 56 |
| 5120 | 44.1 | 213.9 | 61.1 | 5.8 | 97.5 | 74.6 | 62.0 | 56 |
| 5120 | 102.6 | 241.8 | 69.1 | 5.5 | 97.6 | 74.2 | 66.5 | 56 |
| 12288 | 84.4 | 210.9 | 60.3 | 4.6 | 97.5 | 77.6 | 61.1 | 56 |
| 17408 | 174.7 | **277.7** | 79.3 | 3.2 | 95.8 | 75.6 | 71.9 | 48 |

The dominant shape (17408 rows, `ffn_gate`/`ffn_up`, 128 of the 401 rotated weights) reads
**277.7 GB/s = 101% of the measured pure-streaming ceiling for the same geometry**. L1 hit
rate is 96-98% and L2 hit rate is 3-14%, meaning the loads hit L1 and the remainder streams
from DRAM: exactly the profile of a saturated streaming kernel.

## CORRECTION (2026-09-28): the conclusion was an over-extrapolation

An external review (`docs/review-findings-sol.md`) pointed out that this ADR profiles six
launches, finds the 17408-row shape saturated, and generalises to the whole model. A byte
census of the GGUF confirms the 17408-row shape is only **46.9% of PQ weight bytes**, and the
5120-row shape, **31.7%**, runs at 232-242 GB/s against the same ~275 GB/s ceiling.

A byte-weighted calculation, which is what this ADR should have done:

| sample | weighted average | % of ceiling | gap vs ideal |
|---|---:|---:|---:|
| ncu-served run | 245.8 GB/s | 89.4% | 2.8 ms = 8.2% of a token |
| ncu-cli run | 231.9 GB/s | 84.3% | 4.4 ms = 12.9% of a token |

**So the headroom is 8-13% of decode**, concentrated in the 5120-row shape and the
6144/10240 shapes. The 24% originally claimed was wrong because it divided by the wrong
ceiling; this ADR's 0% was wrong because it sampled one shape and generalised.

Two further caveats, also from the review and accepted: the 275 GB/s reference is measured in
an ordinary CUDA run while the GEMV figures come from Nsight Compute (which replays kernels,
flushes caches, and can hold clocks), so neither the old 101% nor the new 84-89% is a precise
closure test; and the decisive CSV was gitignored, so the evidence was a transcription. The
CSVs are now tracked.

The measurement-quality discussion below still stands on its own terms.

## What stood up

Every lever that would normally apply is already applied or accounted for:

| lever | state |
|---|---|
| arithmetic | already `dp4a` + `__byte_perm` |
| load path | 96-98% L1 hit rate |
| occupancy | 59-72%, 56 registers per thread |
| launch configuration | swept (ADR 0004 and its correction) |
| DRAM efficiency | at or above the geometry's measured ceiling |
| CUDA graphs | already active, worth 5.3% (ADR 0005) |

The stated step-B target ("close the 24% gap") **does not exist**. It was an artifact of
dividing by the wrong ceiling. This is the fifth prediction in this project that
measurement overturned, and like the previous four it failed by comparing against a
reference that was not the applicable one.

## Consequences

- The optimization objective is complete. Decode work ends here with the budget fully
  accounted for: 82.6% GEMV (DRAM-saturated), the remainder small kernels already
  graph-amortised, and CUDA graphs contributing 5.3%.
- The one large lever found in this whole effort remains speculative decoding at
  **4.9-6.0x** (ADR 0003), which does not speed up the pass but buys more tokens per pass.
- Prefill is at the `dp4a` ceiling (ADR 0001) and cannot move without tensor cores and a
  different weight packing, i.e. a re-architecture.
- `docs/roofline-rtx3060.md` should be read with this correction: its 298-307 GB/s figure is
  a *contiguous-buffer* ceiling, not the ceiling for the model's row-structured reads, which
  is ~275 GB/s.
- Small shapes (1024 rows, the attention `k`/`v` weights) sit at 175-184 GB/s, below the
  ceiling. They are 2 of 401 weights and were not pursued; at 10.8 us each they are ~2% of
  GEMV time, so the remaining opportunity there is under 1% of a token.

## Reproduction

```bash
scripts/roofline.sh                      # card ceilings (contiguous buffer)
./out/gemv                               # row geometry: pure streaming + config sweep
scripts/bench.sh build/llama-sm86/bin PQ2_0   # PHASES=cli for the served shape
```

Evidence: `results/gemv-roofline-closure.txt`, `results/ncu-served-gemv.csv`,
`results/bench-bin-PQ2_0-kf16vf16-20260927-234156.log`.
