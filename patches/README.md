# Local patch overlay

Kernel changes are kept here as small, rebasable diffs against the pinned fork commit
rather than as edits to the checkout. `third_party/llama.cpp` is disposable: a fresh
clone plus `scripts/build.sh` reproduces the tuned build exactly, because `build.sh`
applies everything in this directory after checking out the pinned commit.

Each patch must name the commit it applies to and carry the measurement that justifies it.
A patch is only accepted when:

1. `scripts/parity-check.sh` still passes (perplexity and greedy tokens against the pinned
   prebuilt binary), and
2. an A/B on this card, with the same toolchain, shows a gain, with the result file
   committed under `results/`.

## 0001-mmvq-nwarps-pq2_0-decode.patch

Applies to `adfffbe41b2cabcd51fff326ab045662265062bb`.

Uses 2 warps instead of 4 per output row for the ternary GEMV at decode on the generic
(NVIDIA non-Turing, non-GB10) device table, which is what sm_86 gets.

Measured in the `llama-bench` shape (`ncols_dst = 4`): `tg128` 29.59 -> **30.19 t/s**
(+2.0%), `tg128 @ d8192` 28.01 -> **28.49 t/s** (+1.7%), unpatched and patched source
builds, `REPS=3`. Parity clean.

**But it is a regression in the served shape.** `llama-cli` streaming
(`ncols_dst = 1`) measures 29.0-29.1 t/s unpatched against 28.8-28.9 t/s patched, about
-0.6%. The two shapes take different paths through `calc_nwarps`/`calc_rows_per_block`.
The target workload is streaming, so this patch is **pending removal**; see
`docs/gemv-benchmark-artifact.md`. It stays applied only until the removal is verified.

Background: a standalone microbenchmark had predicted +20% for this change and was wrong
by 10x, because its simplified K-loop was itself slower than the real kernel's at the same
configuration. See [ADR 0004](../docs/decisions/0004-gemv-launch-config.md); the lesson is
that launch-configuration sweeps in simplified kernels do not transfer.
