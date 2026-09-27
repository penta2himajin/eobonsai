# eobonsai

An inference setup tuned for one specific pairing: **Bonsai 2 27B** (PrismML ternary
GGUF, from Qwen3.8-27B) on a single **RTX 3060 12 GB** (GA106, `sm_86`, Ampere), for
single-user interactive chat and agent work.

The premise is that this pairing needs its own tuning rather than generic advice:

- The model has **262K context** and a hybrid-attention backbone (16 full-attention,
  48 gated-delta-net layers), and fits in 5.95-7.21 GB. On a 12 GB card the interesting
  question is not "does it fit" but **how much context fits alongside it**, because KV
  cache traffic is most of the decode cost at depth.
- The two packings (`PTQ1_0` 5.95 GB, `PQ2_0` 7.21 GB) trade VRAM against throughput in
  a direction that community numbers show differs by GPU generation. See
  [ADR 0001](docs/decisions/0001-packing-choice.md).
- The card has no profiler-friendly counters and only 64 INT32 lanes per SM, so the
  int8 `dp4a` advantage over FP32 is ~2.2x, not the 4x a datacenter part gives. That
  changes which optimizations are worth attempting. Measured in
  [docs/roofline-rtx3060.md](docs/roofline-rtx3060.md).

Every decision in this repository is backed by a measurement on this machine. Numbers
taken from model cards, community tables, or vendor specs are marked as such.

## Requirements

- RTX 3060 12 GB, driver with CUDA 12.4 support, CUDA toolkit 12.4 at `/usr/local/cuda-12.4`
- ~20 GB of disk for weights and binaries
- The PrismML llama.cpp fork. **Stock llama.cpp cannot run this model**: it rejects
  `PQ2_0`/`PTQ1_0` as unknown types, and loads the development `Q2_0` file without a
  warning and produces garbage.

## Quick start

```bash
scripts/setup.sh all          # pinned CUDA binary + both GGUF packs (13 GB, resumable)
scripts/roofline.sh           # measured ceilings for this card (the yardsticks)
scripts/bench.sh bin/cuda PQ2_0
scripts/build.sh              # sm_86 source build, only needed for kernel work
```

## Layout

```
scripts/     setup, build, benchmark, profiling entry points
fixtures/    the prompts every recorded measurement was taken on
tools/       GGUF header inspector, CUDA microbenchmarks
docs/        roofline measurements, decisions (ADR)
results/     raw logs and CSVs from every run (tracked; the evidence)
third_party/ pinned llama.cpp fork checkout (not tracked)
bin/, models/, out/, build/   downloaded or generated (not tracked)
```

## Where this stands

**The headline:** decode is **4.9-6.0x faster** for work that reuses the context, via the
fork's draft-model-free n-gram speculation, once thinking is turned off for that request.
Agentic code editing measured 28.6 -> 140 t/s (CLI) and 28.5 -> 146 t/s (serving).
See [ADR 0003](docs/decisions/0003-ngram-speculation.md).

**Settled and verified end to end:** the pinned fork runs the ternary model correctly on
CUDA 12.4 and the sm_86 source build reproduces it exactly (perplexity and greedy tokens);
prefill is at the card's `dp4a` roofline; packing, KV type, batch shape and slot count are
decided by measurement; the serving layer works (llama-server, OpenAI-compatible endpoint,
start/stop, telemetry); and the decode budget is now decomposed per kernel. The
configuration to run is in [docs/rt3060-profile.md](docs/rt3060-profile.md).

**Open:** decode is 34.45 ms/token and the GPU is saturated (100.7% of wall clock), with
82.6% of it in the ternary GEMV at 84.3% of the bandwidth roofline. The remaining levers
are the GEMV's 15.7% gap and ~2.1 ms of non-bandwidth-saturated norm/quantise plumbing;
both are bounded and hard. See [docs/decode-profile.md](docs/decode-profile.md).

**Rejected by measurement:** `PTQ1_0` (decode -24%), quantized KV caches (slower at every
depth), and `-ub`/`-b`/`-np` tuning (within noise). Two predictions built from bandwidth
alone were overturned by measurement; both are recorded in `docs/decisions/`.

## License

MIT. See `LICENSE`. Note that the model and the llama.cpp fork carry their own licenses
(Apache 2.0 and MIT respectively); this repository tracks neither.
