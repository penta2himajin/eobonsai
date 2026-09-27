# eobonsai

**An experimental lab notebook**, not a product. It records one person's attempt to tune
Bonsai 2 27B (a PrismML ternary GGUF, derived from Qwen3.8-27B) for a single RTX 3060 12 GB,
with every claim tied to a measurement on that machine.

What that means in practice:

- **Numbers are machine-specific and session-specific.** This is one GA106 card with a
  display attached. Several A/Bs here moved ~2% between sessions, which is why the
  comparisons that matter are run in a single session and why the raw logs are committed
  rather than only summaries. Do not read the tables as transferable to another GPU.
- **There is no support and no release.** Scripts may assume this machine's paths, and
  several are written for one measurement at a time.
- **Mistakes are recorded, not tidied away.** Six predictions in this project were overturned
  by measurement and all six are written up, including a review that found a real bug in the
  verification gate. See [docs/review-findings-sol.md](docs/review-findings-sol.md) for the
  list of corrections.

The one idea worth taking from here is the method: state a mechanism, measure it on the
served path, and let the A/B overrule the mechanism. That is what moved decode from 28.6 to
249 t/s, and it is also what killed four plausible-sounding optimizations.

## What is actually achieved

| workload | before | now | |
|---|---:|---:|---|
| code edit, context reuse (the agentic case) | 28.6 t/s | **249 t/s** | 8.7x |
| same, served through llama-server | - | **248 t/s** | |
| plain chat, no context reuse | 28.6 | 30.6 t/s | +7% (kernel work only) |
| prefill `pp512` | 494 t/s | 494-502 t/s | at the `dp4a` roofline |
| an edit turn on a 13.6K-token conversation | 30.9 s | **1.2 s** | 25x, prompt layout only |

The two large wins are not kernel work:

1. **N-gram speculative decoding** (`--spec-type ngram-simple`), tuned to lookup 6 / draft
   384. It buys several tokens per weight pass, which is the only way past the memory
   bandwidth wall. It is 8x when the output reuses the context and inert when it does not.
2. **Prompt layout.** A prefix cache breaks at the first changed token. A file edited early
   in the prompt costs a full re-prefill every turn; keeping stable material first and
   appending the current file state last costs 1.2 s instead of 30.9 s.

Both require the **client** to cooperate — the server cannot fix either. See
[Client requirements](#client-requirements).

## Requirements

- An RTX 3060 12 GB, a driver with CUDA 12.4 support, CUDA toolkit 12.4 at
  `/usr/local/cuda-12.4`, and about 20 GB of disk.
- The PrismML llama.cpp fork. **Stock llama.cpp cannot run this model**: it rejects
  `PQ2_0`/`PTQ1_0` as unknown types, and loads the development `Q2_0` file without a warning
  and produces garbage.

## Quick start

```bash
scripts/setup.sh all                    # pinned CUDA binary + both GGUF packs, 13 GB
scripts/roofline.sh                     # this card's measured ceilings
scripts/build.sh                        # sm_86 source build; needed for kernel work
PHASES=cli scripts/bench.sh build/llama-sm86/bin PQ2_0   # decode in the served shape
scripts/serve.sh start                  # OpenAI-compatible endpoint on :8080
```

`PHASES=cli` matters: `llama-bench` batches four tokens per pass and takes a different kernel
path than a single-token request, and the two differ by about 2x in bandwidth terms.

## Layout

```
scripts/     setup, build, benchmark, profiling, serving, and gate entry points
fixtures/    the prompts every recorded measurement was taken on
tools/       GGUF header inspector, CUDA microbenchmarks
patches/     the kernel overlay, applied by scripts/build.sh
docs/        measurements, and decisions/ for the ADRs
results/     raw logs and CSVs from every run (tracked; this is the evidence)
third_party/ pinned llama.cpp fork checkout (not tracked)
bin/, models/, out/, build/   downloaded or generated (not tracked)
```

## Client requirements

The largest wins are client-side, and this is the part most likely to be left on the table.

**Prompt layout.** Put the system prompt, tool schemas and unchanged documents first, byte
for byte identical every turn. Keep conversation history append-only. Put the current file
state or a diff last. A client that rebuilds its prompt with the new file contents in place
near the top pays a full re-prefill on every turn.

**Reasoning effort.** `reasoning_effort: "none"` removes the thinking trace, which is most of
the wall-clock time on mechanical work (1.9 s against 11.0 s on a measured edit). It is a
quality trade and belongs per request, not as a server default: on six mechanically graded
tasks `none` scored 5/6 against 6/6 for `medium`, failing a three-operation arithmetic
question with a wrong answer. `thinking_budget_tokens: 0` does **not** disable thinking.

## Where this stands

**Settled:** the pinned fork runs the ternary model correctly on CUDA 12.4; the sm_86 source
build reproduces the pinned binary exactly (perplexity and greedy tokens); prefill is at the
`dp4a` roofline; packing, KV type, batch shape, slot count and context ceiling are decided by
measurement; the serving layer works; and the decode budget is decomposed per kernel. The
configuration to run is [docs/rt3060-profile.md](docs/rt3060-profile.md).

**Open:** general decode without context reuse is DRAM-bandwidth-bound at ~30.6 t/s, and
100 tok/s is not reachable there (it would need about 3x the memory bandwidth). The only path
past it is more tokens per weight pass, and a compatible draft model does not exist for
Bonsai 2 — the earlier Bonsai 1.7B/4B/8B use a 151,669-token vocabulary against this model's
248,320, so they cannot be used as drafters.

**Rejected by measurement:** the `PTQ1_0` packing, quantized KV caches, the `-ub`/`-b` sweep,
`-np 2`, an `nwarps` tuning patch, and raising occupancy through `__launch_bounds__`. Several
of these improved a hardware counter without improving the token rate, which is recorded as
its own lesson.

## License

MIT. See `LICENSE`. The model and the llama.cpp fork carry their own licenses (Apache 2.0 and
MIT respectively); neither is vendored here.
