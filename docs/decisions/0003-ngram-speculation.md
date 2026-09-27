# ADR 0003: n-gram speculative decoding is the only lever past the decode roofline

- Status: **accepted** (measured 2026-09-27)
- Date: 2026-09-27
- Context: RTX 3060 12 GB, `PQ2_0`, single user, interactive chat and agent use

## Why this is the only remaining lever

The measured decode profile (nsys, `results/nsys-d8.*`, and `docs/decode-profile.md`):

- decode is **34.45 ms/token and the GPU is saturated** (34.45 ms of kernel time against
  34.22 ms of wall clock, i.e. 100.7%). Launch overhead starves nothing, so CUDA graphs
  and launch batching cannot help.
- **82.6% of that time is the ternary GEMV** (`mul_mat_vec_q`), running at **84.3% of the
  measured 300 GB/s bandwidth roofline** (28.45 ms against a 24.0 ms floor).
- The other 17.4% is spread over about twenty kernel types; the largest single one is
  2.8%.

Even a perfect GEMV leaves 30 ms/token. The roofline itself only moves if one weight pass
produces more than one token. Speculative decoding does exactly that, and the fork
implements it **without a draft model** via `--spec-type ngram-*`.

## Measured

`scripts/specbench.sh` (CLI) and a live `llama-server` A/B, all at 256 generated tokens,
`temperature 0`, same prompt. Raw logs under `results/spec-*`.

| Workload | thinking | spec-type | gen t/s | speedup | drafts | accepted |
|---|---|---|---|---:|---:|---:|
| Verbatim copy | off | none | 28.58 | 1.00x | 0 | - |
| Verbatim copy | off | `ngram-simple` | **171.56** | **6.00x** | 5 | 100% |
| Verbatim copy | off | `ngram-mod` | 142.58 | 4.99x | 4 | 100% |
| Verbatim copy | off | `ngram-cache` | 62.49 | 2.19x | 31 | 100% |
| Code edit (comment each function) | off | none | 28.56 | 1.00x | 0 | - |
| Code edit | off | `ngram-simple` | **139.88** | **4.90x** | 6 | 100% |
| Code edit | **on** | none | 28.45 | 1.00x | 0 | - |
| Code edit | **on** | `ngram-simple` | 28.43 | 1.00x | **0** | - |
| Plain chat, no overlap | off | `ngram-simple` | 28.49 | 1.00x | 0 | - |

Serving path, same code-edit prompt, `--spec-type ngram-simple` on `llama-server`:

| Request | gen t/s | `draft_n` | accepted | reasoning chars |
|---|---:|---:|---:|---:|
| default (medium thinking) | 28.46 | 0 | 0 | 970 |
| `thinking_budget_tokens: 0` | 28.51 | 0 | 0 | **970** |
| **`reasoning_effort: "none"`** | **146.02** | 279 | 237 (85%) | 0 |
| `reasoning_effort: "low"` | 28.74 | 48 | 12 (25%) | 1270 |
| `reasoning_effort: "minimal"` | rejected (HTTP 500) | - | - | - |

## The finding that matters: thinking must be off

**Speculation produces zero drafts while the model is thinking.** The thinking trace does
not quote the context, so there is nothing for an n-gram matcher to propose from. Every
configuration with thinking on measured 1.00x, whether from the CLI or the server.

Two traps found by measurement:

1. **`thinking_budget_tokens: 0` does not disable thinking.** It still produced 970
   characters of reasoning and zero drafts. Only `reasoning_effort: "none"` turned it off.
2. **The help does not list `none`.** It lists `minimal` (rejected), `low`, `medium`,
   `high`, `xhigh`, `max`. `low` still thinks and measured 25% draft acceptance, which is
   not enough to pay for itself.

## Decision

- **Enable `SPEC_TYPE=ngram-simple` as the served default.** When the output does not
  reuse the context, speculation is inert rather than harmful (28.49 vs 28.57 t/s, a 0.3%
  difference inside noise), so there is nothing to lose by leaving it on.
- **Keep the server-wide reasoning default at `medium`.** Turning thinking off is a
  quality decision that belongs to the caller, not to the server default.
- **Agentic and mechanical work should send `reasoning_effort: "none"` per request**, which
  turns the same request from 28 t/s into 140-146 t/s. The API already supports it
  per request, so one server serves both modes.

## Consequences

- For the workload this project targets (agent loops that repeatedly read and rewrite
  files), decode is **~5x faster**, which moves the bottleneck back to prefill and to the
  model's own thinking budget.
- For reasoning-heavy chat, nothing changes: speculation is inert and the measured
  numbers are identical to the pre-ADR baseline.
- `ngram-cache` is strictly worse than `ngram-simple` and `ngram-mod` here (2.19x), so it
  is not worth carrying as an option.
- The 6.00x verbatim figure is an upper bound with 100% acceptance. Real code editing
  measured 4.90x with a 6-draft average; longer drafts help when the copied span is long.

## Correctness: greedy output is unchanged

`scripts/spec-parity.sh` runs the same prompt at `temperature 0` with `--spec-type none`
and `--spec-type ngram-simple` and compares the generated text. Result: **identical, 40
lines, 0 differing lines** (`results/spec-parity-spec-prompt-code-20260927-173422.txt`).

Speculation verifies drafted tokens against the target model, so at greedy decoding it is
exact rather than approximate: the 4.9x costs nothing in output. With `temperature > 0`
speculation is a standard approximate scheme; that case was not measured here, and per-request
sampling settings should be treated as unverified.

## What this does not fix

Prefill is untouched (still at the `dp4a` roofline, ADR 0001), and the GEMV's 15.7% gap to
the bandwidth roofline remains. Speculation changes how many tokens one pass buys, not how
fast a pass runs.
