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

## Correction: the headline numbers are task-shaped, not general

An earlier version of this ADR presented "4.9-6.0x" as a general decode speedup. It is not.
Re-measurement on tasks with little verbatim reuse shows speculation does not fire at all,
and the token rate is unchanged:

| task type | mode | gen t/s | generated | drafts | wall |
|---|---|---:|---:|---:|---:|
| short mechanical edit | none | 28.69 | 42 | **0** | 1.9 s |
| short mechanical edit | medium | 31.80 | 334 | 96 | 11.0 s |
| reasoning / calculation | none | 28.48 | 386 | 0 | 14.0 s |
| reasoning / calculation | medium | 28.68 | 400 | 110 | 14.5 s |

What `none` actually buys is the **removal of thinking tokens**, which shortens only tasks
that were spending their time thinking. Speculation is a second, independent effect that
applies when the output re-quotes the context (long documents, code editing).

Also corrected: the claim that thinking must be off or "nothing fires" holds, but the reverse
framing - that turning thinking off yields 5x - does not. See
`docs/reasoning-mode-trade.md` for the accuracy cost of `none`, which includes a measured
wrong answer on arithmetic.

## Correctness: greedy output is unchanged

`scripts/spec-parity.sh` runs the same prompt at `temperature 0` with `--spec-type none`
and `--spec-type ngram-simple` and compares the generated text. Result: **identical, 40
lines, 0 differing lines** (`results/spec-parity-spec-prompt-code-20260927-173422.txt`).

Speculation verifies drafted tokens against the target model, so at greedy decoding it is
exact rather than approximate: the 4.9x costs nothing in output. With `temperature > 0`
speculation is a standard approximate scheme; that case was not measured here, and per-request
sampling settings should be treated as unverified.

## At depth: the win holds, and it moves the bottleneck

The measurements above are at short context. The target workload is long-context agentic
work, so the same comparison was repeated at 21,113 tokens of prompt
(`results/spec-summary-20260927.txt`, workload `long16k`):

| spec-type | gen t/s | speedup | drafts | accepted |
|---|---:|---:|---:|---:|
| none | 25.25 | 1.00x | 0 | - |
| `ngram-simple` | **101.22** | **4.01x** | 3 | 100% |

Two things fall out of this.

**The win survives depth, and for the reason the mechanic predicts.** Decode at this depth
is 39.6 ms/token without speculation (weight streaming 24.0 ms + KV 4.6 ms + the rest).
With speculation the same work is amortised over ~4 accepted tokens, giving 9.9 ms/token.
Speculation amortises the KV read as well as the weight read, so its relative benefit does
not decay with depth.

**No regression at depth either.** A non-copying task on the same 21K context measured
25.04 t/s without and 24.97 t/s with speculation, with zero drafts: inert, as at short
context.

**But prefill now dominates the request.** A fresh 21K-token request costs 47.7 s of
prefill (21,113 tokens at 444.9 t/s) against 2.5 s of decode with speculation (256 tokens
at 101 t/s). Speculation cut decode from 10.1 s to 2.5 s, which took decode's share of the
request from 18% to **5%**. The next bottleneck is prefill, not decode, which is why the
prefix cache measurement below matters more than any further decode tuning.

## The prefix cache is what makes an agent loop usable

Measured on `llama-server` at 21K context, `results/prefix-cache-20260927.txt`:

| request | prompt_n | cache_n | prompt_ms |
|---|---:|---:|---:|
| fresh 21K context | 21,123 | 0 | 47,743 |
| same context + 1 turn | **26** | **21,186** | **416** |
| same context + 2 turns | 26 | 21,214 | 423 |

A cached turn re-prefills 26 tokens instead of 21,123, **115x less prompt work**. The 47.7 s
is paid once per conversation, not once per turn. Combined with speculation, a cached
mechanical-edit turn costs 0.4 s of prefill plus 2.5 s of decode instead of 10.6 s:
**roughly 3.7x end to end**, not just 5x on the decode phase alone.

## What this does not fix

Prefill is untouched (still at the `dp4a` roofline, ADR 0001), and the GEMV's 15.7% gap to
the bandwidth roofline remains. Speculation changes how many tokens one pass buys, not how
fast a pass runs. On a fresh long-context request it therefore buys little: 47.7 s of
prefill dominates regardless, and prefill is already at the hardware ceiling.
