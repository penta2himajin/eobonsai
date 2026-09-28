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

**REMOVED.** Measured in the `llama-bench` shape (`ncols_dst = 4`) it gave +2.0%
(`tg128` 29.59 -> 30.19 t/s). In the served shape (`ncols_dst = 1`, `llama-cli` streaming)
it is **neutral**: 28.8 t/s patched, 28.8 t/s unpatched, and 28.8 t/s for the pinned
prebuilt that never had it. An earlier reading of -0.6% was measurement noise.

It was removed because a kernel change that buys nothing for the target workload is not
worth carrying, especially when the A/B that justified it ran in a different shape. See
`docs/gemv-benchmark-artifact.md` for the full correction.

## 0002-mmvq-rows-per-block-pq2_0-decode.patch

Applies to `adfffbe41b2cabcd51fff326ab045662265062bb`.

Sets decode `rows_per_block` to 2 for the ternary types on the generic device table, so two
output rows share one read of the quantised activation. The kernel's K loop is already
outermost, so this reduces L1 traffic without touching arithmetic or accumulation order.

Measured, served shape, same session, three reps each:

| | gen t/s | |
|---|---:|---|
| R=1 (before) | 28.83 | |
| **R=2** | **30.50** | **+5.8%**, re-confirmed at 30.63 vs 28.83 (+6.2%) |
| R=4 | 28.93 | register pressure |
| R=8 | 28.23 | register pressure |

With speculation on (ngram-simple 6/128): 231.27 -> 232.97 t/s (+0.7%).

Numerics unchanged: perplexity delta 0.0000%, identical greedy tokens.

New patches must still be A/B'd in the **served shape** via `scripts/bench.sh` with
`PHASES=cli`, and must pass both conditions of `scripts/parity-check.sh`.

Background: a standalone microbenchmark had predicted +20% for this change and was wrong
by 10x, because its simplified K-loop was itself slower than the real kernel's at the same
configuration. See [ADR 0004](../docs/decisions/0004-gemv-launch-config.md); the lesson is
that launch-configuration sweeps in simplified kernels do not transfer.

## 0003-server-per-request-speculative.patch

Applies to `adfffbe41b2cabcd51fff326ab045662265062bb`.

Server-side, not a kernel change. Lets a request turn speculation off for its own generation
with `"speculative": {"type": "none"}`, so one server can run creation turns without
speculation and modification turns with it. The upstream block that declared these request
fields was guarded by `#if 0`, had an unclosed paren, registered flat dotted keys that no
client sends, and linked only if `common_json::get<unsigned short>` was added; the patch
replaces it with a `field_nested("speculative")` holding a single `type` subfield. Only
on/off is supported, because the speculative context is built once at load; any other method
name is rejected with a 400 rather than silently ignored.

Measured on the served shape, one server, `ngram-simple` 6/384, `fixtures/prompts/code-edit.txt`
(max_tokens 256), three reps:

| workload | arm | mean t/s | mean draft_n |
|---|---|---:|---:|
| low-overlap (novel short answer) | off | 29.89 | - |
| low-overlap | on | 29.98 | 0 |
| code-edit (full-file rewrite) | off | 29.80 | - |
| **code-edit** | **on** | **309.16** | **249** (all accepted) |

The on -> off -> on transition was walked twice with the counters following it exactly, so an
off request leaves no stale draft behind.

Gates: `scripts/parity-check.sh` PASS (perplexity 5.3590 both, delta 0.0000%, identical greedy
tokens); `PHASES=cli scripts/bench.sh` 30.1 / 30.6 / 30.3 t/s, inside the 0002 baseline of
28.8-30.6 t/s.

Full write-up: [`results/per-request-spec-20260928.txt`](../results/per-request-spec-20260928.txt).
The schema investigation, including the three blockers found in the upstream block, is in
[`results/schema-probe-20260928.txt`](../results/schema-probe-20260928.txt).
