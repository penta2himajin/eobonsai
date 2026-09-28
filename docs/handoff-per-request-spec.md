# Handoff: per-request speculative decoding selection

> Workstream handoff. The GitHub issue is the live copy; this file is the versioned record.
> Protocol: `docs/handoff-protocol.md`.

## Snapshot

- Branch: `main`
- Last commit: `88b09ba` @ 2026-09-28
- Working tree: clean
- Last session: 2026-09-28 12:0x JST

## Status

in-progress

## Next action

Make `server-schema.cpp`'s speculative request block compile, then reinitialise the slot's
`spec` per task at `server-context.cpp:1720` so a request can choose the speculative method,
and verify that `speculative.type` changes the draft count.

## Verification

Server side, no GPU needed for the first two:

```bash
cd third_party/llama.cpp && git apply --check ../../patches/0003-*.patch   # patch applies cleanly
cmake --build build/llama-sm86 --target llama-server -j8                  # no errors
```

Then, with the server started **without** `--spec-type`:

```bash
# expect drafts = 0 without the request field, and drafts > 0 with it
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"<a long file to copy>"}],"max_tokens":128,
       "reasoning_effort":"none","speculative":{"type":"ngram-simple","ngram_size_n":6,"ngram_size_m":384}}'
# read draft_n in the "timings" object of the response
```

Fallback check: `scripts/bench.sh` with `PHASES=cli` must still show the `patches/0002`
baseline (28.8-30.6 t/s with speculation off), and `scripts/parity-check.sh` must still PASS
with perplexity delta 0.0000% and identical greedy tokens.

Finally the end-to-end goal, which is the reason for the work:

```bash
SPEC_TYPE=none scripts/serve.sh start          # creation turns, speculation off
# ask the client to create a file, then modify the same file with a per-request
# "speculative": {"type": "ngram-simple"} on the modification only
```

Measure both turns: creation should run at the no-speculation rate, modification much faster.

## Context pointers

- Blocked schema block: `third_party/llama.cpp/tools/server/server-schema.cpp` L198 `#if 0`,
  L227 `#endif`. The last field at L205-207 is missing its closing paren.
- Spec context created once from server params:
  `third_party/llama.cpp/tools/server/server-context.cpp` L1189.
- Pattern to copy, sampler reinitialised per task:
  `server-context.cpp` L1720, and `L1748` for a per-request scalar.
- Spec lifetime: declared `common_speculative * spec` at `server-context.cpp` L208 and
  `common_speculative_ptr spec` at L850; reached through `slot.spec` in the decode callback at
  L709. Getting this right is the substance of the change.
- Measurements: `results/per-request-spec-blocked-20260928.txt`,
  `docs/agent-loop-vs-copy-turns.md`, `results/pi-multiturn-20260928.txt`.
- Existing fork change to keep intact: `patches/0002-mmvq-rows-per-block-pq2_0-decode.patch`
  (mmvq.cu, +6.2%). `scripts/build.sh` applies everything in `patches/` after checking out the
  pinned commit, and detects an already-applied patch.
- Acceptance rules: `patches/README.md`. Gate: `scripts/parity-check.sh` (both conditions).
- Client setup already verified: `docs/client-integration.md`, `fixtures/pi-models.json.example`,
  `fixtures/dsh-provider-overlay.yml`.

## Decisions made

- **Speculation is workload-dependent, so the default is off.** Measured through Pi on a real
  agent edit: off 28.2 t/s, draft 32 → 24.9, draft 384 → 16.9; a single long copying answer
  with draft 384 reaches 249 t/s. Recorded in `config/rt3060.env` and
  `docs/agent-loop-vs-copy-turns.md`. Do not re-litigate.
- **The client picks the reasoning field, not the server.** `reasoning_effort: "none"` must be
  sent per request; `thinking_budget_tokens: 0` does not disable thinking.
- **Pi and DSH both preserve the prefix cache naturally** (96.6-99.7% hit after the first
  request) because they append tool results rather than rebuilding the prompt. The
  prompt-layout rule only matters for clients that re-inline file contents.
- **The only fork change kept so far is `patches/0002`.** Everything else was measured and
  rejected.

## Failed approaches

- **Removing the `#if 0` guard and rebuilding.** Fails with:
  `server-schema.cpp:207:85: error: expected ')' before ';' token` /
  `server-schema.cpp:205:8: note: to match this '('`. The block has never been compiled, so its
  syntax was never checked; it is unfinished rather than switched off. Fix the syntax as part
  of the change.
- **Assuming the parse alone would be enough.** It is not: only `params_base.speculative` is
  read after L1189, and `task.params.speculative` is written by the schema and never consumed.
  Both a schema fix and a per-task application are required.
- **Measuring with a stale binary.** After the failed build the old binary was still on disk,
  so a test run reported "no effect" for code that had not been built. Always confirm the
  build succeeded and the binary timestamp advanced before drawing a conclusion.
- Earlier rejected work, do not retry without new evidence: the `PTQ1_0` packing, quantized KV
  caches, the `-ub`/`-b` sweep, `-np 2`, the `nwarps` patch, and raising occupancy via
  `__launch_bounds__`. Each has a measurement recorded under `results/`.

## Open questions for user

- If the fork change turns out to be invasive, is a routing proxy in front of two server
  instances (one with speculation, one without) acceptable instead? It needs no llama.cpp
  change and works today; `tools/logging-proxy.py` already sits in the request path.
- Per-chunk (mid-generation) switching was considered and judged the wrong target: the
  character of a generation does not change within it. Confirm that per-request is enough.
