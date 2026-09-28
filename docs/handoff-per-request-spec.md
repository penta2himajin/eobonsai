# Handoff: per-request speculative decoding selection

> Workstream handoff. The GitHub issue is the live copy; this file is the versioned record.
> Protocol: `docs/handoff-protocol.md`.

## Snapshot

- Branch: `main`
- Last commit: `fdc9a36` @ 2026-09-28
- Working tree: clean
- Last session: 2026-09-28 15:46 JST

## Status

ready-for-review

## Next action

Decide who opts into automatic switching, then confirm it through a real client. Automatic mode
works when a request sends `"speculative": {"type": "auto"}`; the server then starts drafting and
stops for the rest of the generation once the draft acceptance ratio falls under 0.6. Two options
are open and the difference is one line: leave it opt-in, so a client needs the static field once in
its provider config, or make auto the server default whenever `--spec-type` enables speculation, so
no client changes at all. Whichever is chosen, verify through Pi or DSH that a creation turn holds
the no-speculation rate and a modification turn drafts, using `timings.draft_n` from the response.

## Verification

```bash
# server started with --spec-type ngram-simple; the binary must be rebuilt from the patches first
python3 scripts/per-request-spec-probe.py --mode switch   # draft_n 249 on, absent off, 400 on a wrong method
python3 scripts/per-request-spec-probe.py --mode auto     # see the two expected tables below
python3 scripts/per-request-spec-probe.py --mode matrix   # 29.80 off / 309.16 t/s on, code-edit
```

Expected for `--mode auto`, three reps: verbatim-copy off 30.37 / on 261.71 / auto 268.94 t/s with
122 drafts and no latch; roofline off 27.07 / on 20.46 / auto 28.03 t/s with 231 drafts drafted
instead of 1006. The latch must print `speculative auto: stopping, N of M drafted tokens accepted`.

Gates, green at `fdc9a36`:

```bash
bash scripts/parity-check.sh bin/cuda build/llama-sm86/bin PQ2_0   # 5.3590 both, delta 0.0000%, PASS
PHASES=cli bash scripts/bench.sh build/llama-sm86/bin PQ2_0        # 30.1-30.6 t/s, inside the 0002 baseline
bash scripts/build.sh   # fresh checkout: applies 0002 and 0003, then builds llama-server
```

`scripts/build.sh` must also be idempotent: a second run has to report "patch already applied" for
both patches rather than failing. That is exactly what a stacked patch broke, see Failed approaches.

## Context pointers

- Patch: `patches/0003-server-per-request-speculative.patch`, one stack-free patch carrying the
  on/off gate and the auto controller. Rationale and both measurement tables in `patches/README.md`
  §0003.
- Schema field: `third_party/llama.cpp/tools/server/server-schema.cpp` L197-260 (nested
  `speculative` with `type`, `accept_min`, `min_draft`).
- Slot gate and controller: `server-context.cpp` L1752-1762 in `launch_slot_with_task`, and the
  latch at L2927-2945 in the per-step drafting block. `can_speculate()` is L426,
  `get_n_draft_max()` is L439, the accept block that consumes a draft is L3830-3900.
- Task fields: `server-task.h` L79-83.
- Evidence: `results/spec-auto-controller-20260928.txt` (auto summary),
  `results/per-request-spec-20260928.txt` (on/off),
  `results/schema-probe-20260928.txt` (the three blockers in the upstream block),
  `results/per-request-spec-auto-20260928-154122.txt` (raw three-arm table),
  `results/parity-spec-auto-20260928-154529.log`.
- Harness: `scripts/per-request-spec-probe.py`, modes `switch`, `matrix`, `auto`.
- Client setup: `docs/client-integration.md`, `fixtures/pi-models.json.example`,
  `fixtures/dsh-provider-overlay.yml`.
- Environment: the DSH sandbox under the default `workspace-write` file policy hides `/dev/nvidia*`,
  so `nvidia-smi` and any GPU run fail unless the session is set to `danger-full-access`. The driver
  and card are fine; `/sys/bus/pci/devices/0000:01:00.0/driver` is bound.

## Decisions made

- **The switch is on/off or automatic; the method is fixed at load.** The speculative context is
  built once from the launch parameters (`server-context.cpp:1189`), so a request cannot select a
  method. `"none"`, `"auto"` and the server's own set are accepted; anything else is a 400 naming
  what the server runs, rather than the silent ignore this field produced before.
- **Automatic mode decides on the draft acceptance ratio**, not on the prompt. A rejected draft
  costs exactly the target compute it consumes, the recorded ratios are bimodal with a wide gap
  (0.531 against 0.849), and the ratio separates the two regimes after a single verification step.
  Defaults `accept_min` 0.6 and `min_draft` 8, both overridable per request.
- **The latch only fires at a step with no draft pending.** `spec_draft` and `spec_i_batch` are
  consumed as a unit by the accept block; flipping the switch with a draft in flight would leave
  drafted tokens in the batch that are neither accepted nor dropped.
- **A learned decisioner is deferred, not rejected.** If the controller proves insufficient, the
  follow-up is a small classifier, possibly one of the Jev-clone-family models built on
  GLiNER-2.5-base. Notes for that stage: the labels are already free (every recorded request logged
  drafts and accepted counts), the natural home is `tools/logging-proxy.py` so the fork stays
  untouched, and no ML runtime is installed today (`torch`, `onnxruntime`, `transformers`, `sklearn`
  all absent; only numpy). A prompt-side classifier is structurally worse informed than the
  controller, which sees the draft outcome rather than guessing it.
- **Speculation is workload-dependent, so the server default stays off.** Recorded in
  `config/rt3060.env` and `docs/agent-loop-vs-copy-turns.md`. Do not re-litigate. Whether auto
  should become the default once `--spec-type` is set is the open question below.
- **The client picks the reasoning field, not the server.** `reasoning_effort: "none"`.
- **The overlay is stack-free.** Two patches in `patches/` must not touch the same lines.
- **The only fork changes kept are `patches/0002` and `patches/0003`.**

## Failed approaches

- **Stacking the auto controller as `patches/0004`.** Both patches touch the same lines of
  `server-schema.cpp` and `server-context.cpp`, so on the patched tree `0003` could no longer be
  reverse-applied. `scripts/build.sh`'s "patch already applied" check then fell through to a forward
  apply that failed, and a second `build.sh` run would have exited 1. Folded into one patch file.
- **Fixing only the unclosed paren in the guarded block.** The paren is on the `speculative.n_min`
  call, not `ngram_min_hits` as the previous note claimed. Fixing it compiles and then fails to
  link: `undefined reference to 'unsigned short common_json::get<unsigned short>() const'`, because
  three fields bind `uint16_t` ngram sizes. Avoided by not registering those fields.
- **Registering the block as it was written.** It matches flat dotted keys (`"speculative.type"`),
  not the nested object clients send; `has_value` is a literal lookup (`server-schema.cpp:579`).
  The earlier "accepted and ignored" reading was weaker than recorded - the nested field was never
  looked up at all.
- **Starting the server with `--spec-type none` and enabling speculation per request.** Dead end.
  `common_speculative_init` returns `nullptr` when no implementation is enabled
  (`common/speculative.cpp:3409`), so there is no context to turn on.
- **Comparing against `params_base.speculative.types` as-is.** `--spec-type` appends to the default
  `{none}` (`common/arg.cpp:4187`), so that vector is `{none, ngram-simple}` and a legitimate
  `ngram-simple` request was rejected. Drop `none` before comparing. The rejected run is kept as
  `results/per-request-spec-switch-prefix-append-bug-20260928.json`.
- **A routing proxy in front of two server instances.** Not viable on this card: two instances need
  twice the 6872 MiB of weights against 11904 MiB visible, and the prefix cache would be split.
- Earlier rejected work, do not retry without new evidence: the `PTQ1_0` packing, quantized KV
  caches, the `-ub`/`-b` sweep, `-np 2`, the `nwarps` patch, and raising occupancy via
  `__launch_bounds__`.

## Open questions for user

- Should automatic mode become the server default when `--spec-type` enables speculation, so no
  client change is needed at all? Auto won on both measured workloads, including against always-on.
- Close this issue, or keep it open until a real client runs an alternating agent loop against it?
- The controller's loss arm is synthetic (the roofline doc as a long prompt with a short answer,
  24% loss). The Pi agent turn lost about 50% and used the checkpoint path. Worth reproducing that
  path to confirm the latch behaves the same there?
