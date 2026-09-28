# Handoff: per-request speculative decoding selection

> Workstream handoff. The GitHub issue is the live copy; this file is the versioned record.
> Protocol: `docs/handoff-protocol.md`.

## Snapshot

- Branch: `main`
- Last commit: `5159839` @ 2026-09-28
- Working tree: clean
- Last session: 2026-09-28 15:25 JST

## Status

ready-for-review

## Next action

Wire one client to send the field per turn - `"speculative": {"type": "none"}` on creation turns,
the field omitted or set to the server's own type on modification turns - then confirm from the
response `timings.draft_n` that the counters follow the turn type in one server run. The server
half is finished and measured; only the client half is open.

## Verification

The server half is done. These reproduce it against a server started with `--spec-type ngram-simple`:

```bash
python3 scripts/per-request-spec-probe.py --mode switch   # draft_n 249 on the on rows, no draft_n key on the off rows
python3 scripts/per-request-spec-probe.py --mode matrix   # code-edit: 29.80 t/s off, 309.16 t/s on
```

Gates, already green at `5159839`:

```bash
bash scripts/parity-check.sh bin/cuda build/llama-sm86/bin PQ2_0   # 5.3590 both, delta 0.0000%, PASS
PHASES=cli bash scripts/bench.sh build/llama-sm86/bin PQ2_0        # 30.1-30.6 t/s, inside the 0002 baseline
bash scripts/build.sh   # fresh checkout: applies 0002 and 0003 and builds llama-server
```

For the client half: in a single server run, a modification turn must report `draft_n > 0` and a
creation turn must have no `draft_n` key at all.

## Context pointers

- Patch: `patches/0003-server-per-request-speculative.patch`; rationale and measurements in
  `patches/README.md` §0003.
- Schema field: `third_party/llama.cpp/tools/server/server-schema.cpp` L197-227.
- Slot gate: `server-context.cpp` L1750-1756 in `launch_slot_with_task`; `can_speculate()` is
  `server-context.cpp` L426; `get_n_draft_max()` is L439.
- Evidence: `results/per-request-spec-20260928.txt` (summary),
  `results/per-request-spec-matrix-20260928-152232.txt`, `results/schema-probe-20260928.txt`
  (the three blockers in the upstream block), `results/parity-per-request-spec-20260928-152128.log`.
- Harness: `scripts/per-request-spec-probe.py`, modes `switch` and `matrix`.
- Client setup: `docs/client-integration.md`, `fixtures/pi-models.json.example`,
  `fixtures/dsh-provider-overlay.yml`.
- Environment: the DSH sandbox under the default `workspace-write` file policy hides `/dev/nvidia*`,
  so `nvidia-smi` and any GPU run fail unless the session is set to `danger-full-access`. The driver
  and card are fine; `/sys/bus/pci/devices/0000:01:00.0/driver` is bound.

## Decisions made

- **Per-request speculation is on/off only.** The speculative context is built once from the launch
  parameters (`server-context.cpp:1189`), so a request cannot select a method. Any name other than
  `none` or the server's own set gets a 400 naming what the server runs, rather than the silent
  ignore this field produced before.
- **`"none"` turns it off; an absent field keeps the server setting.** Per-request params inherit
  `params_base.speculative` (`server-schema.cpp:524`), so no "was the field present" flag is needed.
- **Speculation is workload-dependent, so the server default stays off.** Measured through Pi on a
  real agent edit: off 28.2 t/s, draft 32 -> 24.9, draft 384 -> 16.9; a single long copying answer
  with draft 384 reaches 249 t/s. Recorded in `config/rt3060.env` and
  `docs/agent-loop-vs-copy-turns.md`. Do not re-litigate.
- **The client picks the reasoning field, not the server.** `reasoning_effort: "none"` must be sent
  per request.
- **The only fork changes kept are `patches/0002` and `patches/0003`.**

## Failed approaches

- **Fixing only the unclosed paren in the guarded block.** The paren is on the `speculative.n_min`
  call, not `ngram_min_hits` as the previous note claimed (paren audit: that statement is +1, all
  others 0). Fixing it compiles and then fails to link:
  `undefined reference to 'unsigned short common_json::get<unsigned short>() const'`, because three
  fields bind `uint16_t` ngram sizes. The patch avoids this by not registering those fields at all.
- **Registering the block as it was written.** It matches flat dotted keys (`"speculative.type"`),
  not the nested object clients send: `{"speculative": {"type": ...}}` never reaches the field.
  `has_value` is a literal lookup (`server-schema.cpp:579`). Every other grouped option in the file
  uses `field_nested`, and so does the patch. The earlier "accepted and ignored" reading was weaker
  than recorded - the nested field was not looked up at all.
- **Starting the server with `--spec-type none` and enabling speculation per request.** Dead end.
  `common_speculative_init` returns `nullptr` when no implementation is enabled
  (`common/speculative.cpp:3409`), so there is no context to turn on. A single-server design must
  launch with speculation on and turn it off per request.
- **Comparing against `params_base.speculative.types` as-is.** `--spec-type` appends to the default
  `{none}` (`common/arg.cpp:4187`), so that vector is `{none, ngram-simple}` and a legitimate
  `ngram-simple` request was rejected with a 400. Drop `none` before comparing. The rejected run is
  kept as `results/per-request-spec-switch-prefix-append-bug-20260928.json`.
- **A routing proxy in front of two server instances.** Not viable on this card: two instances need
  twice the 6872 MiB of weights against 11904 MiB visible, and the prefix cache would be split
  across processes, confounding the comparison. Do not retry.
- Earlier rejected work, do not retry without new evidence: the `PTQ1_0` packing, quantized KV
  caches, the `-ub`/`-b` sweep, `-np 2`, the `nwarps` patch, and raising occupancy via
  `__launch_bounds__`.

## Open questions for user

- Close this issue, or keep it open until a client sends the field per turn end to end?
- `common_speculative_process` (`server-context.cpp:3658`) still runs for an off request, because it
  is gated on the server-wide pointer rather than the slot. It measured as not material
  (29.80-29.89 t/s against a 28.8-30.6 baseline), so it was left alone. Remove it anyway?
- The low-overlap proxy does not reproduce the Pi agent-loop loss. If the creation-turn case needs
  a real measurement rather than a proxy, that means driving Pi or DSH with the field per turn.
