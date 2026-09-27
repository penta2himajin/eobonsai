# Reasoning mode: measured speed and accuracy trade

`reasoning_effort` is the single largest performance lever in this setup, and the only one
that can silently break output. This document records both sides: what each mode costs in
time, and what it costs in correctness.

Evidence: `results/quality-by-mode-20260928.txt` (accuracy), 
`results/reasoning-ab-20260927-235000.txt` (speed on task-shaped prompts).

## The correction that started this

An earlier write-up in this project claimed a **5.1x** speedup from
`reasoning_effort: "none"` on the served path (28.46 -> 146.02 t/s) and framed it as
"unrecovered 5x". Re-measurement shows that figure came from one task where the output
re-quoted a large part of the input, so n-gram speculation fired heavily. On tasks with
little verbatim reuse, speculation does not fire at all under `none` and the token rate is
unchanged:

| task | mode | gen t/s | generated tokens | drafts | wall |
|---|---|---:|---:|---:|---:|
| short mechanical edit | none | 28.69 | 42 | **0** | **1.9 s** |
| short mechanical edit | medium | 31.80 | 334 | 96 | 11.0 s |
| reasoning / calculation | none | 28.48 | 386 | 0 | 14.0 s |
| reasoning / calculation | medium | 28.68 | 400 | 110 | 14.5 s |

So the honest statement is: **`none` does not speed up token generation; it removes
thinking tokens, which shortens only tasks that were spending time thinking.**

## Accuracy, scored mechanically

Six tasks with answers that need no judge: three arithmetic (exact number), two format
(line count and shape), one JSON transform (compared to a computed key).

| mode | accuracy | failures |
|---|---|---|
| `server-default` (medium) | **6/6 = 100%** | - |
| `medium` | **6/6 = 100%** | - |
| **`none`** | **5/6 = 83%** | `arith_2` |

The failure is the important part. Asked for `847 * 29 + 1147`:

```
none    : replied "25700" in 6 tokens, 0.5 s   (correct: 25710)
medium  : replied "25710" after 197 tokens, 7.2 s
```

`none` answers immediately and gets it wrong by 10. It is not a formatting difference or a
verbosity difference: it is a **wrong answer**, produced fast.

## The pattern in the per-task numbers

`none` cuts thinking to zero, and the generated-token count collapses with it:

| task | `none` tokens | `medium` tokens | `none` outcome |
|---|---:|---:|---|
| arith_2 (847*29+1147) | 6 | 197 | **wrong** |
| format_1 (5 status codes) | 30 | 213 | correct |
| format_2 (3 fixed lines) | 10 | 69 | correct |
| transform_1 (JSON) | 32 | 405 | correct |
| arith_1 (context budget) | 567 | 639 | correct |
| arith_3 (tensor size) | 402 | 206 | correct |

Tasks that are purely mechanical survive on 10-32 tokens. The one that needed a multi-step
computation did not, and produced 6 tokens to answer a 3-operation arithmetic question. That
is the discriminator: **`none` is safe where no intermediate reasoning is required, and
unsafe where it is.**

## Decision

- **Do not make `none` the server default.** It costs a measured 17% accuracy on a task set
  this small, and the failure mode is silent wrong answers rather than errors.
- **Keep `medium` as the server default.** It scored 100% and its speed is the same as
  `server-default`, since they are the same mode reached two ways.
- **Use `none` per request, for work with no reasoning content**: file reads and rewrites,
  mechanical edits, format conversions, verbatim re-output. There it is a genuine win
  (1.9 s vs 11.0 s on the short mechanical task) at no measured accuracy cost.
- The `low` mode remains useless: it still thinks, and draft acceptance was 25%, so it
  neither saves the thinking time nor enables speculation well.

## Caveat on scope

Six tasks is enough to catch a mode that fails arithmetic outright; it is not enough to
certify `none` as safe for mechanical work in general. The mechanical tasks here passed, and
the failure was confined to the arithmetic one, which is consistent with the mechanism
described above - but this is a small sample and should be said plainly.

## Decode throughput with tuning, against the 100 tok/s target

`scripts/ngram-tune.sh` swept the n-gram lookup and draft lengths. The fork's defaults
(12/48) are not tuned for this model; lookup=6 draft=128 is better or equal everywhere:

| workload | no speculation | defaults 12/48 | tuned 6/128 |
|---|---:|---:|---:|
| full-file rewrite | 28.6 t/s | 138.5 t/s | **194.3 t/s** |
| explain + quote 3 functions | 28.3 | 42.7 | **44.1** |
| plain chat, no context reuse | 28.3 | 28.3 | 28.3 (no regression) |

Served path with the tuned values: **225.4 t/s**, 248 drafts and 248 accepted on the rewrite.

Against the 100 tok/s target: **met and exceeded for work that reuses the context** (the bulk
of agentic editing), reached in part for partial reuse (44 t/s), and **out of reach for
general decode at 28.3 t/s**, which is DRAM-bandwidth-bound. No mechanism exists to raise
that: the only lever is more tokens per weight pass, and the compatible draft model for that
does not exist (the earlier Bonsai 1.7B/4B/8B use a different vocabulary, 151,669 against
248,320, so they cannot serve as drafters).

## Reproduce

```bash
scripts/serve.sh start
python3 scripts/quality-check.py            # accuracy, all modes
bash scripts/reasoning-ab.sh                # speed on task-shaped prompts
```
