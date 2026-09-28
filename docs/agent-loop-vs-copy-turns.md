# The prefix cache works in an agent loop; speculation does not

Two unverified items from the client work, measured through Pi 0.87.1 with a real two-turn
file edit against this server. Raw analysis: `results/pi-multiturn-20260928.txt`.

## The prefix cache works by construction in a proper agent loop

Pi never rebuilds its prompt with the edited file in place. It appends tool results to the
conversation, so the prefix stays byte-identical and the cache survives. Six requests over two
turns of editing `calc.py`:

| request | prompt tokens | actually processed | cached | hit rate | prefill |
|---:|---:|---:|---:|---:|---:|
| 1 (first ever) | 6242 | 6242 | 0 | 0% | **13157 ms** |
| 2 | 6354 | 75 | 6279 | 98.8% | 383 ms |
| 3 | 6537 | 187 | 6350 | 97.1% | 646 ms |
| 4 | 6690 | 157 | 6533 | 97.7% | 586 ms |
| 5 | 6804 | 31 | 6773 | **99.5%** | 414 ms |
| 6 | 7041 | 241 | 6800 | 96.6% | 768 ms |

**96.6-99.5% cache hit after the first request, and prefill falls from 13.2 s to 0.4-0.8 s.**
The 25x edit-turn result reproduces through a real client, without any client change: an agent
that appends tool results gets this for free. The prompt-layout rule from
`docs/cache-and-file-edits.md` matters for clients that re-inline file contents; Pi does not.

## Speculation is a net loss in an agent loop

The same edit task, same model, comparing server configurations:

| server speculation | mean generation rate | wall time for one turn |
|---|---:|---:|
| **off** | **28.2 t/s** (27.9 / 28.4 / 28.2) | **24.0 s** |
| `ngram-simple` 6/32 | 24.9 t/s (25.4 / 26.1 / 23.3) | 25.5 s |
| `ngram-simple` 6/384 | 16.9 t/s (13.5-21.5 over 6 requests) | 31.4 s |

Even a short draft loses. An agent turn is a series of **short** generations - tool-call JSON
and brief confirmations, 31 to 202 tokens here - and a long verbatim span for the n-gram
matcher to find rarely exists. The matcher runs, drafts get rejected, and the verification
cost is paid with nothing bought back.

This corrects the earlier conclusion that draft 384 was simply better. It is better **only**
for long copying generations:

| workload | best configuration |
|---|---|
| full-file rewrite in one long answer (`fixtures/prompts/code-edit.txt`) | speculation **on**, draft 384: 249 t/s |
| agent loop of tool calls and short edits (this test) | speculation **off**: 28.2 t/s |

The two differ by 9x in the wrong direction, so the server-wide `SPEC_TYPE` default is a real
choice rather than a detail. For the single-user agent workload this project targets, off is
the better default; the 249 t/s figure requires a workload whose answers are long copies.

## Measured since: per-request and automatic switching

`patches/0003` adds it. A request turns speculation off for its own generation with
`"speculative": {"type": "none"}`; omitting the object keeps the server setting. One server, one
prompt, three reps:

| workload | arm | mean t/s | mean draft_n |
|---|---|---:|---:|
| low-overlap (novel short answer) | off | 29.89 | - |
| low-overlap | on | 29.98 | 0 |
| code-edit (full-file rewrite) | off | 29.80 | - |
| **code-edit** | **on** | **309.16** | **249** (all accepted) |

So the 9x gap above is now a per-request choice rather than a server restart, and the on -> off ->
on transition was walked twice with the counters following it exactly. Only on/off is supported:
the context is built once at load, so a request cannot pick a different method, and any other
name is rejected with a 400 instead of being ignored. Full write-up:
`results/per-request-spec-20260928.txt`.

The low-overlap proxy above does not reproduce the Pi loss in the first table (28.2 off against
24.9 and 16.9 on); with draft 384 and no ngram match it shows "no gain" rather than a loss. The
Pi numbers stay the evidence that off is the right default for an agent loop.

### The server can make the decision itself

`"speculative": {"type": "auto"}` drafts first and stops for the rest of the generation once the
draft acceptance ratio falls under 0.6 (tunable with `accept_min` and `min_draft`). Rejected drafts
cost exactly the target compute they consume, and the recorded acceptance ratios are bimodal with a
wide gap (0.531 against 0.849), so the ratio decides it. It is available after one verification
step: the copy turn drafts 249 tokens and accepts all of them at once, the tool-call turn drafts
about 4 per step and accepts almost none. Same server, three reps:

| workload | arm | mean t/s | mean draft_n |
|---|---|---:|---:|
| verbatim-copy | off | 30.37 | - |
| verbatim-copy | on | 261.71 | 122 (all accepted) |
| **verbatim-copy** | **auto** | **268.94** | **122 (never latched)** |
| roofline (long prompt, short answer) | off | 27.07 | - |
| roofline | on | 20.46 | 1006 (7 accepted, 24% slower) |
| **roofline** | **auto** | **28.03** | **231 (latched after one step)** |

Auto beats the better fixed policy on both, so no configuration decision is left to the client.
Write-up: `results/spec-auto-controller-20260928.txt`.

Still not measured: throughput through DSH.
