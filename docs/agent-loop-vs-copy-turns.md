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

## Not measured

Throughput through DSH (only its route and reasoning field were verified), and whether a
per-request mechanism could choose speculation per request rather than per server run.
