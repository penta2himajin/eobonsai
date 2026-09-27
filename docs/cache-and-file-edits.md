# Prefix cache and file edits: a 25x difference from prompt layout alone

The earlier cache experiment (`results/prefix-cache-20260927.txt`) appended questions to an
**unchanged** context. It never showed what happens when an agent rewrites a file, which is
what an agent actually does. An external review flagged this as the one untested
optimization (`docs/review-findings-sol.md`); this document tests it.

## Result

Same 13.6K-token prompt, same edit, three layouts. The only difference is where the edited
file sits relative to the stable material. Reproduced three times
(`results/cache-edit-test-20260928.txt`):

| layout | turn 2 new tokens | tokens reused | turn 2 prefill | vs a fresh prefill |
|---|---:|---:|---:|---:|
| stable material first, file last | 565 | 13,129 | **1.66 s** | **0.05x** |
| **file first, stable material after** | 13,694 | **0** | **30.9 s** | **1.00x** |
| **original kept, new state appended last** | 386 | 13,651 | **1.22 s** | **0.04x** |

Repeats: 1.66/1.67 s, 30.9/31.1 s, 1.21/1.22 s. The effect is ~30 s and does not move.

**Editing a file that sits before the stable material costs a full re-prefill of the entire
context on that turn.** Editing one that sits after it costs 1.7 s. Keeping the original and
appending the new state costs 1.2 s.

## Mechanism

A prefix cache matches from the start and breaks at the first differing token. Every token
after the divergence must be re-prefilled. This is the same property that makes the existing
cache result work, applied to a case the earlier test did not cover:

- In the `file_late` layout the divergence is at token ~13,100, so ~13,100 tokens are reused
  and only the tail is rebuilt.
- In the `file_early` layout the divergence is at token ~250, so **nothing** is reusable and
  the whole 13.6K context is recomputed - which is exactly the naive behaviour of a client
  that rebuilds its prompt with the current file contents in place.

The appended layout wins because it changes no existing byte: the stable material, the
original file and the history all stay identical, and only the new tail is processed.

## The rule this implies

For any client that feeds files to this model:

1. **Stable material first and never mutated.** System prompt, tool schemas, repository
   documentation, and any file not being edited. This is what gets reused.
2. **Conversation history in the middle, append-only.** Never rewrite or reorder earlier
   turns; a change at turn 2 invalidates turns 3..N.
3. **The current file state, or a diff, last.** Appending a fresh copy keeps the prefix
   intact; replacing the file in place near the top destroys it.

Measured cost of the appended copy: ~300 tokens of extra prompt in exchange for ~30 s per
turn. At 493 t/s prefill those 300 tokens cost ~0.6 s, so the trade is strongly positive
even before considering that they are only paid once per turn.

## Caveat the review raised, and it stands

The model now sees both the old and the new version of the file. Superseded versions
accumulate in the context and must be handled explicitly - either by keeping only the newest
appended copy, or by accepting the confusion risk. This experiment measured cache behaviour
and latency; it did **not** test whether the model edits correctly when both versions are
present. That check is still outstanding, and it is the same gap the review named.

## A measurement-hygiene note

The first two attempts at this experiment were invalid and it is worth recording why. The
scenarios shared a stable prefix, and the server's cache is shared across requests, so:

- scenario 2 ran with scenario 1's prefix already cached, making its "turn 1 (fresh)" a
  cache hit (13,123 tokens reused instead of 0);
- a later re-run of the whole script reused the previous *run's* cache, because the nonces
  were per scenario rather than per run, so every "fresh" measurement was a hit.

`scripts/cache-edit-test.py` now stamps a run-unique nonce into every prompt. This is the
same class of error as the earlier ones in this project: the measurement looked right and was
answering a different question than intended.
