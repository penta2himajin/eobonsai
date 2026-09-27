# Prompt fixtures

These are the prompts the documented measurements were taken on. They live in the repository
because several scripts default to them and because a number in `docs/` is only reproducible
if the input that produced it is here.

Each one probes a different amount of context reuse, which is what decides whether n-gram
speculation can fire at all:

| fixture | reuse | what it tests | recorded result |
|---|---|---|---|
| `code-edit.txt` | high | add a comment above every function in a file, output the whole file. The agentic workload. | 249.3 t/s with `ngram-simple` 6/384, against 30.6 t/s without (≈8x) |
| `verbatim-copy.txt` | very high | reproduce a passage exactly, an upper bound on acceptance | 171.6 t/s at the default parameters |
| `moderate-quote.txt` | partial | explain a file and quote three functions in full | 46.9 t/s |
| `low-overlap-chat.txt` | none | a normal question, so speculation finds nothing | 30.37 t/s, no gain |
| `review-prompt.md` | none | the adversarial review put to an external model | `results/consult-sol-output.txt` |

`code-edit.txt` is the headline workload: it is what an agent does when it reads and rewrites
a file, and it is where the largest measured gains come from.

## Long-context prompts

`out/spec-prompt-long.txt` and `out/spec-prompt-longchat.txt` are around 70 KB each and are
built from three tracked engineering documents. They are generated, not committed:

```bash
python3 fixtures/prompts/make-long.py
```

They will not shrink or grow to match a past measurement: the documents they are built from
keep changing, so a re-run today produces a slightly different prompt than the one used for
the 21K-token cache and depth numbers. The conclusions those measurements support are
structural (a cache breaks at the first differing token; speculation is inert without reuse),
but the absolute token counts are tied to the document revision of the day.

## Why this directory exists

These prompts originally lived in `out/`, which is gitignored, so a fresh clone had no input
for `scripts/ngram-tune.sh` and the numbers in `docs/` could not be reproduced. Anything a
recorded measurement depends on belongs in the repository.
