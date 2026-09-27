# Adversarial review by GPT-6 Sol (max effort): findings and corrections

An external review was run against this repository with the explicit instruction to find what
the author missed: `gpt-6-sol` at `model_reasoning_effort="max"`, read-only sandbox, whole
repo in scope. Raw output: `results/consult-sol-output.txt` (191,992 tokens used).

Every claim below was checked against this repository's own data before being accepted.
Three were verified as stated, one was verified to a specific number, and two are accepted
as over-extrapolations by the author's own standard.

## Accepted and corrected

### 1. ADR 0006's "no GEMV headroom" was an over-extrapolation

The review's strongest point. ADR 0006 profiles six launches, finds the 17408-row shape at
277.7 GB/s = 101% of the ~275 GB/s ceiling, and concludes the GEMV has nothing to recover.
But the review measured from the GGUF that the 17408-row shape is only **42.5% of PQ weight
bytes**. **Verified independently here** (byte census of the local GGUF):

| rows | tensors | GB | share | measured GB/s |
|---:|---:|---:|---:|---|
| 1024 | 32 | 0.04 | 0.6% | 176-184 |
| **5120** | 128 | **2.05** | **31.7%** | **232-242** |
| 6144 | 48 | 0.40 | 6.2% | 187 |
| 10240 | 48 | 0.67 | 10.4% | 206 |
| 12288 | 16 | 0.27 | 4.2% | 211 |
| **17408** | 128 | **3.03** | **46.9%** | **250-278** |

A byte-weighted calculation, which is what ADR 0006 should have done:

| sample | weighted average | % of 275 GB/s ceiling | total GEMV | gap vs ideal |
|---|---:|---:|---:|---:|
| ncu-served run | 245.8 GB/s | 89.4% | 26.3 ms | **2.8 ms = 8.2% of a token** |
| ncu-cli run | 231.9 GB/s | 84.3% | 27.9 ms | **4.4 ms = 12.9% of a token** |

**Corrected position: the GEMV headroom is 8-13% of decode, concentrated in the 5120-row
shape (31.7% of bytes, the largest single group, running 15% below the ceiling) and the
6144/10240 shapes.** It is neither the 24% originally claimed nor the 0% ADR 0006 claimed.
The original 24% was wrong because it used the wrong ceiling; ADR 0006 was wrong because it
sampled one shape and generalised to the model.

### 2. "Thinking on means zero speculation" is false

ADR 0003 states speculation produces zero drafts while the model is thinking. The project's
own served-path data contradicts it:

| task | mode | drafts | accepted | thinking chars |
|---|---|---:|---:|---:|
| mechanical | medium | **96** | 51 | 1109 |
| mechanical | low | **144** | 61 | 953 |
| reasoning | medium | **110** | 19 | 824 |

Drafts fire during the answer, after the thinking trace, whenever the answer quotes the
context. The original measurement used short prompts where thinking consumed the entire
output budget, so nothing was left to draft from. The correct rule is: **speculation needs
context reuse in the generated span, and thinking can prevent or delay that, but does not
forbid it.** The blanket dismissal of `low` in the same ADR is also withdrawn: `low` drafted
144 times with 61 accepted on the mechanical task.

This does not establish a large whole-request benefit - the token rates in that table are
27.8-33.3 t/s throughout - but the mechanism claim was wrong.

### 3. `parity-check.sh` could PASS on divergent output

A real defect, confirmed by reading the script: it printed the greedy-token comparison but
the PASS condition tested only the perplexity delta. Perplexity is a mean and can hide
token-level divergence, which is precisely the failure a kernel change introduces. **Fixed**:
both conditions are now required, and the failure message names which one failed. Re-verified
on the current build (perplexity 6.1885 both sides, greedy tokens identical, PASS).

### 4. The prefill roofline charges the wrong operation count

The 483 t/s bound multiplies all 26.9B parameters by every prompt token. The review checked
the local GGUF and found the embedding and output tensors hold 1.271B weights each,
**2.543B = 9.5%** (verified here). The embedding is a row lookup, not a per-token matmul, and
the output head is computed for the last token of an ordinary batch. Removing both raises the
`dp4a` bound to roughly **534 t/s**, so the measured 493.8 t/s is about **92%** of the bound,
not 102% of a hard ceiling.

As the review itself notes, this does **not** establish a recoverable 8%: attention, the
norms, and imperfect `dp4a` utilisation consume the remainder. What it invalidates is the
"measured above the ceiling, therefore closed" reasoning.

### 5. "llama-bench is a lower bound" is wrong

`docs/gemv-benchmark-artifact.md` concluded from per-kernel bandwidth that `llama-bench tg`
understates served throughput. End to end it does not: the `llama-bench` baseline is
**29.55 t/s** and the served-shape CLI baseline is **28.8 t/s** - the benchmark is *higher*.
Per-kernel bandwidth cannot establish end-to-end ordering, since the two shapes differ in
non-GEMV work as well. The shape diagnosis stands; the corollary is withdrawn.

## Accepted with a caveat

### 6. The decisive counters are not in the committed evidence trail

`.gitignore` excludes `results/ncu-*.csv`, so the six-launch CSV that ADR 0006 rests on is a
transcription in a document, not a committed measurement. **Fixed**: the ncu CSVs are now
tracked.

The review also notes the methodological mismatch: the 275 GB/s reference comes from an
ordinary CUDA run of 200,000 rows, while the GEMV figure comes from Nsight Compute, which
replays kernels, flushes caches between passes, and can hold clocks. So `277.7 / 275 = 101%`
is not a precise closure test. This is fair and is recorded in ADR 0006's replacement text.
It cuts both ways: the same caveat applies to the corrected 84-89% figures above.

### 7. Other method caveats accepted

- The 82.6% GEMV share comes from a **truncated** nsys capture that also used the
  subsequently-discredited `llama-bench` shape. It is a valid ranking for that capture, not a
  measured served-token budget. `docs/decode-profile.md` now says so.
- The cached-turn **2.9 s** in `docs/rt3060-profile.md` combines prompt timing from a
  short non-copying reply with decode timing from a separate copying task. It is a
  projection, not an observed turn, and is labelled as such now.
- `scripts/quality-check.py`'s two format scorers check line counts but not the requested
  content. The arithmetic failure under `none` is real, but "5/6 vs 6/6" does not establish
  general safety for mechanical work. The doc already carried a scope caveat; the scorers are
  now described accurately.

## The one genuinely untested optimization

**Prefix-cache preservation across real file-edit turns.** The cache experiment appended two
short questions to an *unchanged* 21K-token context. It never edited text inside that
context. If an agent rewrites a file near the *start* of its prompt, the cache is invalidated
from that point and much of the 47.7 s prefill returns on every turn.

The proposed mitigation is a prompt-layout rule, not a kernel change: keep stable material
first and append the current file state or a diff last, so earlier tokens stay cacheable.
Conditional upper bound: ~47 s per otherwise-invalidated 21K turn.

This is the next experiment, and the review is right that it is untested - including the
edit-correctness side, and what to do with superseded file versions retained in context.

## Summary

| item | author's position | after review |
|---|---|---|
| GEMV headroom | 0% (ADR 0006) | **8-13% of decode**, in the 5120/6144/10240 shapes |
| prefill bound | 483 t/s hard ceiling, at 102% | ~534 t/s corrected, at ~92% |
| thinking vs speculation | zero drafts while thinking | **drafts do fire** during the answer |
| llama-bench vs served | benchmark understates | benchmark is slightly higher end to end |
| parity gate | enforced numerics | **did not** enforce tokens; fixed |
| ncu evidence | gitignored | tracked |
| prefix cache under edits | untested | **the remaining opportunity** |
