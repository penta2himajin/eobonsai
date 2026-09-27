You are reviewing a completed, measurement-driven optimization effort. Your job is to find
what the author missed. Be adversarial about the conclusions, not polite.

Repository: /home/penta/repos/eobonsai (read it; start with docs/rt3060-profile.md,
docs/decisions/0001 through 0006, docs/decode-profile.md, docs/gemv-benchmark-artifact.md,
docs/gemv-roofline-closure.txt, docs/reasoning-mode-trade.md, and scripts/).

## The system

- One GPU: RTX 3060 12 GB, GA106, sm_86, 28 SMs, 1837 MHz sustained, 192-bit GDDR6.
- Model: Bonsai 2 27B (PrismML ternary GGUF, derived from Qwen3.8-27B). PQ2_0 packing,
  7.19 GiB, 1.75-2.13 bits/weight. 64 blocks: 48 gated-delta-net (linear attention) and
  16 full attention. Activation Hadamard rotation, block 1024, applied to 401 weight tensors.
- Runtime: the PrismML llama.cpp fork, pinned commit adfffbe, built from source for sm_86.
  Stock llama.cpp cannot run the packings.
- Use case: a single user doing long-context chat and agentic code editing.

## What was measured (all reproducible from results/)

1. Prefill is at the int8 dp4a roofline. Measured dp4a on this card is 26 TOPS, only 2.2x
   FP32, because GA106 has 64 INT32 lanes per SM against 128 FP32 lanes. PP512 measured
   493.8 t/s against a predicted ceiling of 483 (own dp4a microbenchmark) and 510 (scaling
   the RTX 3070's community number by SM count). Throughput is flat from 512 to 8192 tokens.
2. Decode is 34.45 ms/token and the GPU is saturated (34.45 ms of kernel time against
   34.22 ms wall). 82.6% of it is the ternary GEMV.
3. In the served shape (ncols_dst = 1) the dominant GEMV weight shape runs at 277.7 GB/s,
   which is 101% of the measured row-structured streaming ceiling for that geometry
   (272-275 GB/s, flat above 1 KB rows). L1 hit rate 96-98%, L2 3-14%, occupancy 59-72%,
   56 registers per thread. The inner loop already uses dp4a plus __byte_perm unpacking.
4. CUDA graphs are active and worth 5.3% (measured by toggling GGML_CUDA_DISABLE_GRAPHS).
5. Draft-model-free n-gram speculation (--spec-type ngram-simple) gives 4.9-6.0x when the
   output re-quotes the context (long-document work, code editing, verbatim re-output) and
   is completely inert otherwise. It requires thinking to be off.
6. The prefix cache re-prefills 26 tokens instead of 21,123 on a cached turn, 115x less
   prompt work, so the large prefill is paid once per conversation.
7. reasoning_effort "none" removes thinking tokens. It is much faster on thinking-heavy
   mechanical tasks (1.9 s vs 11.0 s) but scored 5/6 against 6/6 for medium on mechanically
   graded tasks, failing a three-operation arithmetic question with a wrong answer.

Rejected by measurement: the PTQ1_0 packing, 4-bit KV caches (slower at every depth), the
-ub/-b sweep (noise), -np 2, and a GEMV nwarps tuning patch (neutral in the served shape).

## Prior errors in this project, for calibration

The author's predictions were overturned by measurement six times. The recurring failure was
comparing against a reference that was not the applicable one:
(a) treated the Hadamard transform as the top target, measured ~2% of a token;
(b) expected prefill tile tuning to help, found it already at the roofline;
(c) a standalone GEMV microbenchmark predicted +20%, the in-model A/B gave +2.0%;
(d) claimed CUDA graphs were pointless because the GPU was saturated, they are worth 5.3%;
(e) profiled llama-bench's batched shape and mistook it for the served shape;
(f) targeted a "24% GEMV headroom" that came from dividing by the wrong bandwidth ceiling.

## The question

Given the above, and after reading the repository, answer concretely:

1. Is there any remaining optimization for this exact system that the author has not tried?
   Name it, state the mechanism, and state the expected magnitude with your reasoning.
2. Are any of the six "closed" conclusions actually wrong? Say which and why.
3. Is there anything in the measurement methodology that would make the reported numbers
   untrustworthy in a way the author has not noticed?
4. If you conclude nothing material remains, say so plainly rather than inventing work.

Constraints on your answer: be specific and technical, cite the file or measurement you are
relying on, and separate what you verified from what you are inferring. Do not propose a
from-scratch engine rewrite unless you can justify it against the measured roofline numbers.
