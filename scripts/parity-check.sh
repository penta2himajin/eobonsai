#!/usr/bin/env bash
# Correctness gate: two builds must produce the same numbers on this model.
#
# Any kernel change must pass this before its speedup counts. A faster kernel that
# changes the output is worthless, and on a rotated-basis ternary model a broken
# transform can still produce fluent-looking text, so text comparison alone is not
# enough.
#
# Two checks:
#   1. perplexity on a fixed tracked corpus (sensitive to numerics)
#   2. greedy generation with a fixed seed (catches token-level divergence)
#
# Usage:
#   scripts/parity-check.sh bin/cuda build/llama-sm86/bin
#   scripts/parity-check.sh bin/cuda build/llama-sm86/bin PTQ1_0
#
# Exit status: 0 when both checks agree within tolerance, 1 otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
A_DIR="${1:?usage: parity-check.sh <bin-dir-a> <bin-dir-b> [PQ2_0|PTQ1_0]}"
B_DIR="${2:?usage: parity-check.sh <bin-dir-a> <bin-dir-b> [PQ2_0|PTQ1_0]}"
PACK="${3:-PQ2_0}"

MODEL="$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-${PACK}.gguf"
# A tracked file, so the corpus cannot drift out of the repository.
CORPUS="$ROOT/docs/roofline-rtx3060.md"
COMMON=(-m "$MODEL" -ngl 99 -fa on -c 512)
PROMPT="The capital of France is"
OUTDIR="$ROOT/results/parity-$(date +%Y%m%d-%H%M%S)"

[[ -f "$MODEL" ]]  || { echo "model not found: $MODEL" >&2; exit 1; }
[[ -f "$CORPUS" ]] || { echo "corpus not found: $CORPUS" >&2; exit 1; }
for tool in llama-perplexity llama-cli; do
  for dir in "$A_DIR" "$B_DIR"; do
    [[ -x "$ROOT/$dir/$tool" ]] || { echo "missing $ROOT/$dir/$tool" >&2; exit 1; }
  done
done
mkdir -p "$OUTDIR"

# stdin is closed for every invocation: llama-cli's single-turn mode still reads stdin,
# and a script run without a terminal would otherwise block there forever.
ppl() { # ppl <bin-dir> <tag>
  local dir="$1" tag="$2"
  "$ROOT/$dir/llama-perplexity" "${COMMON[@]}" -f "$CORPUS" < /dev/null 2>&1 |
    tee "$OUTDIR/ppl-$tag.log" |
    grep -E "Final estimate" | tail -1 |
    sed -E 's/.*PPL = *([0-9.]+).*/\1/'
}

greedy() { # greedy <bin-dir> <tag>
  local dir="$1" tag="$2"
  # temperature 0 makes sampling deterministic; the seed fixes the RNG path anyway.
  "$ROOT/$dir/llama-cli" "${COMMON[@]}" -st --no-warmup -n 48 \
    --temp 0 --seed 1234 -p "$PROMPT" < /dev/null > "$OUTDIR/greedy-$tag.log" 2>&1 || true
  # Strip the timing banner and the prompt echo so only generated text is compared.
  sed -n '/^The capital of France is/,$p' "$OUTDIR/greedy-$tag.log" | head -40
}

echo "### parity check: $A_DIR  vs  $B_DIR   pack=$PACK"
echo "corpus: ${CORPUS#$ROOT/}  ($(stat -c %s "$CORPUS") bytes)"
echo

echo "--- perplexity"
PPL_A="$(ppl "$A_DIR" a)"; PPL_B="$(ppl "$B_DIR" b)"
echo "  $A_DIR -> $PPL_A"
echo "  $B_DIR -> $PPL_B"

echo
echo "--- greedy tokens (-n 48, temp 0, seed 1234)"
greedy "$A_DIR" a > "$OUTDIR/greedy-a.txt"
greedy "$B_DIR" b > "$OUTDIR/greedy-b.txt"
if diff -q "$OUTDIR/greedy-a.txt" "$OUTDIR/greedy-b.txt" >/dev/null; then
  echo "  identical"
  TOKENS_OK=1
else
  echo "  DIVERGENT:"
  diff "$OUTDIR/greedy-a.txt" "$OUTDIR/greedy-b.txt" | head -10 | sed 's/^/    /'
  TOKENS_OK=0
fi

echo
# Perplexity is a mean over tokens, so tiny fp differences show as small deltas.
# 0.1% is far below any real numerical breakage, which shows as whole points.
if [[ -z "$PPL_A" || -z "$PPL_B" ]]; then
  echo "FAIL: could not parse perplexity (see $OUTDIR)"
  exit 1
fi
DELTA="$(python3 -c "a,b=$PPL_A,$PPL_B; print(f'{abs(a-b)/a*100:.4f}')")"
echo "ppl delta: ${DELTA}%   tokens: $([[ $TOKENS_OK == 1 ]] && echo match || echo differ)"

# BOTH conditions must hold. Until this was fixed the script tested only the perplexity
# delta even though it printed the token comparison, so divergent greedy output could
# still PASS. Perplexity is a mean and can hide token-level divergence, which is exactly
# the failure mode a kernel change introduces.
PPL_OK=0
python3 -c "import sys; sys.exit(0 if $DELTA < 0.1 else 1)" && PPL_OK=1

if [[ "$PPL_OK" == 1 && "$TOKENS_OK" == 1 ]]; then
  echo "PASS"
  echo "logs: $OUTDIR"
  exit 0
fi
[[ "$PPL_OK" == 0 ]] && echo "FAIL: perplexity delta >= 0.1%"
[[ "$TOKENS_OK" == 0 ]] && echo "FAIL: greedy tokens diverged"
echo "logs: $OUTDIR"
exit 1
