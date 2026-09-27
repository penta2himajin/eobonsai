#!/usr/bin/env bash
# Measure n-gram speculative decoding against a plain baseline on this machine.
#
# The fork implements draft-model-free speculation (`--spec-type ngram-*`), which is
# the only lever that changes the decode roofline: it verifies several tokens per
# weight pass instead of one. How much it helps depends entirely on how much the
# output overlaps the context, so the prompt file matters more than the flags.
#
# Usage:
#   scripts/specbench.sh <prompt-file> [spec-type...]      # default: none ngram-mod
#
# Env:
#   N=300        tokens to generate
#   REASONING=off|on|auto   thinking mode (off is what makes copy tasks speculative)
#
# Writes one log per configuration under results/ and prints a comparison table.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/rt3060.env"

PROMPT_FILE="${1:?usage: specbench.sh <prompt-file> [spec-type...]}"
shift || true
TYPES=("$@")
[[ ${#TYPES[@]} -eq 0 ]] && TYPES=(none ngram-mod)

N="${N:-300}"
REASONING="${REASONING:-off}"
MODEL="${MODEL:-$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-${PACK:-PQ2_0}.gguf}"
BIN="${BIN_DIR:-$ROOT/bin/cuda}/llama-cli"
STAMP="$(date +%Y%m%d-%H%M%S)"
TAG="$(basename "$PROMPT_FILE" .txt)"

mkdir -p "$ROOT/results"
printf '%-14s %8s %8s %7s %7s %6s %6s\n' spec-type gen_t/s speedup runs drafts acc 'accept%' >&2

BASE=""
for st in "${TYPES[@]}"; do
  LOG="$ROOT/results/spec-${TAG}-${st}-${STAMP}.log"
  "$BIN" -m "$MODEL" -ngl 99 -fa on -st --no-warmup \
    -n "$N" --temp 0 -f "$PROMPT_FILE" --spec-type "$st" -rea "$REASONING" --verbose \
    > "$LOG" 2>&1 || true

  # llama-cli with --verbose prints a perf block and, when speculation ran, a stats line.
  # The perf line reads: "eval time = X ms / N tokens ( T ms per token, R tokens per second)".
  # "prompt eval time" also contains the substring, so the last match is the generation one.
  perf="$(grep -E "eval time =" "$LOG" | tail -1 || true)"
  gts="$(printf '%s' "$perf" | sed -nE 's/.*, *([0-9.]+) tokens per second.*/\1/p')"
  runs="$(printf '%s' "$perf" | grep -oE '/ *[0-9]+ (runs|tokens)' | grep -oE '[0-9]+' | head -1 || true)"
  drafts=$(grep -oE "#gen drafts = *[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+" || true)
  acc=$(grep -oE "#acc drafts = *[0-9]+" "$LOG" | tail -1 | grep -oE "[0-9]+" || true)

  gts="${gts:-0}"; runs="${runs:-0}"; drafts="${drafts:-0}"; acc="${acc:-0}"
  if [[ "$st" == "none" || -z "$BASE" ]]; then BASE="$gts"; fi
  sp="$(python3 -c "b=$BASE; g=$gts; print(f'{g/b:.2f}x' if b else 'n/a')" 2>/dev/null || echo n/a)"
  pct="$(python3 -c "print(f'{100*$acc/$drafts:.0f}' if $drafts else '-')" 2>/dev/null || echo -)"
  printf '%-14s %8s %8s %7s %7s %6s %6s\n' "$st" "$gts" "$sp" "$runs" "$drafts" "$acc" "$pct" >&2
done

echo >&2
echo "logs: results/spec-${TAG}-*-${STAMP}.log" >&2
