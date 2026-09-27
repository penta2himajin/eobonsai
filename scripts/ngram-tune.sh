#!/usr/bin/env bash
# Sweep the n-gram speculation parameters on a fixed prompt.
#
# n-gram speculation is the only mechanism that raises decode throughput above the DRAM
# roofline, because it buys several tokens per weight pass. Its defaults (lookup n-gram 12,
# draft length 48) were not tuned for this model or these prompts, and the two knobs trade
# off against each other: a shorter lookup matches more often but proposes more junk, and a
# longer draft pays more when a proposal is rejected.
#
# Usage:
#   scripts/ngram-tune.sh [prompt-file] [N]     # default fixtures/prompts/code-edit.txt
#
# Env:
#   SPEC_BASE=ngram-simple   which n-gram variant to tune
#   BIN_DIR=build/llama-sm86/bin
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/rt3060.env"

PROMPT="${1:-$ROOT/fixtures/prompts/code-edit.txt}"
N="${2:-256}"
SPEC_BASE="${SPEC_BASE:-ngram-simple}"
BIN="${BIN_DIR:-$ROOT/bin/cuda}/llama-cli"
MODEL="${MODEL:-$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-${PACK:-PQ2_0}.gguf}"

[[ -f "$PROMPT" ]] || { echo "prompt not found: $PROMPT" >&2; exit 1; }
[[ -x "$BIN" ]]    || { echo "llama-cli not found: $BIN" >&2; exit 1; }

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/results/ngram-tune-$STAMP.txt"

# size_n size_m min_hits label
CONFIGS=(
  "12 48  1 default"
  "12 48  2 min-hits=2"
  "6  48  1 lookup=6"
  "24 48  1 lookup=24"
  "12 24  1 draft=24"
  "12 96  1 draft=96"
  "12 128 1 draft=128"
  "6  96  1 lookup=6 draft=96"
  "6  128 1 lookup=6 draft=128"
  "24 96  1 lookup=24 draft=96"
)

{
  echo "# n-gram speculation parameter sweep"
  echo "# date: $(date -Iseconds)"
  echo "# prompt: ${PROMPT#$ROOT/}   n=$N   temp=0   spec=$SPEC_BASE"
  echo "# model: $(basename "$MODEL")"
  echo "# loadavg: $(cat /proc/loadavg)"
  echo
  printf '%-26s %8s %7s %8s %9s %10s\n' config gen_t/s wall_s tokens drafts accepted
} | tee "$OUT"

# Baseline with speculation off, for the speedup column.
base_log="$ROOT/out/_ngram_base.log"
"$BIN" -m "$MODEL" -ngl 99 -fa on -st --no-warmup -n "$N" --temp 0 -f "$PROMPT" \
  --spec-type none -rea off > "$base_log" 2>&1 || true
BASE="$(grep -oE "Generation: [0-9.]+" "$base_log" | tail -1 | grep -oE "[0-9.]+" || echo 0)"
printf '%-26s %8s %7s %8s %9s %10s\n' "no-speculation (base)" "$BASE" - - - - | tee -a "$OUT"

for cfg in "${CONFIGS[@]}"; do
  read -r sn sm mh label <<<"$cfg"
  log="$ROOT/out/_ngram_${sn}_${sm}_${mh}.log"
  timeout 900 "$BIN" -m "$MODEL" -ngl 99 -fa on -st --no-warmup -n "$N" --temp 0 -f "$PROMPT" \
    --spec-type "$SPEC_BASE" \
    --spec-ngram-simple-size-n "$sn" --spec-ngram-simple-size-m "$sm" \
    --spec-ngram-simple-min-hits "$mh" -rea off --verbose > "$log" 2>&1 || true

  perf="$(grep -E "eval time =" "$log" | tail -1 || true)"
  gts="$(printf '%s' "$perf" | sed -nE 's/.*, *([0-9.]+) tokens per second.*/\1/p')"
  runs="$(printf '%s' "$perf" | grep -oE '/ *[0-9]+ (runs|tokens)' | grep -oE '[0-9]+' | head -1 || true)"
  drafts="$(grep -oE "#gen drafts = *[0-9]+" "$log" | tail -1 | grep -oE "[0-9]+" || true)"
  acc="$(grep -oE "#acc drafts = *[0-9]+" "$log" | tail -1 | grep -oE "[0-9]+" || true)"
  wall="$(grep -oE "Generation: [0-9.]+" "$log" | tail -1 | grep -oE "[0-9.]+" || true)"

  printf '%-26s %8s %7s %8s %9s %10s\n' \
    "$label" "${gts:-ERR}" "${wall:--}" "${runs:-0}" "${drafts:-0}" "${acc:-0}" | tee -a "$OUT"
done

{
  echo
  echo "# base is --spec-type none on the same prompt, so gen_t/s divided by base is the"
  echo "# speedup. drafts and accepted come from the fork's own speculation statistics."
} | tee -a "$OUT"

echo
echo "saved: $OUT"
