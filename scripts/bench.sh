#!/usr/bin/env bash
# Baseline / A-B benchmark for Bonsai 2 27B on the RTX 3060.
#
# PP (prompt processing) and TG (token generation) are run as separate phases:
# llama-bench crosses -p with -d, and a pp8192 x depth-16384 cell costs a lot of
# KV for no information. The phases stress different limits anyway:
#   PP -> dp4a throughput (measured ceiling: ~25 TOPS, see scripts/roofline.sh)
#   TG -> memory bandwidth (measured read ceiling: ~307 GB/s) plus KV traffic at depth
#
# Launch overhead on this card is 1.4-2.9 us per launch and moves with CPU load,
# so run this on a quiet machine and check the load average printed in the header.
#
# Usage:
#   scripts/bench.sh <bin-dir> PQ2_0 [extra llama-bench flags...]
#   KV_TYPE=q4_0 scripts/bench.sh build/llama-sm86/bin PQ2_0
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${1:?usage: bench.sh <bin-dir> <PQ2_0|PTQ1_0> [flags...]}"
PACK="${2:?usage: bench.sh <bin-dir> <PQ2_0|PTQ1_0> [flags...]}"
shift 2

MODEL="$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-${PACK}.gguf"
BENCH="$ROOT/$BIN_DIR/llama-bench"
TAG="$(echo "$BIN_DIR" | tr '/' '_')"
KV_TYPE="${KV_TYPE:-f16}"
# K and V can be quantized independently; V quantization is the more disruptive of
# the two, so KV_TYPE_K=q4_0 KV_TYPE_V=f16 is worth measuring on its own.
KV_TYPE_K="${KV_TYPE_K:-$KV_TYPE}"
KV_TYPE_V="${KV_TYPE_V:-$KV_TYPE}"
REPS="${REPS:-5}"

[[ -x "$BENCH" ]] || { echo "llama-bench not found: $BENCH" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

KV_ARGS=()
if [[ "$KV_TYPE_K" != "f16" || "$KV_TYPE_V" != "f16" ]]; then
  KV_ARGS=(-ctk "$KV_TYPE_K" -ctv "$KV_TYPE_V")
fi

OUT="$ROOT/results/bench-${TAG##*_}-${PACK}-k${KV_TYPE_K}v${KV_TYPE_V}-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$ROOT/results"

{
  echo "### provenance"
  echo "binary_dir : $BIN_DIR"
  echo "model      : $(basename "$MODEL")  ($(stat -c %s "$MODEL") bytes)"
  echo "fork_commit: $(git -C "$ROOT/third_party/llama.cpp" rev-parse HEAD 2>/dev/null || echo n/a)"
  echo "kv_type    : k=$KV_TYPE_K v=$KV_TYPE_V    reps: $REPS"
  echo "date       : $(date -Iseconds)"
  echo "loadavg    : $(cat /proc/loadavg)   <- must be idle for a fair comparison"
  echo "### gpu"
  nvidia-smi --query-gpu=name,memory.used,memory.total,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit \
             --format=csv 2>&1
} | tee "$OUT"

run_phase() {
  local label="$1"; shift
  echo "" | tee -a "$OUT"
  echo "### $label" | tee -a "$OUT"
  echo "\$ llama-bench -m <model> -ngl 99 -fa on ${KV_ARGS[*]} -r $REPS $*" | tee -a "$OUT"
  # shellcheck disable=SC2086
  "$BENCH" -m "$MODEL" -ngl 99 -fa on "${KV_ARGS[@]}" -r "$REPS" "$@" 2>&1 | tee -a "$OUT"
}

# PP: how fast a fresh prompt is ingested (dp4a-bound).
# TG: how fast tokens come out, measured with a live KV cache.
# Select with PHASES=pp or PHASES=tg; default runs both.
PHASES="${PHASES:-pp,tg}"
[[ "$PHASES" == *pp* ]] && run_phase "PP (prompt processing)" -p "${PP_LIST:-128,512,2048,8192}" -n 0 -d 0 "$@"
[[ "$PHASES" == *tg* ]] && run_phase "TG (token generation)" -p 0 -n 128 -d "${TG_DEPTHS:-0,8192,32768}" "$@"

echo
echo "saved: $OUT"
