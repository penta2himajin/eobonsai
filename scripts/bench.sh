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

# Sample the SM clock for the whole run. The clock is not fixed: it idles near 1837 MHz
# but boosts to ~1935-1965 MHz under a compute-bound int8 phase, and the dp4a ceiling
# scales with it, so a pp figure without the clock it ran at is not comparable across
# sessions. See results/pp512-clock-attribution-20260928.txt.
CLK_SAMPLES="$(mktemp)"
if command -v nvidia-smi >/dev/null 2>&1; then
  ( while true; do
      nvidia-smi --query-gpu=clocks.sm,power.draw,utilization.gpu --format=csv,noheader
      sleep 0.25
    done ) > "$CLK_SAMPLES" 2>/dev/null &
  CLK_PID=$!
  trap 'kill "$CLK_PID" 2>/dev/null; rm -f "$CLK_SAMPLES"' EXIT
fi

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
# CLI: the served shape. llama-bench batches tokens, so it runs MMVQ with ncols_dst=4
#      (rows_per_block=2, grid halved, ~117 GB/s); a server or CLI generating one token
#      at a time runs ncols_dst=1 (grid = true row count, ~229 GB/s). The two shapes are
#      2x apart, so llama-bench tg is a lower bound on served throughput, not a proxy.
#      See docs/gemv-benchmark-artifact.md.
# Select with PHASES=pp, PHASES=tg, PHASES=cli, or a comma list; default runs pp,tg.
PHASES="${PHASES:-pp,tg}"
[[ "$PHASES" == *pp* ]] && run_phase "PP (prompt processing)" -p "${PP_LIST:-128,512,2048,8192}" -n 0 -d 0 "$@"
[[ "$PHASES" == *tg* ]] && run_phase "TG (token generation)" -p 0 -n 128 -d "${TG_DEPTHS:-0,8192,32768}" "$@"

if [[ "$PHASES" == *cli* ]]; then
  CLI_BIN="${BIN_DIR%/bin}/bin/llama-cli"
  [[ -x "$CLI_BIN" ]] || CLI_BIN="$ROOT/$BIN_DIR/llama-cli"
  echo "" | tee -a "$OUT"
  echo "### CLI (served shape: one token at a time, ncols_dst=1)" | tee -a "$OUT"
  for i in 1 2 3; do
    echo "\$ llama-cli -n ${N:-200} --temp 0 --spec-type none -p <fixed prompt>   (run $i)" | tee -a "$OUT"
    "$CLI_BIN" -m "$MODEL" -ngl 99 -fa on -st --no-warmup -n "${N:-200}" --temp 0 \
      -p "${SERVED_PROMPT:-Count from 1 to 40 slowly.}" --spec-type "${SPEC_TYPE:-none}" -rea off \
      < /dev/null 2>&1 | grep -E "Generation:|Prompt:" | tee -a "$OUT"
  done
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  kill "$CLK_PID" 2>/dev/null || true
  sleep 0.3
  echo "" | tee -a "$OUT"
  echo "### clocks while the phases ran (0.25 s samples)" | tee -a "$OUT"
  python3 - "$CLK_SAMPLES" <<'PY' | tee -a "$OUT"
import sys, statistics
rows = []
for line in open(sys.argv[1]):
    p = [x.strip() for x in line.strip().split(',')]
    if len(p) != 3 or not p[0][:1].isdigit():
        continue
    n = lambda s: float(''.join(c for c in s if c.isdigit() or c == '.'))
    rows.append((n(p[0]), n(p[2])))
if not rows:
    print("no samples")
    raise SystemExit(0)
sm = [r[0] for r in rows]
busy = [r[0] for r in rows if r[1] >= 50]
print(f"samples     : {len(rows)}")
print(f"SM clock    : min {min(sm):.0f}  median {statistics.median(sm):.0f}  max {max(sm):.0f} MHz")
if busy:
    print(f"SM clock    : median {statistics.median(busy):.0f} MHz while util >= 50% (n={len(busy)})")
else:
    print("SM clock    : no sample caught util >= 50%; use the median above with care")
# samples are in MHz; dp4a peak = 28 SM x 64 INT32 lanes x 4 MAC x f, against 23.8492 G MAC/token
f_mhz = statistics.median(busy or sm)
print(f"dp4a ceiling at {f_mhz:.0f} MHz, PQ2_0: {28*64*4*f_mhz*1e6/23.8492e9:.1f} t/s")
PY
fi

echo
echo "saved: $OUT"
