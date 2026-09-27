#!/usr/bin/env bash
# Split one decode step into per-kernel GPU time, using Nsight Compute.
#
# The roofline in docs/roofline-rtx3060.md leaves roughly 5 ms of a ~29 ms decode
# token unaccounted for. This script answers where it goes instead of guessing.
#
# Two phases, and the difference between them is itself a result:
#   1. a normal run, for the real token time (llama-cli reports it)
#   2. an ncu run, for the per-kernel GPU time
# ncu serialises kernel launches, so durations in phase 2 are inflated by roughly
# 100-1000x in wall terms and small-kernel overheads vanish. The sum of phase 2 is
# therefore the GPU work only; real time minus that sum is launch/scheduling cost.
#
# Hardware counters need root on this machine (RmProfilingAdminOnly = 1), so run:
#   sudo scripts/profile-ncu.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-$ROOT/bin/cuda}"
MODEL="${MODEL:-$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
PACK_LABEL="$(basename "$MODEL" .gguf)"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.4}"
NCU="${NCU:-$CUDA_ROOT/bin/ncu}"
PROMPT="${PROMPT:-The capital of France is}"
N_GEN="${N_GEN:-8}"

[[ -x "$NCU" ]] || { echo "ncu not found at $NCU; install cuda-nsight-compute-12-4" >&2; exit 1; }
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

STAMP="$(date +%Y%m%d-%H%M%S)"
BASE="$ROOT/results/ncu-${PACK_LABEL}-${STAMP}"
mkdir -p "$ROOT/results"

run_cli() { # run_cli <logfile>
  "$BIN_DIR/llama-cli" -m "$MODEL" -ngl 99 -fa on -st --no-warmup \
    -n "$N_GEN" -p "$PROMPT" 2>&1 | tee "$1"
}

echo "### phase 1: real timing (no profiler)" >&2
run_cli "$BASE.real.log" | grep -E "prompt eval time|eval time|total time" | tail -6 >&2

echo >&2
echo "### phase 2: per-kernel GPU time (ncu; wall time here is meaningless)" >&2
"$NCU" \
  --target-processes all \
  --kernel-name-base demangled \
  --metrics gpu__time_duration.sum,launch__grid_size,launch__block_size \
  --csv --log-file "$BASE.kernels.csv" \
  "$BIN_DIR/llama-cli" -m "$MODEL" -ngl 99 -fa on -st --no-warmup -n "$N_GEN" -p "$PROMPT" \
  2>&1 | grep -E "ERROR|WARNING|Connected" | head -5 >&2

python3 - "$BASE.kernels.csv" <<'PY' >&2 || echo "csv parse failed; see $BASE.kernels.csv" >&2
import csv, sys, collections
tot, cnt = collections.Counter(), collections.Counter()
try:
    fh = open(sys.argv[1], newline="")
except OSError as e:
    print(f"cannot open {e}"); raise SystemExit(1)
with fh:
    for row in csv.DictReader(fh):
        name = row.get("Kernel Name") or row.get("Function Name") or "?"
        raw = row.get("gpu__time_duration.sum", "")
        try:
            ns = float(raw.replace(",", "").split()[0])
        except (ValueError, IndexError):
            continue
        tot[name] += ns; cnt[name] += 1
if not tot:
    print("no kernel rows; the profile probably failed (counters need root)")
    raise SystemExit(1)
s = sum(tot.values())
print(f"\nprofiled GPU work: {s/1e6:.1f} ms across {sum(cnt.values())} launches")
print(f"{'ms':>9} {'%':>6} {'launches':>9}  kernel")
for k, v in tot.most_common(30):
    print(f"{v/1e6:9.2f} {100*v/s:6.2f} {cnt[k]:9d}  {k[:92]}")
PY

echo >&2
echo "artifacts: $BASE.real.log  $BASE.kernels.csv" >&2
