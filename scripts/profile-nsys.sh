#!/usr/bin/env bash
# Per-kernel time breakdown of one decode step, using Nsight Systems.
#
# This is the root-free alternative to scripts/profile-ncu.sh. ncu needs the
# hardware-counter permission (RmProfilingAdminOnly = 1 on this machine, so root),
# but nsys reads CUPTI activity records instead, which an ordinary user may do.
# Kernel durations are all we need to find where the unaccounted decode time goes.
#
# Install once (the only step that needs root):
#   sudo apt-get install -y cuda-nsight-systems-12-4
#
# Then, as a normal user:
#   scripts/profile-nsys.sh
#
# N_GEN defaults to 6: enough to cover the prefill plus several decode steps. The
# report is per-kernel totals, so a few tokens are plenty.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="${BIN_DIR:-$ROOT/bin/cuda}"
MODEL="${MODEL:-$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
PACK_LABEL="$(basename "$MODEL" .gguf)"
PROMPT="${PROMPT:-The capital of France is}"
N_GEN="${N_GEN:-6}"

NSYS="${NSYS:-}"
if [[ -z "$NSYS" ]]; then
  for c in nsys /usr/local/cuda-12.4/bin/nsys /opt/nvidia/nsight-systems/*/bin/nsys; do
    [[ -x "$c" ]] && { NSYS="$c"; break; }
  done
fi
[[ -n "$NSYS" ]] || {
  echo "nsys not found. Install it once with:" >&2
  echo "  sudo apt-get install -y cuda-nsight-systems-12-4" >&2
  exit 1
}
[[ -f "$MODEL" ]] || { echo "model not found: $MODEL" >&2; exit 1; }

STAMP="$(date +%Y%m%d-%H%M%S)"
BASE="$ROOT/results/nsys-${PACK_LABEL}-${STAMP}"
mkdir -p "$ROOT/results"

echo "### phase 1: real timing (no profiler)" >&2
"$BIN_DIR/llama-cli" -m "$MODEL" -ngl 99 -fa on -st --no-warmup -n "$N_GEN" -p "$PROMPT" \
  < /dev/null 2>&1 | grep -E "prompt eval time|eval time|total time" | tail -4 >&2

echo >&2
echo "### phase 2: nsys kernel trace" >&2
# --sample=none and --cpuctxsw=none keep the report to CUDA activity only.
"$NSYS" profile \
  --trace=cuda \
  --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  -o "$BASE" \
  "$BIN_DIR/llama-cli" -m "$MODEL" -ngl 99 -fa on -st --no-warmup -n "$N_GEN" -p "$PROMPT" \
  < /dev/null > "$BASE.run.log" 2>&1 || true

echo "### stats" >&2
"$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-export=true \
  "$BASE.nsys-rep" > "$BASE.kernels.csv" 2>/dev/null || {
    echo "nsys stats failed; inspect $BASE.nsys-rep" >&2; exit 1; }

python3 - "$BASE.kernels.csv" <<'PY' >&2 || echo "csv parse failed: $BASE.kernels.csv" >&2
import csv, sys, collections
tot, cnt = collections.Counter(), collections.Counter()
with open(sys.argv[1], newline="") as fh:
    for row in csv.reader(fh):
        if len(row) < 2:
            continue
        name = row[0].strip().strip('"')
        # Columns vary by nsys version; find the first parseable numeric field that
        # looks like a total time in ns.
        vals = []
        for cell in row[1:]:
            try:
                vals.append(float(cell.replace(",", "").strip().strip('"')))
            except ValueError:
                vals.append(None)
        nums = [v for v in vals if v is not None]
        if not nums:
            continue
        ns_total = max(nums)   # the widest time column is the total
        tot[name] += ns_total
        cnt[name] += 1
if not tot:
    print("no kernel rows parsed; open the csv")
    raise SystemExit(1)
s = sum(tot.values())
print(f"\nGPU kernel time: {s/1e6:.1f} ms across {sum(cnt.values())} launches "
      f"(under nsys, so wall time is not meaningful)")
print(f"{'ms':>9} {'%':>6} {'launches':>9}  kernel")
for k, v in tot.most_common(30):
    nm = k if len(k) < 96 else k[:93] + "..."
    print(f"{v/1e6:9.2f} {100*v/s:6.2f} {cnt[k]:9d}  {nm}")
PY

echo >&2
echo "artifacts: $BASE.nsys-rep  $BASE.kernels.csv  $BASE.run.log" >&2
