#!/usr/bin/env bash
# Build and run the roofline microbenchmarks, saving raw output under results/.
#
# These are the measured denominators for every later claim about the model:
# read bandwidth, launch overhead, dp4a peak, CUDA-core FMA, and FWHT cost.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.4}"
ARCH="${ARCH:-sm_86}"
OUT="$ROOT/results/roofline-$(date +%Y%m%d-%H%M%S).txt"

mkdir -p "$ROOT/out" "$ROOT/results"
"$CUDA_ROOT/bin/nvcc" -O3 -arch="$ARCH" -o "$ROOT/out/roofline" "$ROOT/tools/microbench/roofline.cu"

{
  echo "### roofline microbenchmarks"
  echo "date     : $(date -Iseconds)"
  echo "arch     : $ARCH   cuda: $("$CUDA_ROOT/bin/nvcc" --version | tail -2 | head -1)"
  echo "gpu      : $(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,memory.total --format=csv,noheader)"
  echo
  "$ROOT/out/roofline"
} | tee "$OUT"

echo
echo "saved: $OUT"
