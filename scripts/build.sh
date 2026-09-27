#!/usr/bin/env bash
# Build the PrismML llama.cpp fork for this machine (RTX 3060, sm_86, CUDA 12.4).
#
# A source build is required for two reasons:
#   1. kernel work needs a tree we can modify,
#   2. the baseline for A/B must come from the same toolchain as the optimized build.
#
# The fork tree is pinned to the commit that the pinned release binary was built from,
# so source and prebuilt results are comparable.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
BUILD="$ROOT/build/llama-sm86"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.4}"
FORK_COMMIT="adfffbe41b2cabcd51fff326ab045662265062bb"

log() { printf '[build] %s\n' "$*" >&2; }

[[ -d "$SRC/.git" ]] || { log "fork missing; run scripts/setup.sh then clone"; exit 1; }

if [[ -d "$SRC/.git" ]] && ! git -C "$SRC" cat-file -e "$FORK_COMMIT^{commit}" 2>/dev/null; then
  log "fetching pinned commit $FORK_COMMIT"
  git -C "$SRC" fetch --depth 1 origin "$FORK_COMMIT"
fi
current="$(git -C "$SRC" rev-parse HEAD)"
if [[ "$current" != "$FORK_COMMIT" ]]; then
  log "checking out $FORK_COMMIT (was ${current:0:9})"
  git -C "$SRC" checkout --detach "$FORK_COMMIT"
fi

# Apply the local patch overlay. The checkout stays disposable: patches/ is the source
# of truth, so a fresh clone plus this loop reproduces the tuned build exactly.
for p in "$ROOT"/patches/*.patch; do
  [[ -e "$p" ]] || continue
  if git -C "$SRC" apply --reverse --check "$p" 2>/dev/null; then
    log "patch already applied: $(basename "$p")"
  elif git -C "$SRC" apply "$p"; then
    log "applied patch: $(basename "$p")"
  else
    log "ERROR: cannot apply $(basename "$p") - the pinned commit may have changed"
    exit 1
  fi
done

log "CUDA: $CUDA_ROOT"
log "configure -> $BUILD"
cmake -S "$SRC" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" \
  -DCUDAToolkit_ROOT="$CUDA_ROOT" \
  -DLLAMA_CURL=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

log "building (this takes a while)"
# llama-perplexity is part of the verification gate (scripts/parity-check.sh), so it is
# built alongside the benchmark and serving binaries.
cmake --build "$BUILD" --target llama-bench llama-cli llama-server llama-perplexity -j"$(nproc)"

log "binaries: $BUILD/bin"
ls -1 "$BUILD/bin" | head -20
log "done"
