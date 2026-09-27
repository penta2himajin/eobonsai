#!/usr/bin/env bash
# Fetch the pinned PrismML llama.cpp build and the Bonsai 2 27B GGUF packs.
#
# Everything is pinned: the fork release tag, the tarball SHA-256 (from the GitHub
# release API), and the model repo. Re-running is safe and resumable.
#
# Usage:
#   scripts/setup.sh all            # binaries + every pack (default)
#   scripts/setup.sh binaries       # CUDA binary only
#   scripts/setup.sh model PQ2_0    # one pack only
#   scripts/setup.sh mmproj         # vision projector only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN_DIR="$ROOT/bin/cuda"
MODEL_DIR="$ROOT/models/bonsai2"
mkdir -p "$BIN_DIR" "$MODEL_DIR"

# --- pinned artifacts -------------------------------------------------------
FORK_TAG="prism-b10743-adfffbe"
FORK_URL="https://github.com/PrismML-Eng/llama.cpp/releases/download/${FORK_TAG}"
BIN_ASSET="llama-${FORK_TAG}-bin-linux-cuda-12.4-x64.tar.gz"
BIN_SHA256="fef4c7e8d83ff261d89809c1b302fe5f826adc1a74e13654b67ec056fbc1c639"

MODEL_REPO="prism-ml/Ternary-Bonsai-2-27B-gguf"
MODEL_BASE="https://huggingface.co/${MODEL_REPO}/resolve/main"

# pack -> expected size in bytes (from the HF API; 0 = do not check)
declare -A PACK_SIZE=(
  [PQ2_0]=7206168928
  [PTQ1_0]=5946648928
  [mmproj-Q8_0]=0
)

log() { printf '[setup] %s\n' "$*" >&2; }
die() { printf '[setup] ERROR: %s\n' "$*" >&2; exit 1; }

fetch() { # fetch <url> <dest>
  local url="$1" dest="$2"
  if [[ -s "$dest" ]]; then
    log "exists, skipping: $(basename "$dest")"
    return 0
  fi
  log "downloading $(basename "$dest")"
  curl -fL --retry 5 --retry-delay 5 -C - -o "$dest" "$url"
}

get_binaries() {
  local tar="$BIN_DIR/$BIN_ASSET"
  if [[ ! -x "$BIN_DIR/llama-bench" ]]; then
    fetch "$FORK_URL/$BIN_ASSET" "$tar"
    log "verifying SHA-256"
    echo "$BIN_SHA256  $tar" | sha256sum -c - \
      || die "tarball hash mismatch — refusing to extract"
    log "extracting"
    tar -xzf "$tar" -C "$BIN_DIR" --strip-components=1
    rm -f "$tar"
  fi
  [[ -x "$BIN_DIR/llama-bench" ]] || die "llama-bench missing after extract"
  log "binaries ready: $BIN_DIR"
}

get_pack() { # get_pack <PQ2_0|PTQ1_0>
  local pack="$1"
  local file="Ternary-Bonsai-2-27B-${pack}.gguf"
  local dest="$MODEL_DIR/$file"
  local expect="${PACK_SIZE[$pack]:-0}"
  fetch "$MODEL_BASE/$file" "$dest"
  if [[ "$expect" != 0 ]]; then
    local actual; actual=$(stat -c %s "$dest")
    [[ "$actual" == "$expect" ]] \
      || die "$file size $actual != expected $expect (truncated? re-run to resume)"
  fi
  log "model ready: $dest"
}

get_mmproj() {
  local file="Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf"
  fetch "$MODEL_BASE/$file" "$MODEL_DIR/$file"
  log "vision projector ready (optional; ~0.63 GB)"
}

record_hashes() {
  ( cd "$MODEL_DIR" && sha256sum ./*.gguf > SHA256SUMS 2>/dev/null ) || true
  log "wrote $MODEL_DIR/SHA256SUMS"
}

case "${1:-all}" in
  binaries) get_binaries ;;
  model)    get_pack "${2:?usage: setup.sh model PQ2_0|PTQ1_0}" ; record_hashes ;;
  mmproj)   get_mmproj ; record_hashes ;;
  all)
    get_binaries
    get_pack PQ2_0
    get_pack PTQ1_0
    record_hashes
    ;;
  *) die "unknown target '$1' (all|binaries|model <pack>|mmproj)" ;;
esac
log "done"
