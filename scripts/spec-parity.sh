#!/usr/bin/env bash
# Correctness gate for speculative decoding: greedy output must be unchanged.
#
# n-gram speculation verifies drafted tokens against the target model, so with greedy
# decoding it should be exact rather than approximate. This checks that claim on this
# model, because a 5x speedup that silently alters output is worthless.
#
# Usage: scripts/spec-parity.sh <prompt-file> [spec-type]     # default ngram-simple
#
# Exit status: 0 if the generated text is identical, 1 otherwise.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/rt3060.env"

PROMPT_FILE="${1:?usage: spec-parity.sh <prompt-file> [spec-type]}"
SPEC="${2:-ngram-simple}"
N="${N:-64}"
MODEL="${MODEL:-$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-${PACK:-PQ2_0}.gguf}"
BIN="${BIN_DIR:-$ROOT/bin/cuda}/llama-cli"
TAG="$(basename "$PROMPT_FILE" .txt)"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/results/spec-parity-${TAG}-${STAMP}.txt"

run() { # run <spec-type> <dest>
  "$BIN" -m "$MODEL" -ngl 99 -fa on -st --no-warmup -n "$N" --temp 0 \
    -f "$PROMPT_FILE" --spec-type "$1" -rea off > "$2" 2>&1 || true
}

{
  echo "### spec parity: $SPEC vs none   prompt=$TAG   n=$N   temp=0"
  echo "date: $(date -Iseconds)"
  echo "model: $(basename "$MODEL")"
} | tee "$OUT"

run none "$OUT.none"
run "$SPEC" "$OUT.spec"

python3 - "$OUT.none" "$OUT.spec" >> "$OUT" <<'PY'
import re, sys
def clean(p):
    t = open(p, errors="replace").read().split("Exiting...")[0]
    # Drop banner and timing lines: those are environment, not generated text.
    t = re.sub(r"^(load_backend:|ggml_cuda_init:|  Device 0:|\[ Prompt:).*$", "", t, flags=re.M)
    t = re.sub(r"^\s*$", "", t, flags=re.M)
    return t.strip()
a, b = clean(sys.argv[1]), clean(sys.argv[2])
if a == b:
    print(f"RESULT: PASS - identical generated text ({len(a.splitlines())} lines)")
    sys.exit(0)
la, lb = a.splitlines(), b.splitlines()
diff = sum(1 for x, y in zip(la, lb) if x != y) + abs(len(la) - len(lb))
print(f"RESULT: FAIL - {diff} differing lines (none={len(la)}, {len(lb)})")
for i, (x, y) in enumerate(zip(la, lb)):
    if x != y:
        print(f"  line {i}: {x[:60]!r} != {y[:60]!r}")
sys.exit(1)
PY
STATUS=$?

echo
tail -1 "$OUT"
echo "artifact: $OUT"
exit "$STATUS"
