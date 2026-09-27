#!/usr/bin/env bash
# Reasoning-mode comparison against a running server.
#
# Speculative decoding only fires when thinking is off, and the difference measured here is
# ~5x, so the reasoning mode dominates every other performance choice in this project. This
# script measures the trade honestly: the same prompts, several modes, with time, token
# count and draft statistics for each.
#
# Usage:
#   scripts/serve.sh start
#   scripts/reasoning-ab.sh                    # built-in prompt set
#   scripts/reasoning-ab.sh my-prompts.txt     # "=== name ===" delimited blocks
#
# It measures the served path, not llama-cli, because that is what clients use.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/rt3060.env"

PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/results/reasoning-ab-$STAMP.txt"
PROMPT_FILE="${1:-}"
mkdir -p "$ROOT/out" "$ROOT/results"

# Prompts are passed to the driver through files, never through shell variables: a
# multi-line prompt in a variable loses its newlines and silently changes the task.
if [[ -z "$PROMPT_FILE" ]]; then
  PROMPT_FILE="$ROOT/out/reasoning-ab-prompts.txt"
  cat > "$PROMPT_FILE" <<'EOF'
=== mechanical ===
Given the note below, output it verbatim with every occurrence of "The" replaced by "THE".
Output only the modified note, with no commentary.
---
The model runs on a single GPU. The KV cache grows with the context. The prefill phase
is compute bound. The decode phase is bandwidth bound. The measured ceiling is the limit.
===
=== reasoning ===
A 12 GB GPU holds a model whose weights take 7.19 GiB, and the KV cache costs 64 KiB per
token. Compute how many tokens of context fit in the remaining memory if compute buffers
take 760 MiB and the card reports 11904 MiB usable. Show the arithmetic, then state the
answer in one line.
===
EOF
fi

cat > "$ROOT/out/_reasoning_ask.py" <<'PY'
import json, sys, time, urllib.request, urllib.error
mode, prompt_path, port = sys.argv[1], sys.argv[2], sys.argv[3]
prompt = open(prompt_path).read().strip()
body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 400, "temperature": 0}
if mode != "server-default":
    body["reasoning_effort"] = mode
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
try:
    d = json.load(urllib.request.urlopen(req, timeout=1800))
except urllib.error.HTTPError as e:
    print(f"{'ERR':>8} {'-':>8} {'-':>6} {'-':>7} {'-':>8} {e.code:>11}")
    raise SystemExit
wall = time.time() - t0
t = d["timings"]
msg = d["choices"][0]["message"]
think = len(msg.get("reasoning_content") or "")
print(f"{t['predicted_per_second']:8.2f} {wall:8.1f} {t['predicted_n']:6d} "
      f"{t.get('draft_n', 0):7d} {t.get('draft_n_accepted', 0):8d} {think:11d}")
PY

# Emit "name<tab>path" pairs, one file per prompt so newlines survive.
python3 - "$PROMPT_FILE" "$ROOT/out" <<'PY' > "$ROOT/out/_reasoning_prompts.tsv"
import re, sys, pathlib
text, outdir = open(sys.argv[1]).read(), pathlib.Path(sys.argv[2])
blocks = [b.strip() for b in re.split(r"^===.*===\s*$", text, flags=re.M) if b.strip()]
names = re.findall(r"^===\s*(.+?)\s*===\s*$", text, flags=re.M)
for i, body in enumerate(blocks):
    name = names[i] if i < len(names) else f"prompt{i+1}"
    path = outdir / f"_reasoning_prompt_{i}.txt"
    path.write_text(body)
    print(f"{name}\t{path}")
PY

if [[ ! -s "$ROOT/out/_reasoning_prompts.tsv" ]]; then
  echo "no prompts parsed from $PROMPT_FILE (expected '=== name ===' delimited blocks)" >&2
  exit 1
fi

{
  echo "# reasoning-mode A/B against the served path"
  echo "# date: $(date -Iseconds)   server: http://$HOST:$PORT"
  echo "# gpu: $(nvidia-smi --query-gpu=name,memory.used,clocks.max.sm --format=csv,noheader)"
  echo "# loadavg: $(cat /proc/loadavg)"
  echo "# prompt file: ${PROMPT_FILE#$ROOT/}"
  echo
  printf '%-12s %-15s %8s %8s %6s %7s %8s %11s\n' \
         prompt mode gen_t/s wall_s tokens drafts accepted think_chars
} | tee "$OUT"

while IFS=$'\t' read -r name path; do
  for mode in server-default none low medium; do
    row="$(python3 "$ROOT/out/_reasoning_ask.py" "$mode" "$path" "$PORT")"
    printf '%-12s %-15s %s\n' "$name" "$mode" "$row" | tee -a "$OUT"
  done
  echo | tee -a "$OUT"
done < "$ROOT/out/_reasoning_prompts.tsv"

cat | tee -a "$OUT" <<'EOF'
# Reading this table:
#   think_chars > 0 -> the model reasoned, drafts are 0, so speculation was inert
#   drafts > 0      -> speculation fired; that is where the ~5x comes from
#   server-default  -> whatever config/rt3060.env passes as the chat-template default
#   none is the only mode measured to disable thinking; low and medium still think
EOF

echo
echo "saved: $OUT"
