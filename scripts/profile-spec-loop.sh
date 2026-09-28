#!/usr/bin/env bash
# Profile the server's speculative decode loop with Nsight Systems.
#
# scripts/profile-nsys.sh traces llama-cli. The speculative loop that matters for the
# served measurement lives in llama-server (per-slot drafting, verification, accept), so
# this traces the server instead and captures a window that contains exactly one request.
#
# The capture window is wider than the request on purpose: nsys must launch the app before
# anything exists to trace, and idle time contributes no kernels, so the kernel totals
# belong to the request. Compare them against the generation wall time the server itself
# reports in the response timings, not against wall clock under the profiler.
#
# Usage:
#   scripts/profile-spec-loop.sh                          # verbatim-copy, 512 tokens
#   PROMPT_FILE=fixtures/prompts/code-edit.txt MAX_TOKENS=256 scripts/profile-spec-loop.sh
#
# Env:
#   PROMPT_FILE  default fixtures/prompts/verbatim-copy.txt
#   MAX_TOKENS   default 512
#   PORT         default 8080
#
# Root is not needed: nsys reads CUPTI activity records, unlike ncu which needs the
# hardware-counter permission.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NSYS="${NSYS:-/usr/local/bin/nsys}"
BIN="${BIN_DIR:-$ROOT/build/llama-sm86/bin}/llama-server"
MODEL="${MODEL:-$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf}"
PROMPT_FILE="${PROMPT_FILE:-$ROOT/fixtures/prompts/verbatim-copy.txt}"
MAX_TOKENS="${MAX_TOKENS:-512}"
PORT="${PORT:-8080}"
STAMP="$(date +%Y%m%d-%H%M%S)"
BASE="$ROOT/results/nsys-spec-loop-${STAMP}"

[[ -x "$NSYS" ]] || { echo "nsys not found at $NSYS" >&2; exit 1; }
[[ -x "$BIN" ]]  || { echo "server not found: $BIN (run scripts/build.sh)" >&2; exit 1; }

echo "### capturing: $PROMPT_FILE, ${MAX_TOKENS} tokens, port $PORT" >&2
"$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -y 6 -d 30 -o "$BASE" \
  "$BIN" -m "$MODEL" -ngl 99 -fa on -ub 512 -b 2048 -t 8 -c 32768 -np 1 \
  --host 127.0.0.1 --port "$PORT" --metrics --jinja \
  --spec-type ngram-simple --spec-ngram-simple-size-n 6 --spec-ngram-simple-size-m 384 \
  --chat-template-kwargs '{"reasoning_effort":"none"}' --reasoning-budget 0 \
  > "${BASE}.server.log" 2>&1 &
NSYS_PID=$!

for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1 && break
  sleep 1
done

send() { # send <label> <max_tokens> <speculative-json-or-empty>
  python3 - "$1" "$2" "$3" "$PROMPT_FILE" "${BASE}.requests.jsonl" "$PORT" <<'PY'
import json, sys, time, urllib.request
label, mt, spec, path, out, port = (sys.argv[1], int(sys.argv[2]), sys.argv[3],
                                    sys.argv[4], sys.argv[5], sys.argv[6])
prompt = open(path, encoding="utf-8").read()
body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": mt,
        "temperature": 0, "reasoning_effort": "none"}
if spec:
    body["speculative"] = json.loads(spec)
t0 = time.time()
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=900) as r:
    d = json.loads(r.read())
t = d.get("timings", {})
rec = {"label": label, "t_start": t0, "t_end": time.time(),
       "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
       "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
       "predicted_n": t.get("predicted_n"), "predicted_ms": t.get("predicted_ms"),
       "predicted_per_second": t.get("predicted_per_second")}
open(out, "a", encoding="utf-8").write(json.dumps(rec) + "\n")
print(json.dumps(rec))
PY
}

# warm the prefix cache before the capture window opens
send warmup 4 '{"type":"none"}' >/dev/null
sleep 12
echo "### measured request" >&2
send copy "$MAX_TOKENS" ''
sleep 26

pkill -f "$BIN" 2>/dev/null || true
wait "$NSYS_PID" 2>/dev/null || true
sleep 3
echo "### report: ${BASE}.nsys-rep" >&2
echo "### next: nsys stats --report cuda_gpu_kern_sum --format csv ${BASE}.nsys-rep" >&2
