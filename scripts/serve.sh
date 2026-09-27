#!/usr/bin/env bash
# Start, stop, or inspect the local Bonsai 2 27B server.
#
# Wraps llama-server with the profile in config/rt3060.env and exposes its
# OpenAI-compatible API on loopback. Every profile value can be overridden per run:
#
#   scripts/serve.sh                       # start with config/rt3060.env
#   CTX=65536 KV_TYPE=q4_0 scripts/serve.sh
#   scripts/serve.sh status
#   scripts/serve.sh stop
#
# The effective flag line is written to the log, so a result can always be traced
# back to the exact invocation that produced it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../config/rt3060.env
# The profile only assigns values that are not already set, so a per-run override
# works: CTX=65536 scripts/serve.sh
source "$ROOT/config/rt3060.env"

BIN_DIR="${BIN_DIR:-$ROOT/bin/cuda}"
BIN="$BIN_DIR/llama-server"
MODEL="$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-${PACK}.gguf"
PIDFILE="$ROOT/out/llama-server.pid"
LOGDIR="$ROOT/results"
mkdir -p "$ROOT/out" "$LOGDIR"

usage() { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

build_args() {
  ARGS=(-m "$MODEL" -ngl "$NGL" -fa "$FA" -ub "$UB" -b "$BATCH" -t "$THREADS"
        -c "$CTX" -np "$NP" --host "$HOST" --port "$PORT" --metrics --jinja
        --chat-template-kwargs "{\"reasoning_effort\":\"$REASONING_EFFORT\"}"
        --reasoning-budget "$REASONING_BUDGET")
  if [[ "$KV_TYPE" != "f16" ]]; then
    ARGS+=(-ctk "$KV_TYPE" -ctv "$KV_TYPE")
  fi
  if [[ "$MMPROJ" == "1" ]]; then
    ARGS+=(--mmproj "$ROOT/models/bonsai2/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf")
    [[ "$MMPROJ_CPU" == "1" ]] && ARGS+=(--no-mmproj-offload)
  fi
}

case "${1:-start}" in
  start)
    [[ -x "$BIN" ]]   || { echo "llama-server not found: $BIN (run scripts/setup.sh)" >&2; exit 1; }
    [[ -f "$MODEL" ]] || { echo "model not found: $MODEL (run scripts/setup.sh)" >&2; exit 1; }
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "already running (pid $(cat "$PIDFILE")); use 'scripts/serve.sh stop'" >&2; exit 1
    fi

    build_args
    LOG="$LOGDIR/serve-${PACK}-kv${KV_TYPE}-ctx${CTX}-$(date +%Y%m%d-%H%M%S).log"
    {
      echo "### llama-server"
      echo "date    : $(date -Iseconds)"
      echo "loadavg : $(cat /proc/loadavg)"
      nvidia-smi --query-gpu=memory.used,memory.total,clocks.sm,temperature.gpu,power.draw --format=csv,noheader
      echo "flags   : ${ARGS[*]}"
    } > "$LOG"

    nohup "$BIN" "${ARGS[@]}" >> "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "starting pid $(cat "$PIDFILE") -> $LOG"

    for _ in $(seq 1 120); do
      if curl -sf "http://$HOST:$PORT/health" >/dev/null 2>&1; then
        echo "ready: http://$HOST:$PORT  (OpenAI-compatible: /v1/chat/completions, /v1/models)"
        echo "log  : $LOG"
        exit 0
      fi
      kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo "server exited; tail of $LOG:" >&2; tail -20 "$LOG" >&2; exit 1; }
      sleep 2
    done
    echo "timed out waiting for /health; see $LOG" >&2
    exit 1
    ;;
  stop)
    [[ -f "$PIDFILE" ]] || { echo "no pidfile; nothing to stop" >&2; exit 0; }
    PID="$(cat "$PIDFILE")"
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID"
      for _ in $(seq 1 30); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
      kill -0 "$PID" 2>/dev/null && kill -9 "$PID"
      echo "stopped pid $PID"
    else
      echo "pid $PID not running"
    fi
    rm -f "$PIDFILE"
    ;;
  status)
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "running pid $(cat "$PIDFILE")"
      curl -sf "http://$HOST:$PORT/props" | python3 -c \
        'import json,sys; d=json.load(sys.stdin); print("n_ctx:", d.get("default_generation_settings",{}).get("n_ctx")); print("model:", d.get("model_path"))' \
        2>/dev/null || true
      nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
      curl -sf "http://$HOST:$PORT/metrics" | grep -E "^llamacpp:(prompt|tokens|requests)" | head -10 || true
    else
      echo "not running"
    fi
    ;;
  *) usage ;;
esac
