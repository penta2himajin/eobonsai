#!/usr/bin/env bash
# Sample GPU and server telemetry while the model is serving.
#
# The point is to see throughput and memory under real load, not at the start:
# the counters in /metrics are monotonic, so the useful number is the delta
# between samples.
#
# Usage:
#   scripts/serve.sh start
#   scripts/telemetry.sh                 # 10 samples, 3 s apart, prints a table
#   scripts/telemetry.sh 5 60            # 60 samples, 5 s apart
#
# Output is appended to a CSV under results/ as well as printed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/config/rt3060.env"

INTERVAL="${1:-3}"
COUNT="${2:-10}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"

CSV="$ROOT/results/telemetry-$(date +%Y%m%d-%H%M%S).csv"

metric() { # metric <name>
  curl -s "http://$HOST:$PORT/metrics" 2>/dev/null |
    awk -v k="llamacpp:$1" '$1 == k {print $2; exit}' || echo ""
}

echo "sampling every ${INTERVAL}s x ${COUNT}; server at http://$HOST:$PORT" >&2
echo "time,vram_mib,gpu_util_pct,power_w,temp_c,sm_mhz,pred_tokens,prompt_tokens,gen_tps,prompt_tps" > "$CSV"

prev_tok=""; prev_prompt=""; prev_t=""
printf '%8s %9s %6s %7s %6s %8s %8s %9s\n' time VRAM_MiB GPU% W C tok/s prompt_t/s >&2

for _ in $(seq 1 "$COUNT"); do
  now="$(date +%H:%M:%S)"
  read -r vram util power temp sm <<<"$(nvidia-smi \
      --query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu,clocks.current.sm \
      --format=csv,noheader,nounits | tr -d ',')"

  tok="$(metric tokens_predicted_total)"
  prompt="$(metric prompt_tokens_total)"
  psec="$(metric tokens_predicted_seconds_total)"
  prsec="$(metric prompt_seconds_total)"

  gen_tps=""; prompt_tps=""
  if [[ -n "$prev_tok" && -n "$tok" && "$tok" != "$prev_tok" ]]; then
    gen_tps="$(python3 -c "print(f'{($tok-$prev_tok)/max($psec-${prev_psec:-0},1e-9):.1f}')" 2>/dev/null || echo "")"
  fi
  if [[ -n "$prompt" && -n "$prsec" ]]; then
    prompt_tps="$(python3 -c "print(f'{$prompt/max($prsec,1e-9):.1f}')" 2>/dev/null || echo "")"
  fi

  echo "$now,$vram,$util,$power,$temp,$sm,${tok:-},${prompt:-},${gen_tps:-},${prompt_tps:-}" >> "$CSV"
  printf '%8s %9s %6s %7s %6s %8s %9s\n' "$now" "$vram" "$util" "$power" "$temp" "${gen_tps:--}" "${prompt_tps:--}" >&2

  prev_tok="${tok:-}"; prev_psec="${psec:-0}"; prev_prompt="${prompt:-}"
  sleep "$INTERVAL"
done

echo >&2
echo "csv: $CSV" >&2
