#!/usr/bin/env python3
"""Probe per-request speculative decoding switching on a running llama-server.

The server must be started with speculation on (for example --spec-type ngram-simple). A server
started with --spec-type none has no speculative context, so nothing can be switched on, only off.

Two modes:

  switch  one prompt, a sequence that walks on -> off -> on twice, plus the rejection cases.
          Draft counters are the decisive column: generation rate varies by 20% or more between
          single requests against the same server, so one rate reading proves nothing.
  matrix  two workloads x two arms (speculation off / on), several reps each, with a warm-up per
          workload so the prefix cache is hot for both arms.

Usage:
    scripts/per-request-spec-probe.py --mode switch
    scripts/per-request-spec-probe.py --mode matrix --reps 3
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

# The order matters: it walks the on -> off -> on transition twice, so a stale per-slot draft
# left behind by an off request would show up as drafts on the following request.
SWITCH_CASES = [
    ("absent",            None),
    ("type=none",         {"type": "none"}),
    ("absent again",      None),
    ("type=ngram-simple", {"type": "ngram-simple"}),
    ("type=ngram-mod",    {"type": "ngram-mod"}),
    ("type=bogus",        {"type": "bogus"}),
    ("type=none",         {"type": "none"}),
    ("absent final",      None),
]

# workload name, prompt file, max_tokens. These are the two workloads that
# docs/agent-loop-vs-copy-turns.md separates: a novel short answer, and a full-file rewrite that
# copies its own input.
WORKLOADS = [
    ("low-overlap", "fixtures/prompts/low-overlap-chat.txt", 128),
    ("code-edit",   "fixtures/prompts/code-edit.txt",        256),
]
ARMS = [("off", {"type": "none"}), ("on", None)]


def post(url, body, timeout):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def cell(v, width=8, prec=None):
    if v is None:
        return f"{'-':>{width}}"
    if prec is not None:
        return f"{v:>{width}.{prec}f}"
    return f"{v:>{width}}"


def request(url, prompt, spec, max_tokens, timeout):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
    }
    if spec is not None:
        body["speculative"] = spec
    return post(url, body, timeout)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def run_switch(args):
    with open(args.prompt, encoding="utf-8") as f:
        prompt = f.read()

    print(f"POST {args.url}/v1/chat/completions   prompt={args.prompt} "
          f"({len(prompt)} chars)   max_tokens={args.max_tokens}")
    print(f"{'case':<20} {'HTTP':>4} {'draft_n':>8} {'accepted':>8} {'cache_n':>8} "
          f"{'prompt_n':>8} {'gen t/s':>9}  note")
    print("-" * 104)

    rows = []
    for label, spec in SWITCH_CASES:
        status, resp = request(args.url, prompt, spec, args.max_tokens, args.timeout)
        if status != 200:
            msg = resp.get("error", {}).get("message", "")
            print(f"{label:<20} {status:>4} {cell(None)} {cell(None)} {cell(None)} "
                  f"{cell(None)} {cell(None, 9)}  {msg}")
            rows.append({"case": label, "status": status, "error": msg})
            continue

        t = resp.get("timings", {})
        print(f"{label:<20} {status:>4} "
              f"{cell(t.get('draft_n'))} {cell(t.get('draft_n_accepted'))} "
              f"{cell(t.get('cache_n'))} {cell(t.get('prompt_n'))} "
              f"{cell(t.get('predicted_per_second'), 9, 2)}")
        rows.append({
            "case": label,
            "status": status,
            "draft_n": t.get("draft_n"),
            "draft_n_accepted": t.get("draft_n_accepted"),
            "cache_n": t.get("cache_n"),
            "prompt_n": t.get("prompt_n"),
            "predicted_per_second": t.get("predicted_per_second"),
        })
    return rows


def run_matrix(args):
    summary = []
    for wname, path, max_tokens in WORKLOADS:
        with open(path, encoding="utf-8") as f:
            prompt = f.read()

        # one warm-up so the prefix cache is hot for both arms
        request(args.url, prompt, {"type": "none"}, max_tokens, args.timeout)

        print(f"\n{wname}  ({path}, max_tokens={max_tokens})")
        print(f"{'arm':<6} {'rep':>4} {'HTTP':>4} {'draft_n':>8} {'accepted':>8} "
              f"{'cache_n':>8} {'prompt_n':>8} {'gen t/s':>9}")
        print("-" * 72)

        for arm, spec in ARMS:
            rates, drafts, accepted = [], [], []
            for rep in range(1, args.reps + 1):
                status, resp = request(args.url, prompt, spec, max_tokens, args.timeout)
                if status != 200:
                    print(f"{arm:<6} {rep:>4} {status:>4}   "
                          f"{resp.get('error', {}).get('message', '')}")
                    continue
                t = resp.get("timings", {})
                rate = t.get("predicted_per_second")
                rates.append(rate)
                drafts.append(t.get("draft_n"))
                accepted.append(t.get("draft_n_accepted"))
                print(f"{arm:<6} {rep:>4} {status:>4} {cell(t.get('draft_n'))} "
                      f"{cell(t.get('draft_n_accepted'))} {cell(t.get('cache_n'))} "
                      f"{cell(t.get('prompt_n'))} {cell(rate, 9, 2)}")

            summary.append({
                "workload": wname,
                "arm": arm,
                "reps": args.reps,
                "mean_tps": mean(rates),
                "mean_draft_n": mean(drafts),
                "mean_accepted": mean(accepted),
            })
            print(f"{'':<6} {'mean':>4} {'':>4} {cell(mean(drafts))} {cell(mean(accepted))} "
                  f"{'':>8} {'':>8} {cell(mean(rates), 9, 2)}")

    print(f"\n{'workload':<14} {'arm':<5} {'mean t/s':>9} {'mean draft_n':>13} "
          f"{'mean accepted':>14}")
    print("-" * 62)
    for r in summary:
        print(f"{r['workload']:<14} {r['arm']:<5} {cell(r['mean_tps'], 9, 2)} "
              f"{cell(r['mean_draft_n'], 13)} {cell(r['mean_accepted'], 14)}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["switch", "matrix"], default="switch")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt", default="fixtures/prompts/code-edit.txt")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = run_switch(args) if args.mode == "switch" else run_matrix(args)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
