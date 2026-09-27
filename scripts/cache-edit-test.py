#!/usr/bin/env python3
"""Does an edit inside the prompt destroy the prefix cache, and does layout matter?

The earlier cache experiment (results/prefix-cache-20260927.txt) appended questions to an
UNCHANGED context, so it never tested the case an agent actually hits: a file it just
rewrote.

What matters is not whether the file was edited but WHERE it sits relative to the stable
material. A prefix cache breaks at the first changed token and everything after it must be
re-prefilled, so:

  file_late    stable material first, file last   -> an edit invalidates only the tail
  file_early   file first, stable material after  -> an edit invalidates the whole block
  file_early_appended  keep the original file and append the new state at the end

Each scenario carries a unique nonce so the server cannot reuse a prefix across scenarios,
which contaminated the first version of this test.

Usage:
  scripts/serve.sh start
  scripts/cache-edit-test.py [--tokens N]
"""
import argparse
import json
import pathlib
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:8080/v1/chat/completions"


def call(messages, tag, max_tokens=32):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "reasoning_effort": "none"}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    wall = time.time() - t0
    t = d["timings"]
    print(f"  {tag:20s} prompt_n={t['prompt_n']:6d} cache_n={t['cache_n']:6d} "
          f"prompt_ms={t['prompt_ms']:8.0f} wall={wall:5.1f}s")
    return t


def build_corpus(target_tokens):
    docs = ["docs/roofline-rtx3060.md", "docs/decode-profile.md", "docs/rt3060-profile.md",
            "docs/gemv-ncu-analysis.md", "docs/reasoning-mode-trade.md",
            "docs/review-findings-sol.md"]
    text = "\n\n".join((ROOT / d).read_text() for d in docs if (ROOT / d).exists())
    stable = text[: target_tokens * 4]
    file_v1 = ("def load_config(path):\n"
               "    with open(path) as fh:\n"
               "        return json.load(fh)\n\n"
               "def score(rows):\n"
               "    return sum(r.value for r in rows) / len(rows)\n") * 8
    file_v2 = file_v1.replace("json.load(fh)", "json.loads(fh.read())") \
                     .replace("def score(rows):", "def mean_score(rows):")
    return stable, file_v1, file_v2


RUN_ID = str(int(time.time()))  # unique per run: a repeated run must not reuse the
                                 # previous run's cache, which silently made every
                                 # "turn 1 (fresh)" a cache hit and hid the effect.


def scene(name, stable, v1, v2):
    """Return (turn1_messages, turn2_messages) for one scenario."""
    nonce = f"[run {RUN_ID} session {name}]\n"
    ref = "=== REFERENCE ===\n" + stable + "\n\n"
    src1 = "=== SOURCE FILE ===\n" + v1 + "\n"
    src2 = "=== SOURCE FILE ===\n" + v2 + "\n"
    t1 = "TASK: list the function names in the source file, one per line."
    t2 = "TASK: how many functions did you list? Reply with just the number."
    reply = {"role": "assistant", "content": "load_config\nscore"}

    if name == "file_late":
        turn1 = [{"role": "user", "content": nonce + ref + src1 + t1}]
        turn2 = [{"role": "user", "content": nonce + ref + src2 + t1}, reply,
                 {"role": "user", "content": t2}]
    elif name == "file_early":
        turn1 = [{"role": "user", "content": nonce + src1 + ref + t1}]
        turn2 = [{"role": "user", "content": nonce + src2 + ref + t1}, reply,
                 {"role": "user", "content": t2}]
    elif name == "file_early_appended":
        turn1 = [{"role": "user", "content": nonce + src1 + ref + t1}]
        # History is preserved byte for byte; the new state is appended at the end.
        turn2 = [{"role": "user", "content": nonce + src1 + ref + t1}, reply,
                 {"role": "user", "content": t2 + "\n\nCURRENT FILE STATE:\n" + v2}]
    else:
        raise ValueError(name)
    return turn1, turn2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=12000)
    args = ap.parse_args()

    stable, v1, v2 = build_corpus(args.tokens)
    print(f"run id {RUN_ID}; stable block ~{len(stable)//4} tokens, "
          f"source file ~{len(v1)//4} tokens\n")
    print("every scenario carries a unique nonce so no prefix is reused across runs "
          "or scenarios\n")
    print(f"{'scenario':22s} {'turn1 prompt_n':>14s} {'turn2 prompt_n':>14s} "
          f"{'turn2 cache_n':>13s} {'turn2 ms':>9s} {'vs turn1 ms':>11s}")

    for name in ("file_late", "file_early", "file_early_appended"):
        t1 = None
        turn1, turn2 = scene(name, stable, v1, v2)
        print(f"[{name}]")
        t1 = call(turn1, "turn 1 (fresh)")
        t2 = call(turn2, "turn 2 (after edit)")
        ratio = t2["prompt_ms"] / max(t1["prompt_ms"], 1)
        print(f"  -> edit cost {t2['prompt_ms']:.0f} ms = {ratio:.2f}x a fresh prefill; "
              f"{t2['cache_n']} tokens reused\n")


if __name__ == "__main__":
    main()
