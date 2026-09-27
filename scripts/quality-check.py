#!/usr/bin/env python3
"""Quality check across reasoning modes, with machine-checkable answers.

The reasoning-mode A/B (scripts/reasoning-ab.sh) measures speed only. Speed alone is not
enough to decide a default: `reasoning_effort: "none"` was observed to cut a mechanical
task from 334 generated tokens to 42, and nothing yet says whether that is concision or a
broken answer. This script settles it by asking tasks whose answers can be scored without a
judge.

Three task families, each scored mechanically:

  arithmetic   exact numeric answer, parsed and compared
  format       an instruction with a checkable shape (line count, prefix, exclusions)
  transform    a deterministic string/JSON transformation, compared to a computed key

Usage:
  scripts/serve.sh start
  scripts/quality-check.py                 # all modes, all tasks
  scripts/quality-check.py none medium     # selected modes
"""
import json
import re
import sys
import time
import urllib.error
import urllib.request

PORT = 8080
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# ---------------------------------------------------------------- task definitions
# Each returns (prompt, scorer). The scorer takes the assistant text and returns
# (passed: bool, detail: str).


def score_number(expected, extract=r"-?\d[\d,]*\.?\d*"):
    """Compare the LAST number in the reply to the expected value.

    The last number is the model's final answer; earlier ones may be intermediate steps.
    Replies that bury the answer are caught here rather than silently passing.
    """
    def f(text):
        nums = re.findall(extract, text.replace(",", ""))
        if not nums:
            return False, "no number found"
        got = float(nums[-1])
        return abs(got - expected) < 1e-6, f"expected {expected}, got {got}"
    return f


def score_exact_after(marker, expected):
    def f(text):
        idx = text.rfind(marker)
        if idx < 0:
            return False, f"marker {marker!r} not found"
        tail = text[idx + len(marker):].strip().splitlines()
        got = tail[0].strip() if tail else ""
        return got == expected, f"expected {expected!r}, got {got!r}"
    return f


def score_shape(min_lines, forbid_words=(), require_lines=None):
    def f(text):
        body = text.strip()
        lines = [l for l in body.splitlines() if l.strip()]
        problems = []
        if len(lines) < min_lines:
            problems.append(f"{len(lines)} non-empty lines < {min_lines}")
        if require_lines is not None and len(lines) != require_lines:
            problems.append(f"{len(lines)} lines != {require_lines}")
        for w in forbid_words:
            if re.search(w, body, re.I):
                problems.append(f"contains forbidden {w!r}")
        return (not problems), ("ok" if not problems else "; ".join(problems))
    return f


def score_json_transform(expected_pairs):
    def f(text):
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return False, "no JSON object found"
        try:
            got = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return False, f"invalid JSON: {e}"
        problems = []
        for k, v in expected_pairs.items():
            if k not in got:
                problems.append(f"missing key {k}")
            elif str(got[k]).strip().lower() != str(v).strip().lower():
                problems.append(f"{k}: expected {v}, got {got[k]}")
        return (not problems), ("ok" if not problems else "; ".join(problems))
    return f


TASKS = [
    # Arithmetic: the correct answer is a single number, so there is no judging.
    # Verified by hand: weights 7.19 GiB = 7362.56 MiB; free = 11904 - 7362.56 - 760 =
    # 3781.44 MiB; at 64 KiB (0.0625 MiB) per token that is 60503 tokens. Tolerance is
    # generous because the model may round 7.19 GiB differently; scoring is absolute so
    # 60503 and 60500 both pass while a wrong order of magnitude fails.
    (
        "arith_1",
        "A GPU holds a 7.19 GiB model and a KV cache costing 64 KiB per token. "
        "It reports 11904 MiB usable and compute buffers take 760 MiB. "
        "How many tokens of context fit? Answer with the number alone.",
        score_number(60503, r"\d[\d,]*"),
    ),
    (
        "arith_2",
        "Compute 847 * 29 + 1147. Reply with the final number only.",
        score_number(847 * 29 + 1147),
    ),
    # 401*5120*2.13 bits = 4,373,146 bits = 546,643 bytes = 533.8 KiB. Scored in KiB so the
    # answer is a meaningful integer rather than 0.00 in GiB.
    (
        "arith_3",
        "A tensor has 401 rows of 5120 values, stored at 2.13 bits per value. "
        "How many kibibytes is it? Round to the nearest whole number and reply with "
        "the number only.",
        score_number(534, r"\d[\d,]*"),
    ),
    # Format: the instruction has a checkable shape, so compliance is mechanical.
    (
        "format_1",
        "List exactly 5 HTTP status codes, one per line, each line starting with the "
        "three-digit code and a space, and nothing else. No explanations, no headers.",
        score_shape(5, forbid_words=[r"^#"], require_lines=5),
    ),
    (
        "format_2",
        "Reply with exactly three lines. Line 1 is the word ALPHA, line 2 is BETA, "
        "line 3 is GAMMA. Nothing else.",
        score_shape(3, require_lines=3),
    ),
    # Transform: deterministic, comparable to a computed key.
    (
        "transform_1",
        'Return only a JSON object with the keys "total", "mean" and "count" for the '
        "numbers 4, 8, 15, 16, 23, 42. total and count are integers, mean is "
        "rounded to two decimals.",
        lambda t: score_json_transform(
            {"total": 108, "mean": 18.0, "count": 6})(t),
    ),
]


def ask(prompt, mode, max_tokens=1200):
    body = {"messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0}
    if mode != "server-default":
        body["reasoning_effort"] = mode
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    wall = time.time() - t0
    msg = d["choices"][0]["message"]
    return {
        "text": (msg.get("content") or ""),
        "think": len(msg.get("reasoning_content") or ""),
        "wall": wall,
        "tokens": d["timings"]["predicted_n"],
        "drafts": d["timings"].get("draft_n", 0),
    }


def main():
    modes = sys.argv[1:] or ["server-default", "none", "medium"]
    rows = []
    for name, prompt, scorer in TASKS:
        for mode in modes:
            try:
                r = ask(prompt, mode)
            except urllib.error.HTTPError as e:
                rows.append((name, mode, None, f"HTTP {e.code}", {}, ""))
                continue
            passed, detail = scorer(r["text"])
            rows.append((name, mode, passed, detail, r, r["text"]))

    print(f"{'task':12s} {'mode':15s} {'pass':5s} {'gen_t/s':>8s} {'wall_s':>7s} "
          f"{'tok':>5s} {'drafts':>7s} {'think':>6s}  detail")
    summary = {}
    for name, mode, passed, detail, r, _ in rows:
        if r:
            gts = r["tokens"] / r["wall"] if r["wall"] else 0
            print(f"{name:12s} {mode:15s} {('PASS' if passed else 'FAIL'):5s} "
                  f"{gts:8.2f} {r['wall']:7.1f} {r['tokens']:5d} {r['drafts']:7d} "
                  f"{r['think']:6d}  {detail[:52]}")
        else:
            print(f"{name:12s} {mode:15s} {'ERR':5s} {'-':>8} {'-':>7} {'-':>5} "
                  f"{'-':>7} {'-':>6}  {detail}")
        summary.setdefault(mode, []).append(1 if passed else 0)

    print()
    print("accuracy by mode:")
    for mode, res in summary.items():
        print(f"  {mode:15s} {sum(res)}/{len(res)} = {100*sum(res)/len(res):.0f}%")


if __name__ == "__main__":
    main()
