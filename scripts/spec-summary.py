#!/usr/bin/env python3
"""Extract a committable summary from the verbose speculative-decoding run logs.

The verbose llama-cli logs are large (about 1 MB each) and are not tracked; this pulls
out the numbers that docs and ADRs cite, into one small table.

Usage: python3 scripts/spec-summary.py [results-dir]
"""
import csv
import glob
import os
import re
import sys

SPEC = re.compile(r"(none|ngram-[a-z0-9-]+)")
EVAL = re.compile(
    r"eval time = *([0-9.]+) ms / *([0-9]+) tokens \( *[0-9.]+ ms per token, *([0-9.]+) tokens per second\)"
)
PROMPT = re.compile(r"prompt eval time = *[0-9.]+ ms / *([0-9]+) tokens")


def parse(path):
    text = open(path, errors="replace").read()
    evals = EVAL.findall(text)
    if not evals:
        return None
    _, gen_tokens, gen_tps = evals[-1]
    prompts = PROMPT.findall(text)
    draft = re.search(r"#gen drafts = *([0-9]+)", text)
    acc = re.search(r"#acc drafts = *([0-9]+)", text)
    drafts = int(draft.group(1)) if draft else 0
    accepted = int(acc.group(1)) if acc else 0

    stem = os.path.basename(path).removeprefix("spec-").removesuffix(".log")
    stem = re.sub(r"-\d{8}-\d{6}$", "", stem)          # drop the timestamp
    specs = SPEC.findall(stem)
    spec = specs[-1] if specs else "?"
    workload = stem[: stem.rfind(spec)].rstrip("-") if spec in stem else stem
    # llama-cli marks the thinking trace, so its presence tells us which mode ran.
    thinking = "on" if ("[Start thinking]" in text or "Reasoning effort is set to" in text) else "off"
    return {
        "workload": workload or "?",
        "spec_type": spec,
        "thinking": thinking,
        "prompt_tokens": int(prompts[-1]) if prompts else 0,
        "gen_tokens": int(gen_tokens),
        "gen_t_per_s": float(gen_tps),
        "drafts": drafts,
        "accepted": accepted,
        "accept_pct": f"{100 * accepted / drafts:.0f}%" if drafts else "-",
    }


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "results"
    rows = [r for r in (parse(p) for p in sorted(glob.glob(f"{root}/spec-*.log"))) if r]
    out = f"{root}/spec-summary-20260927.txt"
    with open(out, "w", newline="") as fh:
        fh.write("# n-gram speculative decoding, extracted (llama-cli, temp 0, REASONING=off)\n")
        fh.write("# source: verbose logs from scripts/specbench.sh and the depth runs.\n")
        fh.write("# Those logs are large and regenerable, so they are not tracked; this table is.\n")
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["workload"], r["spec_type"])):
            w.writerow(r)
    print(open(out).read())


if __name__ == "__main__":
    main()
