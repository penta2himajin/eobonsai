#!/usr/bin/env python3
"""Rebuild the long-context prompts used by the depth experiments.

They are ~58 KB each and derived entirely from tracked documents, so they are generated
rather than committed. Both are 3 concatenated copies of the repo's engineering docs
(about 13.5K tokens) plus a task, which is what the depth measurements used.

Usage:
  python3 fixtures/prompts/make-long.py            # writes into out/
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "out"
SOURCES = ["docs/roofline-rtx3060.md", "docs/decode-profile.md", "docs/rt3060-profile.md"]


def filler() -> str:
    return "\n\n".join((ROOT / p).read_text() for p in SOURCES) * 3


def needle_prompt() -> str:
    """Copy task: the answer is a section at the end of a long document."""
    target = """NEEDLE_START
The measured read bandwidth of this card is 298 to 307 gigabytes per second, and the
ternary GEMV achieves 84.3 percent of that ceiling. Prefill is bound by the int8 dp4a
rate, which on this part delivers only 2.2 times the FP32 rate because GA106 has 64
integer lanes per SM against 128 FP32 lanes. Sixteen of the sixty four blocks are full
attention and the remaining forty eight are gated delta net layers, so the KV cache
grows by exactly 65536 bytes per token.
NEEDLE_END"""
    return (
        "You will be given a long document. At the very end there is a section delimited by "
        "NEEDLE_START and NEEDLE_END. Reproduce that section word for word, including the "
        "delimiters, and output nothing else.\n\n" + filler() + "\n\n" + target + "\n"
    )


def chat_prompt() -> str:
    """Non-copying task on the same long context, to check speculation does not regress."""
    return (
        "Below is a long technical document. Summarise its three most important conclusions "
        "in your own words, in about 200 words. Do not quote the document verbatim.\n\n"
        + filler() + "\n"
    )


def main():
    OUT.mkdir(exist_ok=True)
    for name, text in (("spec-prompt-long.txt", needle_prompt()),
                       ("spec-prompt-longchat.txt", chat_prompt())):
        path = OUT / name
        path.write_text(text)
        print(f"wrote {path.relative_to(ROOT)}  ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
