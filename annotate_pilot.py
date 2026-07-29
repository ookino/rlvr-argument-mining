"""Pilot hand-annotation for RQ1: check whether ARI's links are actually right.

Two steps:
  python annotate_pilot.py make      -> writes results/E0/annotation_sheet.md
  (you fill it in by hand)
  python annotate_pilot.py score     -> reads it back, prints precision/recall

You mark each link ARI drew as [y] (real) or [n] (wrong), and write any link
it MISSED. That's the whole job. Precision = of the links it drew, how many
were real. Recall = of the real links, how many it caught.
"""

import json
import random
import re
import sys

from reward.xaif_build import build_trace

CORPUS = "results/E0/corpus_rescored.jsonl"
SHEET = "results/E0/annotation_sheet.md"
N_PER_FAMILY = 2          # a couple per family so every task is represented
SEED = 7                  # fixed so the sample is reproducible

KIND_NAME = {"RA": "supports", "CA": "attacks", "MA": "restates"}


def load():
    with open(CORPUS) as fh:
        return [json.loads(line) for line in fh]


def sample(rows):
    # a couple of traces from each family, only ones with enough steps to judge
    by_family = {}
    for r in rows:
        by_family.setdefault(r["family"], []).append(r)
    rng = random.Random(SEED)
    picked = []
    for family in sorted(by_family):
        usable = [r for r in by_family[family]
                  if len(build_trace(r["trace"]).steps) >= 3]
        rng.shuffle(usable)
        picked.extend(usable[:N_PER_FAMILY])
    return picked


def make():
    rows = sample(load())
    out = ["# Pilot annotation sheet",
           "",
           "For each link ARI drew, change `[ ]` to `[y]` if the link is real,",
           "`[n]` if it is wrong. If ARI missed a real link, add a line under",
           "MISSED like `MISS: 2 -> 5 supports`. Then run `score`.",
           "",
           f"{len(rows)} traces sampled (seed {SEED}).",
           ""]
    for n, r in enumerate(rows, 1):
        trace = build_trace(r["trace"])
        steps = trace.steps
        concl = trace.conclusion_index
        out.append(f"## Trace {n}  (id {r['id']}, {r['family']})")
        out.append(f"model answer {'correct' if r['correct'] else 'wrong'}\n")
        out.append("Steps:")
        for i, s in enumerate(steps):
            tag = "  <- conclusion" if i == concl else ""
            out.append(f"  {i}. {s}{tag}")
        out.append("")
        out.append("Links ARI drew:")
        if not r["relations"]:
            out.append("  (none)")
        for k, rel in enumerate(r["relations"], 1):
            name = KIND_NAME.get(rel["k"], rel["k"])
            out.append(f"  [ ] {rel['s']} {name} {rel['t']}   (conf {rel['c']:.2f})")
        out.append("")
        out.append("MISSED (add real links ARI did not draw, one per line):")
        out.append("")
    with open(SHEET, "w") as fh:
        fh.write("\n".join(out))
    print(f"wrote {SHEET} with {len(rows)} traces. fill it in, then: score")


LINK_RE = re.compile(r"\[([ynYN ])\]")
MISS_RE = re.compile(r"MISS:", re.I)


def score():
    with open(SHEET) as fh:
        lines = fh.readlines()
    drawn = agreed = missed = 0
    for line in lines:
        m = LINK_RE.search(line)
        if m and "conf" in line:          # a link row
            mark = m.group(1).strip().lower()
            drawn += 1
            if mark == "y":
                agreed += 1
            elif mark != "n":
                print(f"  unmarked link: {line.strip()}")
        elif MISS_RE.search(line):
            missed += 1

    real = agreed + missed
    precision = agreed / drawn if drawn else 0.0
    recall = agreed / real if real else 0.0
    print(f"links ARI drew:      {drawn}")
    print(f"  judged real (y):   {agreed}")
    print(f"  judged wrong (n):  {drawn - agreed}")
    print(f"links ARI missed:    {missed}")
    print(f"precision (of its links, how many real): {precision:.2f}")
    print(f"recall    (of real links, how many caught): {recall:.2f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "make"
    {"make": make, "score": score}[cmd]()
