"""Within-group reward variance under two designs (the pre-registered probe)."""
import json, sys, statistics
from collections import defaultdict

LAMBDA = 0.5
SIX   = ["support_density", "attachment", "connectivity", "support_depth",
         "conflict_rate", "restatement_rate"]
THREE = ["support_density", "conflict_rate", "restatement_rate"]
PEN   = {"conflict_rate", "restatement_rate"}


def composite(r, feats):
    vals = [(1 - r[f]) if f in PEN else r[f] for f in feats]
    return sum(vals) / len(vals)


def run(path="results/E0/probe.jsonl"):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    words = sorted(len(r["trace"].split()) for r in rows)
    target = words[int(0.75 * len(words))] if words else 1

    def reward(r, feats):
        lp = min(1.0, target / max(1, len(r["trace"].split())))
        return int(r["correct"]) + LAMBDA * composite(r, feats) * lp

    groups = defaultdict(list)
    for r in rows:
        groups[r["question_id"]].append(r)
    n = len(groups)
    print(f"{len(rows)} completions across {n} questions "
          f"(mean {len(rows)/max(1,n):.1f} per question)")

    for label, feats in (("6-measure", SIX), ("3-count", THREE)):
        sds, degen, ug, udegen = [], 0, 0, 0
        for g in groups.values():
            rs = [reward(r, feats) for r in g]
            sd = statistics.pstdev(rs)
            sds.append(sd)
            if sd < 1e-9:
                degen += 1
            if len({r["correct"] for r in g}) == 1:
                ug += 1
                if sd < 1e-9:
                    udegen += 1
        print(f"\n=== {label} ===")
        print(f"  mean within-group sd : {statistics.mean(sds):.4f}")
        print(f"  degenerate groups    : {degen}/{n} ({100*degen/max(1,n):.0f}%)  <- zero gradient")
        print(f"  uniform-correctness  : {udegen}/{ug} degenerate (structure is the only lever)")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "results/E0/probe.jsonl")

