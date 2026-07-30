"""Compare ARI's native edge direction vs our reconfigured one.

We store support links premise -> conclusion (earlier -> later) because a chain
of thought builds up to its conclusion. The published model was trained on
argumentative data (debate/AIF) where the convention may be the other way,
conclusion -> premise. This flips the saved links and recomputes the six
measures, so we can see how much the direction actually changes the numbers -
without re-running ARI (the raw links are already saved).

    from direction_check import run
    run()
"""

import json
import statistics

from reward.ari import Relation, RelationResult
from reward.features import compute
from reward.xaif_build import build_trace

KEYS = ["support_density", "attachment", "connectivity",
        "support_depth", "conflict_rate", "restatement_rate"]


def _features(row, flip):
    trace = build_trace(row["trace"])
    rels = []
    for r in row["relations"]:
        s, t = (r["t"], r["s"]) if flip else (r["s"], r["t"])
        rels.append(Relation(source=s, target=t, kind=r["k"], confidence=r["c"]))
    return compute(trace, RelationResult(relations=rels)).as_dict()


def run(path="results/E0/corpus_rescored.jsonl"):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ours = {k: [] for k in KEYS}
    his = {k: [] for k in KEYS}
    changed = 0
    for row in rows:
        fo = _features(row, flip=False)   # ours: premise -> conclusion
        fh = _features(row, flip=True)    # his:  conclusion -> premise
        for k in KEYS:
            ours[k].append(fo[k])
            his[k].append(fh[k])
        if (abs(fo["connectivity"] - fh["connectivity"]) > 0.01
                or abs(fo["support_depth"] - fh["support_depth"]) > 0.01):
            changed += 1

    print(f"{len(rows)} traces\n")
    print(f"{'feature':18} {'ours (P->C)':>12} {'his (C->P)':>12} {'diff':>9}")
    for k in KEYS:
        mo, mh = statistics.mean(ours[k]), statistics.mean(his[k])
        flag = "  <-- moves" if abs(mh - mo) > 0.01 else ""
        print(f"{k:18} {mo:12.3f} {mh:12.3f} {mh - mo:+9.3f}{flag}")
    print(f"\ntraces where connectivity/depth changed: {changed}/{len(rows)}")


if __name__ == "__main__":
    run()
