"""
E2 correlation study: does argument structure predict answer correctness?

Reads the E0 corpus and runs the analyses that were pre-registered in
docs/experiment_plan.md, in this order:
  1. point-biserial correlation of each feature with correctness (pooled + per family)
  2. logistic regression with trace length controlled, so we can tell structure
     apart from "correct answers are just shorter"
  3. area under the ROC curve (AUC) for the structure composite vs a length-only
     baseline
  4. the pre-registered verdict (AUC thresholds 0.60 / 0.55)
  5. proposed reward weights and length target for calibration.yaml

Run it:
    python analyse.py
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.metrics import roc_auc_score

FEATURES = ["support_density", "attachment", "connectivity",
            "support_depth", "conflict_rate", "restatement_rate"]

# Which way each feature should point if it helps. Support features should be
# higher for correct answers; the two penalties should be lower.
EXPECTED_SIGN = {
    "support_density": +1, "attachment": +1, "connectivity": +1,
    "support_depth": +1, "conflict_rate": -1, "restatement_rate": -1,
}


def load(path="results/E0/corpus.jsonl"):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    # word count of the trace, a stand-in for token length
    df["length"] = df["trace"].str.split().str.len()
    return df


def composite(df):
    # Uniform combination of the six, with the two penalties flipped so higher
    # always means "better structured". This is the no-weights-yet baseline.
    parts = [df[f] if EXPECTED_SIGN[f] > 0 else (1 - df[f]) for f in FEATURES]
    return sum(parts) / len(parts)


def run(path="results/E0/corpus.jsonl"):
    df = load(path)
    n_all = len(df)

    # Extraction failures have an unreliable correctness label, so drop them for
    # the correctness analysis and report how many.
    n_fail = int((~df["extract_ok"]).sum())
    df = df[df["extract_ok"]].copy()
    print(f"corpus: {n_all} traces, dropped {n_fail} extraction failures, "
          f"{len(df)} used")
    print(f"correct: {int(df['correct'].sum())}/{len(df)} "
          f"({100*df['correct'].mean():.0f}%)\n")

    y = df["correct"].astype(int).values

    # --- 1. point-biserial correlation, pooled ---
    print("=== 1. point-biserial correlation with correctness (pooled) ===")
    print(f"{'feature':18s} {'r':>7} {'p':>8}  expected sign")
    pooled_r = {}
    for f in FEATURES + ["length"]:
        r, p = pointbiserialr(y, df[f].values)
        pooled_r[f] = r
        exp = "" if f == "length" else ("+" if EXPECTED_SIGN[f] > 0 else "-")
        flag = ""
        if f in EXPECTED_SIGN and abs(r) > 0.1 and np.sign(r) == EXPECTED_SIGN[f]:
            flag = "  <- predictive, right direction"
        print(f"{f:18s} {r:+7.3f} {p:8.3f}  {exp}{flag}")

    # --- 1b. per family ---
    print("\n=== per family (point-biserial r, n) ===")
    fams = sorted(df["family"].unique())
    header = "family".ljust(34) + "".join(f"{f[:8]:>9}" for f in FEATURES)
    print(header)
    for fam in fams:
        sub = df[df["family"] == fam]
        yy = sub["correct"].astype(int).values
        cells = []
        for f in FEATURES:
            if sub[f].nunique() < 2 or len(set(yy)) < 2:
                cells.append(f"{'--':>9}")
            else:
                r, _ = pointbiserialr(yy, sub[f].values)
                cells.append(f"{r:+9.2f}")
        print(f"{fam:34s}" + "".join(cells) + f"  (n={len(sub)})")

    # --- 2. logistic regression, length controlled ---
    print("\n=== 2. logistic regression: correct ~ features + length + family ===")
    try:
        import statsmodels.formula.api as smf
        df_m = df.copy()
        df_m["correct"] = df_m["correct"].astype(int)
        formula = ("correct ~ " + " + ".join(FEATURES) +
                   " + length + C(family)")
        res = smf.logit(formula, data=df_m).fit(disp=False, maxiter=200)
        print("  (coefficient, p-value; only features + length shown)")
        for f in FEATURES + ["length"]:
            print(f"    {f:18s} coef {res.params[f]:+7.3f}   p {res.pvalues[f]:.3f}")
    except Exception as exc:  # noqa: BLE001
        print("  logistic regression did not fit cleanly:", exc)

    # --- 3. AUC: composite vs length-only ---
    print("\n=== 3. AUC (higher = better predictor of correctness) ===")
    comp = composite(df).values
    auc_comp = roc_auc_score(y, comp)
    auc_len = roc_auc_score(y, -df["length"].values)   # shorter -> correct
    print(f"  structure composite : {auc_comp:.3f}")
    print(f"  length only         : {auc_len:.3f}")
    print(f"  (0.5 = no better than a coin flip)")

    # --- 4. pre-registered verdict ---
    print("\n=== 4. verdict (pre-registered thresholds) ===")
    if auc_comp > 0.60:
        verdict = "MEANINGFUL signal: proceed as planned."
    elif auc_comp >= 0.55:
        verdict = "WEAK signal: proceed but temper the hypothesis."
    else:
        verdict = ("NO predictive signal: reframe C3 as testing whether "
                   "optimisation CREATES structure the base model does not exploit.")
    print(f"  composite AUC = {auc_comp:.3f}  ->  {verdict}")
    if auc_len > auc_comp:
        print("  NOTE: length alone predicts better than structure. The length "
              "confound is doing the work, not argument structure.")

    # --- 5. proposed weights + length target ---
    print("\n=== 5. proposed calibration ===")
    kept = {f: abs(pooled_r[f]) for f in FEATURES
            if abs(pooled_r[f]) > 0.1 and np.sign(pooled_r[f]) == EXPECTED_SIGN[f]}
    if kept:
        total = sum(kept.values())
        weights = {f: (round(kept[f] / total, 3) if f in kept else 0.0)
                   for f in FEATURES}
        print("  features that predict correctness (|r|>0.1, right sign) get "
              "weight proportional to |r|; the rest get 0:")
    else:
        weights = {f: 0.0 for f in FEATURES}
        print("  NO feature clears |r|>0.1 in the right direction. All weights "
              "would be 0, i.e. the structure reward has no empirical basis "
              "from this corpus. This is the null outcome the plan anticipated.")
    for f in FEATURES:
        print(f"    {f:18s} {weights[f]}")
    l_target = float(np.percentile(df["length"], 75))
    print(f"  length_target (p75 of trace word count) = {l_target:.0f}")

    return dict(auc_comp=auc_comp, auc_len=auc_len, weights=weights,
                length_target=l_target, pooled_r=pooled_r)


if __name__ == "__main__":
    run()
