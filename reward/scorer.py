"""Combine the six measures into the reward number.

Two things I'm being strict about:
- the weights come from the correlation study, not picked by hand (picking them
  and then reporting an improvement would be circular). the code refuses to run
  if they're not set.
- the length target is the 75th percentile of the base model's own trace length,
  not a made-up constant.

Both are written by calibrate.py into calibration.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from reward.features import Features

CALIBRATION_PATH = Path(__file__).parent / "calibration.yaml"


@dataclass
class Calibration:
    weights: dict[str, float]
    length_target: float          # 75th pct of base-model trace length
    min_steps: int = 3

    @classmethod
    def load(cls, path: Path | str = CALIBRATION_PATH) -> "Calibration":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        weights = data.get("weights") or {}
        unset = sorted(k for k, v in weights.items() if v is None)
        if unset or not weights:
            raise ValueError(
                f"weights {unset or '(none)'} not set - run calibrate.py first, "
                "weights come from the correlation study not by hand"
            )
        if data.get("length_target") is None:
            raise ValueError("length_target not set (calibrate.py writes it)")
        return cls(
            weights=weights,
            length_target=float(data["length_target"]),
            min_steps=int(data.get("min_steps", 3)),
        )


def length_penalty(n_tokens: int, target: float) -> float:
    # writing more text makes more links, so cap it: past the target length the
    # score gets shrunk so rambling doesn't pay
    if n_tokens <= target:
        return 1.0
    return target / n_tokens


def structural_score(
    features: Features,
    n_tokens: int,
    calibration: Calibration,
    use_conflict: bool = True,
    use_restatement: bool = True,
) -> float:
    # use_conflict / use_restatement off = the ablation run (support only), to
    # check whether it's the argument content or just any dense signal helping
    values = features.as_dict()

    # conflict and restatement are penalties - flip them so more = lower score
    values["restatement_rate"] = 1.0 - values["restatement_rate"]
    values["conflict_rate"] = 1.0 - values["conflict_rate"]

    if not use_conflict:
        values.pop("conflict_rate", None)
    if not use_restatement:
        values.pop("restatement_rate", None)

    active = {k: w for k, w in calibration.weights.items() if k in values and w}
    if not active:
        return 0.0

    total = sum(active.values())
    score = sum(values[k] * w for k, w in active.items()) / total
    return score * length_penalty(n_tokens, calibration.length_target)


def total_reward(
    correct: bool,
    features: Features,
    n_tokens: int,
    calibration: Calibration,
    lambda_: float = 0.5,
    use_structure: bool = True,
    **ablation,
) -> float:
    # lambda_=0 -> correctness only (the baseline). otherwise correctness plus
    # the weighted structure score.
    reward = 1.0 if correct else 0.0
    if use_structure and lambda_:
        reward += lambda_ * structural_score(features, n_tokens, calibration, **ablation)
    return reward
