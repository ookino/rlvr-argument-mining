"""Turning the six measurements into the single number training uses.

READ THIS BEFORE CHANGING ANYTHING HERE
---------------------------------------
This file is the scientific core of the project. In a viva you will be asked
to justify every line of it. Nothing in here should be a number you cannot
explain the origin of.

Two rules the design follows:

1. WEIGHTS COME FROM THE CORRELATION STUDY, NOT FROM TASTE.
   You measure which features actually predict correct answers, and weight
   them accordingly. Features that predict nothing get weight zero. Setting
   these by hand and then reporting an improvement is the exact thing that
   makes a result unpublishable, so the code refuses to run without them.

2. THRESHOLDS ARE PERCENTILES OF THE UNTRAINED MODEL'S OWN OUTPUT.
   The length target is not "about 300 tokens", it is "the 75th percentile of
   what the base model produced before training". That way it is calibrated to
   the actual distribution rather than guessed, and you can say so in the
   method chapter.

Both come from calibrate.py, which writes reward/calibration.yaml. That file
is frozen and committed at the end of week 2; changing it afterwards is a
deviation log entry.
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
    length_target: float          # 75th percentile of baseline trace length
    min_steps: int = 3

    @classmethod
    def load(cls, path: Path | str = CALIBRATION_PATH) -> "Calibration":
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}

        weights = data.get("weights") or {}
        unset = sorted(k for k, v in weights.items() if v is None)
        if unset or not weights:
            raise ValueError(
                f"Feature weights {unset or '(none defined)'} are not set. "
                "Run calibrate.py on the baseline corpus first: weights come "
                "from the correlation study, never by hand. See scorer.py."
            )
        if data.get("length_target") is None:
            raise ValueError(
                "length_target is not set. It is the 75th percentile of "
                "baseline trace length, emitted by calibrate.py."
            )
        return cls(
            weights=weights,
            length_target=float(data["length_target"]),
            min_steps=int(data.get("min_steps", 3)),
        )


def length_penalty(n_tokens: int, target: float) -> float:
    """Stops the model padding its way to a better score.

    Every measurement is a fraction, so writing more text tends to create more
    links and inflate the graph. This multiplies the score down once a trace
    runs past the target length, so verbosity stops paying.
    """
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
    """The structural part of the reward, in [0, 1].

    `use_conflict` and `use_restatement` exist for the ablation run: switching
    both off leaves support structure alone, which is what tests whether the
    benefit comes from argument structure specifically or from any dense
    scoring signal at all.
    """
    values = features.as_dict()

    # Restatement and conflict are penalties, not rewards. Restating a step to
    # inflate the graph should cost, and a trace contradicting itself should
    # cost. Both are flipped so that more of them means a lower score.
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
    """What the training loop actually receives.

    Baseline condition: lambda_ = 0, so only correctness matters. This is the
    comparison everything else is measured against.
    Structural condition: correctness plus a weighted structural score.
    """
    reward = 1.0 if correct else 0.0
    if use_structure and lambda_:
        reward += lambda_ * structural_score(features, n_tokens, calibration, **ablation)
    return reward
