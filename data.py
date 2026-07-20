"""Loading the frozen data splits.

The split is declared in docs/splits.json: which task families are trained on,
which are held back entirely, and what fraction of items is reserved. The
item-level split is computed here, deterministically from the seed in that
file, so the exact question lists are reproducible without committing the data.

FOUR TIERS, AT INCREASING DISTANCE FROM TRAINING
------------------------------------------------
    train      80% of items from the training families
    held_out   the other 20%, SAME families      -> new questions, familiar reasoning
    transfer   entirely unseen families          -> new reasoning, same benchmark
    gsm8k      a different benchmark             -> never seen at all

Reporting all four is what turns "does it generalise?" from a question into a
table. The gap between held_out and transfer is the interesting one: it
separates learning the questions from learning to reason.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

SPLITS_PATH = Path(__file__).parent / "docs" / "splits.json"


@dataclass
class Item:
    id: str
    family: str
    question: str
    answer: str


@dataclass
class Splits:
    train: list[Item] = field(default_factory=list)
    held_out: list[Item] = field(default_factory=list)
    transfer: list[Item] = field(default_factory=list)
    gsm8k: list[Item] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "held_out": len(self.held_out),
            "transfer": len(self.transfer),
            "gsm8k": len(self.gsm8k),
        }


def load_spec(path: Path | str = SPLITS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def spec_hash(path: Path | str = SPLITS_PATH) -> str:
    """Fingerprint of the frozen split, recorded alongside every run.

    If this changes between two runs, those runs are not comparable, and the
    run log will show it rather than leaving you to wonder.
    """
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw).hexdigest()[:12]


def _split_items(items: list[Item], fraction: float, seed: int) -> tuple[list[Item], list[Item]]:
    """Deterministic item-level split. Same seed always gives the same lists."""
    shuffled = list(items)
    random.Random(seed).shuffle(shuffled)
    cut = int(len(shuffled) * (1 - fraction))
    return shuffled[:cut], shuffled[cut:]


def load_splits(
    path: Path | str = SPLITS_PATH,
    include_gsm8k: bool = True,
) -> Splits:
    """Materialise all four tiers from Hugging Face."""
    from datasets import load_dataset

    spec = load_spec(path)
    benchmark = spec["benchmark"]
    seed = spec["split_seed"]
    fraction = spec["held_out_fraction"]

    # Fail loudly on an unknown family name rather than silently producing an
    # empty split. A typo here would quietly shrink the training set.
    declared = (
        set(spec["train_families"])
        | set(spec["transfer_families"])
        | {f for group in spec["excluded"].values() for f in group["families"]}
    )

    splits = Splits()

    def load_family(family: str) -> list[Item]:
        try:
            data = load_dataset(benchmark, family, split="test")
        except Exception as exc:
            raise ValueError(
                f"Could not load family {family!r} from {benchmark}. Task names "
                f"in docs/splits.json must match the dataset exactly. "
                f"Underlying error: {exc}"
            ) from exc
        return [
            Item(
                id=f"{family}::{i}",
                family=family,
                question=row["input"],
                answer=str(row["target"]),
            )
            for i, row in enumerate(data)
        ]

    for family in spec["train_families"]:
        items = load_family(family)
        train_items, held_items = _split_items(items, fraction, seed)
        splits.train.extend(train_items)
        splits.held_out.extend(held_items)

    for family in spec["transfer_families"]:
        splits.transfer.extend(load_family(family))

    if include_gsm8k:
        cfg = spec["external_eval"]["gsm8k"]
        data = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])
        indices = list(range(len(data)))
        random.Random(cfg["sample_seed"]).shuffle(indices)
        for i in indices[: cfg["n_items"]]:
            row = data[i]
            # GSM8K answers end with "#### <number>".
            answer = row["answer"].split("####")[-1].strip()
            splits.gsm8k.append(
                Item(id=f"gsm8k::{i}", family="gsm8k", question=row["question"], answer=answer)
            )

    _sanity_check(splits, declared)
    return splits


def _sanity_check(splits: Splits, declared: set[str]) -> None:
    """The checks worth failing on before a training run, not after."""
    if not splits.train:
        raise ValueError("training split is empty")

    train_ids = {it.id for it in splits.train}
    for name, tier in (("held_out", splits.held_out), ("transfer", splits.transfer)):
        overlap = train_ids & {it.id for it in tier}
        if overlap:
            raise ValueError(
                f"{len(overlap)} items appear in both train and {name}. "
                "Every accuracy number downstream would be contaminated."
            )

    train_families = {it.family for it in splits.train}
    transfer_families = {it.family for it in splits.transfer}
    if train_families & transfer_families:
        raise ValueError(
            f"families in both train and transfer: {train_families & transfer_families}. "
            "Transfer must be entirely unseen or it measures nothing."
        )


if __name__ == "__main__":
    print("split fingerprint:", spec_hash())
    splits = load_splits()
    print(json.dumps(splits.summary(), indent=2))
