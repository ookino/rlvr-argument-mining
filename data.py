"""Load the data splits (declared in docs/splits.json).

Five tiers, each further from training. Only `train` is trained on; the rest
test generalisation.
    train      80% of the training families
    held_out   the other 20% (same families)
    transfer   unseen BBH families
    logiqa     different benchmark (logic comprehension)
    gsm8k      different benchmark (maths)
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
    logiqa: list[Item] = field(default_factory=list)

    def summary(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "held_out": len(self.held_out),
            "transfer": len(self.transfer),
            "gsm8k": len(self.gsm8k),
            "logiqa": len(self.logiqa),
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
    include_external: bool = True,
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

    if include_external:
        ext = spec["external_eval"]
        splits.gsm8k = _load_gsm8k(ext["gsm8k"])
        splits.logiqa = _load_logiqa(ext["logiqa"])

    _sanity_check(splits, declared)
    return splits


def _sample_indices(n_rows: int, n_items: int, seed: int) -> list[int]:
    indices = list(range(n_rows))
    random.Random(seed).shuffle(indices)
    return indices[:n_items]


def _load_gsm8k(cfg: dict) -> list[Item]:
    """Grade-school maths. Answers end with '#### <number>'."""
    from datasets import load_dataset

    data = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])
    items = []
    for i in _sample_indices(len(data), cfg["n_items"], cfg["sample_seed"]):
        row = data[i]
        items.append(
            Item(
                id=f"gsm8k::{i}",
                family="gsm8k",
                question=row["question"],
                answer=row["answer"].split(cfg["answer_marker"])[-1].strip(),
            )
        )
    return items


def _load_logiqa(cfg: dict) -> list[Item]:
    """Logical reading comprehension, multiple choice.

    Several LogiQA mirrors exist on Hugging Face with different field names,
    so this probes for the ones it knows and reports what it actually found
    rather than guessing and producing silently malformed questions.
    """
    from datasets import load_dataset

    args = [cfg["dataset"]] + ([cfg["config"]] if cfg.get("config") else [])
    try:
        data = load_dataset(*args, split=cfg["split"])
    except Exception as exc:
        raise ValueError(
            f"Could not load LogiQA from {cfg['dataset']!r}. Try an alternative "
            f"mirror and update docs/splits.json. Underlying error: {exc}"
        ) from exc

    fields = set(data.column_names)
    layouts = [
        ("context", "query", "options", "correct_option"),
        ("context", "question", "options", "label"),
        ("premise", "question", "answers", "label"),
    ]
    layout = next((l for l in layouts if set(l[:3]) <= fields), None)
    if layout is None:
        raise ValueError(
            f"LogiQA loaded but its fields are unrecognised: {sorted(fields)}. "
            f"Expected one of {layouts}. Add the correct layout to _load_logiqa."
        )
    ctx_f, q_f, opt_f, ans_f = layout

    items = []
    for i in _sample_indices(len(data), cfg["n_items"], cfg["sample_seed"]):
        row = data[i]
        options = row[opt_f]
        lettered = "\n".join(
            f"({chr(65 + k)}) {opt}" for k, opt in enumerate(options)
        )
        answer_idx = row[ans_f]
        if isinstance(answer_idx, str) and answer_idx.isdigit():
            answer_idx = int(answer_idx)
        items.append(
            Item(
                id=f"logiqa::{i}",
                family="logiqa",
                question=f"{row[ctx_f]}\n\n{row[q_f]}\n{lettered}",
                answer=chr(65 + int(answer_idx)) if isinstance(answer_idx, int) else str(answer_idx),
            )
        )
    return items


def _sanity_check(splits: Splits, declared: set[str]) -> None:
    """The checks worth failing on before a training run, not after."""
    if not splits.train:
        raise ValueError("training split is empty")

    train_ids = {it.id for it in splits.train}
    for name, tier in (("held_out", splits.held_out), ("transfer", splits.transfer),
                       ("gsm8k", splits.gsm8k), ("logiqa", splits.logiqa)):
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
