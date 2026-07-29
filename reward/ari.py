"""ARI - the argument relation model (Ramon's model). We don't train this one.

Give it two statements, it says one of: Inference (A supports B -> RA),
Conflict (A contradicts B -> CA), Rephrase (A restates B -> MA), or nothing.

We load the checkpoint directly instead of the oAMF web service, because that
needs Docker and Colab doesn't have it. Same model, same thresholds. (D-002)

Cost: it scores PAIRS of steps. window=None does every pair (n*(n-1)/2),
window=k only neighbours (n-1). That one setting drives the training cost.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

# torch/transformers are imported inside ARI() so the rest of the file (pairing,
# features, scorer) still imports on a laptop with no GPU stack.

logger = logging.getLogger(__name__)

MODEL_ID = "raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L"

# confidence floors, copied from the published module (support is stricter). we
# don't retune these - they're Ramon's numbers.
THRESHOLDS = {
    "Inference": 0.9,
    "Conflict": 0.7,
    "Rephrase": 0.7,
}

LABEL_TO_RELATION = {
    "Inference": "RA",
    "Conflict": "CA",
    "Rephrase": "MA",
}

# the model has a 4th class: it can say a pair is just unrelated. we count that
# separately from "guessed a relation but below the floor" - different things.
NO_RELATION_LABELS = {"No-Relation", "NoRelation", "None", "no-relation"}


@dataclass
class Relation:
    source: int          # step doing the supporting/attacking
    target: int          # step being supported/attacked
    kind: str            # RA / CA / MA
    confidence: float


@dataclass
class RelationResult:
    relations: list[Relation] = field(default_factory=list)
    n_pairs_scored: int = 0
    n_below_threshold: int = 0       # guessed a relation, below its floor
    n_model_no_relation: int = 0     # said "these are unrelated"


class ARI:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 256,
        encoding: str = "concat",     # matches the published module (see _encode)
        thresholds: dict | None = None,   # override to sweep the confidence floors
    ):
        if encoding not in ("concat", "pair"):
            raise ValueError("encoding must be 'concat' or 'pair'")
        self.encoding = encoding
        self.thresholds = thresholds or THRESHOLDS

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(device).eval()

        self.id2label = self.model.config.id2label

        # warn if the checkpoint ever shows a label we don't know how to handle
        recognised = set(THRESHOLDS) | NO_RELATION_LABELS
        unknown = set(self.id2label.values()) - recognised
        if unknown:
            logger.warning("unrecognised labels %s, treating as no-relation", sorted(unknown))

    @staticmethod
    def make_pairs(n_steps: int, window: int | None = None) -> list[tuple[int, int]]:
        # window=None = all pairs; window=k = each step vs the next k-1. always
        # (earlier, later), no duplicates.
        if n_steps < 2:
            return []
        if window is None:
            return list(itertools.combinations(range(n_steps), 2))
        if window < 2:
            raise ValueError("window must be >= 2, or None for all pairs")
        return [
            (i, j)
            for i in range(n_steps)
            for j in range(i + 1, min(i + window, n_steps))
        ]

    def _encode(self, chunk: list[tuple[str, str]]):
        # two ways to feed a sentence pair: glue them ("A. B") or pass as two
        # args. the published module glues them, so that's the default. keeping
        # the other as an option for a sanity check.
        if self.encoding == "concat":
            texts = [f"{a}. {b}" for a, b in chunk]
            return self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
        return self.tokenizer(
            [a for a, _ in chunk],
            [b for _, b in chunk],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def _classify(self, pairs: list[tuple[str, str]]) -> list[tuple[str, float]]:
        import torch

        out: list[tuple[str, float]] = []
        with torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                chunk = pairs[start : start + self.batch_size]
                encoded = self._encode(chunk).to(self.device)
                probs = self.model(**encoded).logits.softmax(dim=-1)
                best = probs.argmax(dim=-1)
                for row, idx in enumerate(best.tolist()):
                    out.append((self.id2label[idx], probs[row, idx].item()))
        return out

    def identify(self, steps: list[str], window: int | None = None) -> RelationResult:
        index_pairs = self.make_pairs(len(steps), window)
        if not index_pairs:
            return RelationResult()

        text_pairs = [(steps[i], steps[j]) for i, j in index_pairs]
        predictions = self._classify(text_pairs)

        result = RelationResult(n_pairs_scored=len(index_pairs))
        for (i, j), (label, confidence) in zip(index_pairs, predictions):
            if label in NO_RELATION_LABELS:
                result.n_model_no_relation += 1
                continue
            floor = self.thresholds.get(label)
            if floor is None or confidence <= floor:
                result.n_below_threshold += 1
                continue
            # direction: earlier step -> later step (premise -> conclusion). the
            # published module stores it the other way (essay convention, claim
            # first). a chain of thought is premises-first, so we flip it. this
            # flip is itself an essay-vs-CoT finding.
            result.relations.append(
                Relation(
                    source=i,
                    target=j,
                    kind=LABEL_TO_RELATION[label],
                    confidence=confidence,
                )
            )
        return result


def estimate_pairs(step_counts: list[int], window: int | None = None) -> dict:
    # given trace step-counts, how many pairs a run would score. used in nb 01.
    per_trace = [len(ARI.make_pairs(n, window)) for n in step_counts]
    total = sum(per_trace)
    return {
        "window": window,
        "n_traces": len(step_counts),
        "mean_steps": sum(step_counts) / max(1, len(step_counts)),
        "mean_pairs_per_trace": total / max(1, len(step_counts)),
        "max_pairs_per_trace": max(per_trace, default=0),
        "total_pairs": total,
    }
