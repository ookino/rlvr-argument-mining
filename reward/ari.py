"""Argument Relation Identification.

Takes a list of reasoning steps and works out which ones support, contradict,
or restate which others. This is the one model in the project we do not train.

WHAT THIS IS, IN PLAIN TERMS
----------------------------
The model is a text classifier. You hand it two sentences and it answers with
one of three labels plus a confidence:

    Inference -> A supports B      (stored as an "RA" link)
    Conflict  -> A contradicts B   (stored as a "CA" link)
    Rephrase  -> A restates B      (stored as an "MA" link)

If the confidence is too low, we record no link at all.

WHY WE DO NOT USE THE oAMF WEB SERVICE
--------------------------------------
The published module (arg-tech/AMF_ARI) wraps this same model in a Flask app
in a Docker container. Running that locally requires Docker, which Google
Colab does not provide. But the model itself is a public Hugging Face
checkpoint, so we load it directly. Same weights, same thresholds, same
pairing logic, no container and no network call in the training loop.

Logged in docs/deviation_log.md as D-002.

THE COST MODEL, WHICH IS THE THING TO UNDERSTAND
------------------------------------------------
The model scores PAIRS of steps, one forward pass each. So the cost of one
trace depends on how many pairs you make from it:

    window = None  ->  every step against every other:  n*(n-1)/2 pairs
    window = 2     ->  neighbours only:                 n-1 pairs

For a 15-step trace that is 105 pairs versus 14. For a 30-step trace, 435
versus 29. This single setting dominates the training-time budget, which is
why it is a config value and not a constant.

All-pairs sees long-range links (a step supporting the conclusion from five
steps earlier), which is exactly what the path-to-conclusion and chain-depth
features are meant to measure. Neighbours-only cannot see them. Measure both
before choosing; do not pick on vibes.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field

# torch and transformers are imported inside the methods that need them, not
# here. That keeps the pairing logic, the features, and the scorer importable
# and testable on a laptop with no training stack installed, which is where
# most of the development happens. Only ARI() itself needs the heavy imports.

logger = logging.getLogger(__name__)

MODEL_ID = "raruidol/ArgumentMining-EN-ARI-AIF-RoBERTa_L"

# Confidence floors, copied verbatim from the published module
# (arg-tech/AMF_ARI, app/ari.py). Support is held to a stricter standard than
# the other two. These are the tool's own numbers and we keep them: retuning a
# supervisor's published thresholds invites the question "did you adjust this
# until it worked?", which is not a question worth answering in a viva.
THRESHOLDS = {
    "Inference": 0.9,
    "Conflict": 0.7,
    "Rephrase": 0.7,
}

# Label as the classifier says it -> node type as AIF calls it.
LABEL_TO_RELATION = {
    "Inference": "RA",
    "Conflict": "CA",
    "Rephrase": "MA",
}


@dataclass
class Relation:
    """One detected link between two steps."""

    source: int          # index of the step doing the supporting/attacking
    target: int          # index of the step being supported/attacked
    kind: str            # RA, CA or MA
    confidence: float


@dataclass
class RelationResult:
    relations: list[Relation] = field(default_factory=list)
    n_pairs_scored: int = 0
    n_below_threshold: int = 0     # pairs the model was not confident enough about


class ARI:
    """Wraps the relation model. Load once, reuse for every trace."""

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = 256,
        encoding: str = "concat",     # matches the published module; see _encode
    ):
        if encoding not in ("concat", "pair"):
            raise ValueError("encoding must be 'concat' or 'pair'")
        self.encoding = encoding

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length

        logger.info("Loading %s onto %s", model_id, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(device).eval()

        self.id2label = self.model.config.id2label
        logger.info("Model labels: %s", self.id2label)

        # Sanity check worth having: if the checkpoint ever changes, or if it
        # turns out to carry a fourth "no relation" class, we want a loud
        # signal rather than silently mis-mapping labels.
        unknown = set(self.id2label.values()) - set(THRESHOLDS)
        if unknown:
            logger.warning(
                "Model exposes labels with no threshold defined: %s. These are "
                "treated as 'no relation'. If one of them is a genuine relation "
                "type, THRESHOLDS needs updating.",
                sorted(unknown),
            )

    @staticmethod
    def make_pairs(n_steps: int, window: int | None = None) -> list[tuple[int, int]]:
        """Which step pairs to score. See the cost model in the module docstring.

        window=None scores everything; window=k scores steps within k of each
        other. Pairs are always ordered (earlier, later) and never duplicated,
        which the published module's sliding-window loop does not guarantee.
        """
        if n_steps < 2:
            return []
        if window is None:
            return list(itertools.combinations(range(n_steps), 2))
        if window < 2:
            raise ValueError("window must be at least 2, or None for all pairs")
        return [
            (i, j)
            for i in range(n_steps)
            for j in range(i + 1, min(i + window, n_steps))
        ]

    def _encode(self, chunk: list[tuple[str, str]]):
        """Turn sentence pairs into model input.

        THIS DETAIL MATTERS MORE THAN IT LOOKS. There are two standard ways to
        hand a model two sentences:

          "concat"  glue them into one string:  "A. B"
          "pair"    pass them as two arguments, so the tokenizer inserts a
                    separator and marks them as separate segments

        They tokenise differently and the model answers differently. The
        published module (arg-tech/AMF_ARI, pipeline_predictions) uses CONCAT:

            sample = data['text'][i] + '. ' + data['text2'][i]

        so that is our default. The file also defines a `tokenize_sequence`
        helper that does it the "pair" way, but never calls it, which suggests
        that route was started and abandoned.

        The option is kept so the two can be compared on a sample of traces.
        If they disagree materially, that is worth a sentence in the write-up;
        if they agree, that is a cheap validity check to have run.
        """
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
        """Run the model over sentence pairs, returning (label, confidence)."""
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
        """Find the relations between a trace's reasoning steps.

        `steps` is the trace already split into statements; we own that split
        (see reward/xaif_build.py) rather than using an upstream segmenter.
        """
        index_pairs = self.make_pairs(len(steps), window)
        if not index_pairs:
            return RelationResult()

        text_pairs = [(steps[i], steps[j]) for i, j in index_pairs]
        predictions = self._classify(text_pairs)

        result = RelationResult(n_pairs_scored=len(index_pairs))
        for (i, j), (label, confidence) in zip(index_pairs, predictions):
            floor = THRESHOLDS.get(label)
            # Strictly greater, matching the published module's `> 0.9` rather
            # than `>= 0.9`. Immaterial in practice, but there is no reason to
            # differ from the reference implementation.
            if floor is None or confidence <= floor:
                result.n_below_threshold += 1
                continue
            # Direction follows the published module: the SECOND member of the
            # pair is the source, the first is the target. Verify this against
            # a hand-built example before trusting any path-based feature.
            result.relations.append(
                Relation(
                    source=j,
                    target=i,
                    kind=LABEL_TO_RELATION[label],
                    confidence=confidence,
                )
            )
        return result


def estimate_pairs(step_counts: list[int], window: int | None = None) -> dict:
    """Cost projection helper for the throughput measurement.

    Feed it the step counts of a sample of real traces and it tells you how
    many forward passes a training run would cost. Used in notebook 01.
    """
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
