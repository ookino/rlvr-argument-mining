"""Turning a raw reasoning trace into a list of steps, and then into a graph.

WHY WE DO THIS OURSELVES
------------------------
The relation model expects a list of statements and finds the links between
them. It does not split text into statements; that is a separate tool in the
oAMF toolkit (a "segmenter"), built for debate transcripts and student essays.

Doing the split ourselves is a genuine improvement, not a shortcut. A chain of
thought is already written in steps, so splitting it is easy and reliable,
whereas a segmenter trained on essays would guess. More importantly, it means
any structure we measure is attributable to ONE model's errors instead of two
models' errors compounding, which makes the characterisation study cleaner.

Logged in docs/deviation_log.md as D-001.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Steps are usually newline-separated or numbered. Sentence splitting is the
# fallback for traces written as one paragraph.
_NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# The trace is instructed to end with this. It is the conclusion, and it is
# excluded from the reasoning steps and tracked separately.
_ANSWER = re.compile(r"(?:the\s+answer\s+is)\s*:?\s*(.+?)\s*\.?\s*$", re.IGNORECASE)

MIN_STEP_CHARS = 15     # below this a "step" is punctuation or a stray token


@dataclass
class Trace:
    """A reasoning trace, split up and ready for relation identification."""

    steps: list[str] = field(default_factory=list)
    conclusion: str | None = None
    conclusion_index: int | None = None   # position of the conclusion in steps
    raw: str = ""

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def is_degenerate(self) -> bool:
        """Too small to have structure worth scoring.

        Fewer than three steps cannot form a chain, and with no conclusion
        there is nothing for the other steps to connect to. Both cases score
        zero rather than being scored on a graph that was never there.
        """
        return self.n_steps < 3 or self.conclusion_index is None


def extract_answer(text: str) -> tuple[str | None, bool]:
    """Pull the final answer out of a trace.

    Returns (answer, matched_forced_format). A failure to match is recorded,
    never silently scored as a wrong answer, because those are different
    things and conflating them corrupts the accuracy numbers.
    """
    match = _ANSWER.search(text.strip())
    if match:
        return match.group(1).strip(), True

    # Fallback: last non-empty line. Flagged so the count is reportable.
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return (lines[-1], False) if lines else (None, False)


def split_steps(text: str) -> list[str]:
    """Split a trace into reasoning steps.

    Newlines first, since that is how models actually write chains of thought.
    Falls back to sentence splitting for single-paragraph traces.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    if len(lines) < 2:
        lines = [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]

    steps = []
    for line in lines:
        line = _NUMBERED.sub("", line).strip()
        if len(line) >= MIN_STEP_CHARS:
            steps.append(line)
    return steps


def build_trace(text: str) -> Trace:
    """Full pipeline: raw model output to a Trace ready for scoring."""
    steps = split_steps(text)
    answer, _ = extract_answer(text)

    conclusion_index = None
    if answer and steps:
        # The conclusion is normally the last step. Match on the answer text
        # so a trailing "So the answer is B." is identified rather than assumed.
        for idx in range(len(steps) - 1, -1, -1):
            if answer.lower() in steps[idx].lower():
                conclusion_index = idx
                break

    return Trace(
        steps=steps,
        conclusion=answer,
        conclusion_index=conclusion_index,
        raw=text,
    )
