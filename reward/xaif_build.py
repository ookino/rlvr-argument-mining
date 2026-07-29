"""Split a reasoning trace into steps.

ARI wants a list of statements and finds links between them. It doesn't do the
splitting, so we do it here. oAMF has a segmenter for this but it's trained on
essays, and a chain of thought is already written in steps so splitting on
newlines works fine. (see D-001)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
_ANSWER = re.compile(r"(?:the\s+answer\s+is)\s*:?\s*(.+?)\s*\.?\s*$", re.IGNORECASE)

MIN_STEP_CHARS = 15     # shorter than this and it's punctuation or a stray token


@dataclass
class Trace:
    steps: list[str] = field(default_factory=list)
    conclusion: str | None = None
    conclusion_index: int | None = None
    raw: str = ""

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def is_degenerate(self) -> bool:
        # too small to have any structure worth scoring -> score 0
        return self.n_steps < 3 or self.conclusion_index is None


def extract_answer(text: str) -> tuple[str | None, bool]:
    # returns (answer, did_it_match_the_format). keep the flag so we can count
    # format failures separately instead of calling them wrong answers.
    match = _ANSWER.search(text.strip())
    if match:
        return match.group(1).strip(), True
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    return (lines[-1], False) if lines else (None, False)


def split_steps(text: str) -> list[str]:
    # newlines first (that's how the model writes it), sentences as fallback
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 2:
        lines = [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]

    steps = []
    for line in lines:
        line = _NUMBERED.sub("", line).strip()
        if len(line) >= MIN_STEP_CHARS:
            steps.append(line)
    return steps


# a line that just says "The answer is X" is formatting, not reasoning. ARI never
# links the reasoning to it, so it made a bad conclusion node. drop it. (D-008)
_ANSWER_LINE = re.compile(
    r"^\s*(so\s+|thus,?\s+|therefore,?\s+|hence,?\s+|in\s+conclusion,?\s+)?"
    r"(the\s+)?(final\s+)?answer\s*(is|:)",
    re.IGNORECASE,
)


def _is_answer_line(step: str) -> bool:
    return bool(_ANSWER_LINE.match(step))


def build_trace(text: str) -> Trace:
    steps = split_steps(text)
    answer, _ = extract_answer(text)

    # drop trailing "The answer is X" lines. the answer is already extracted
    # above, so nothing is lost, and the conclusion becomes the last real step.
    while steps and _is_answer_line(steps[-1]):
        steps.pop()

    conclusion_index = len(steps) - 1 if steps else None
    return Trace(steps=steps, conclusion=answer, conclusion_index=conclusion_index, raw=text)
