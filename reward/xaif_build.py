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


def _split_think(text: str) -> tuple[str, str]:
    # A reasoning model (Qwen3 in thinking mode) puts its working inside
    # <think>...</think> and the final answer after it. We mine the reasoning
    # (inside think) and read the answer from the part after. A plain instruct
    # model has no tags, so the whole thing is both. Returns (reasoning, answer_part).
    lower = text.lower()
    if "<think>" not in lower:
        return text, text
    start = lower.index("<think>") + len("<think>")
    end = lower.find("</think>")
    if end == -1:                       # model got cut off before closing the tag
        reasoning = text[start:].strip()
        return reasoning, reasoning
    reasoning = text[start:end].strip()
    after = text[end + len("</think>"):].strip()
    return reasoning, (after or reasoning)


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
    # read the answer from AFTER the thinking, so "the answer is" inside the
    # reasoning doesn't get picked up by mistake.
    _, answer_part = _split_think(text)
    answer_part = answer_part.strip()
    match = _ANSWER.search(answer_part)
    if match:
        return match.group(1).strip(), True
    lines = [ln.strip() for ln in answer_part.splitlines() if ln.strip()]
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


def to_sentences(steps: list[str]) -> list[str]:
    # One sentence per step. ARI wants single claims, but a reasoning model's
    # newlines separate paragraphs, so newline steps come out ~3 sentences long
    # and ARI reads them as restatements of each other. (D-014)
    out: list[str] = []
    for step in steps:
        for part in _SENTENCE.split(step):
            part = _NUMBERED.sub("", part).strip()
            if len(part) >= MIN_STEP_CHARS:
                out.append(part)
    return out


# a line that just says "The answer is X" is formatting, not reasoning. ARI never
# links the reasoning to it, so it made a bad conclusion node. drop it. (D-008)
_ANSWER_LINE = re.compile(
    r"^\s*(so\s+|thus,?\s+|therefore,?\s+|hence,?\s+|in\s+conclusion,?\s+)?"
    r"(the\s+)?(final\s+)?answer\s*(is|:)",
    re.IGNORECASE,
)


def _is_answer_line(step: str) -> bool:
    return bool(_ANSWER_LINE.match(step))


# reasoning models state their verdict, then keep second-guessing:
# "But wait, let me reconsider...", "Alternatively, suppose...". Like the answer
# line, that trailing chatter isn't the argumentative conclusion, so it must not
# become the conclusion node - the conclusion should land on the verdict. We drop
# these only when they TRAIL (mid-trace reconsidering is real reasoning and
# stays). Deliberately conservative: bare "Hmm"/"Actually" openers are excluded
# because they usually carry real content. (D-011)
_REFLECTION_LINE = re.compile(
    r"^\s*(but\s+)?"
    r"(wait\b|hold\s+on\b|alternatively\b|"
    r"let\s+me\s+(re-?check|check\s+again|verify|re-?consider|re-?examine|"
    r"double[-\s]?check|confirm|make\s+sure|think\s+again|go\s+(back|over)))",
    re.IGNORECASE,
)


def _is_trailing_noise(step: str) -> bool:
    # a step that shouldn't be allowed to be the conclusion: an answer
    # announcement or trailing self-doubt.
    return _is_answer_line(step) or bool(_REFLECTION_LINE.match(step))


DEDUPE_THRESHOLD = 0.6   # a step sharing >= this fraction of its words with an
                         # earlier kept step is treated as repetition and dropped


def _word_set(step: str) -> set:
    return set(w for w in "".join(c.lower() if c.isalnum() else " " for c in step).split())


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def dedupe_steps(steps: list[str], threshold: float = DEDUPE_THRESHOLD) -> list[str]:
    # Reasoning models repeat themselves heavily - re-deriving the same chain,
    # re-listing the premises. Drop a step if it shares most of its words with an
    # earlier kept step, so the relation model sees distinct claims instead of
    # padding. Lexical only: a first-pass cleanup, not semantic. (Arm B, D-010)
    kept, kept_sets = [], []
    for s in steps:
        ss = _word_set(s)
        if any(_jaccard(ss, k) >= threshold for k in kept_sets):
            continue
        kept.append(s)
        kept_sets.append(ss)
    return kept


def build_trace(text: str, dedupe: bool = False,
                sentence_level: bool = True) -> Trace:
    # mine the reasoning (inside <think> for a reasoning model; the whole text
    # for a plain instruct model), read the answer from after it. dedupe=True is
    # Arm B: collapse the model's repetition before scoring (D-010).
    # sentence_level=True breaks paragraph-steps into single claims, the unit
    # ARI is trained on (D-014). Pass False to reproduce pre-D-014 scoring.
    reasoning, _ = _split_think(text)
    steps = split_steps(reasoning)
    # drop trailing noise BEFORE splitting, so a whole "But wait..." paragraph
    # goes rather than just its last sentence (D-011)
    while steps and _is_trailing_noise(steps[-1]):
        steps.pop()
    if sentence_level:
        steps = to_sentences(steps)
    if dedupe:
        steps = dedupe_steps(steps)
    answer, _ = extract_answer(text)

    # drop trailing answer-announcement and self-doubt lines, so the conclusion
    # lands on the verdict rather than a "But wait..." afterthought. the answer
    # is already extracted above, so nothing is lost. (D-008, D-011)
    while steps and _is_trailing_noise(steps[-1]):
        steps.pop()

    conclusion_index = len(steps) - 1 if steps else None
    return Trace(steps=steps, conclusion=answer, conclusion_index=conclusion_index, raw=text)
