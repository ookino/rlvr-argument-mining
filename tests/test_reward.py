"""Tests for the scoring pipeline.

These are cheap to write and they are evidence of rigour in the write-up. They
also catch the failure that would hurt most: a scoring function that quietly
rewards padding.

No model is loaded here. Relations are hand-built, so these run in under a
second on a laptop with no graphics card.
"""

import pytest

from reward.ari import ARI, Relation, RelationResult
from reward.features import compute
from reward.scorer import Calibration, length_penalty, structural_score
from reward.xaif_build import build_trace, extract_answer, split_steps

# --- pairing, which is the whole cost model ---------------------------------


def test_all_pairs_is_quadratic():
    assert len(ARI.make_pairs(15, window=None)) == 105      # 15*14/2
    assert len(ARI.make_pairs(30, window=None)) == 435


def test_neighbours_only_is_linear():
    assert len(ARI.make_pairs(15, window=2)) == 14
    assert len(ARI.make_pairs(30, window=2)) == 29


def test_pairs_are_never_duplicated():
    """The published module's sliding window can repeat pairs; ours must not."""
    for window in (None, 2, 3, 5):
        pairs = ARI.make_pairs(12, window=window)
        assert len(pairs) == len(set(pairs))


def test_pairs_are_always_ordered_earlier_then_later():
    assert all(i < j for i, j in ARI.make_pairs(10, window=4))


@pytest.mark.parametrize("n", [0, 1])
def test_too_few_steps_gives_no_pairs(n):
    assert ARI.make_pairs(n) == []


# --- splitting traces --------------------------------------------------------


def test_split_on_newlines():
    text = "First we consider the options.\nSecond we rule one out.\nThe answer is B."
    assert len(split_steps(text)) == 3


def test_split_strips_numbering():
    steps = split_steps("1. First we consider this.\n2. Then we rule that out.")
    assert steps[0].startswith("First")


def test_split_falls_back_to_sentences():
    text = "First we consider the options. Then we rule one out. The answer is B."
    assert len(split_steps(text)) == 3


def test_short_fragments_are_dropped():
    assert "ok." not in split_steps("ok.\nThis is a real reasoning step here.")


def test_extract_answer_forced_format():
    answer, matched = extract_answer("Reasoning goes here.\nThe answer is B.")
    assert (answer, matched) == ("B", True)


def test_extract_answer_failure_is_flagged_not_guessed():
    """Extraction failures must be countable, never silently marked wrong."""
    _, matched = extract_answer("Some reasoning with no final answer line")
    assert matched is False


def test_conclusion_skips_the_answer_line():
    # "The answer is B." is an announcement, not reasoning, so it is dropped
    # from the steps and the conclusion is the real last reasoning step.
    trace = build_trace(
        "We start by ruling out option A here.\n"
        "That leaves us with only option B remaining.\n"
        "The answer is B."
    )
    assert trace.conclusion == "B"
    assert trace.n_steps == 2                       # answer line dropped
    assert trace.conclusion_index == 1              # the real conclusion
    assert "answer is" not in trace.steps[trace.conclusion_index].lower()


# --- degenerate traces -------------------------------------------------------


def test_two_step_trace_is_degenerate():
    trace = build_trace("A single reasoning step here.\nThe answer is B.")
    assert trace.is_degenerate


def test_degenerate_trace_scores_zero_on_everything():
    trace = build_trace("Too short to have any structure at all.")
    features = compute(trace, RelationResult())
    assert all(v == 0.0 for v in features.as_dict().values())


def test_empty_trace_does_not_crash():
    compute(build_trace(""), RelationResult())


# --- the features ------------------------------------------------------------


def _chain_trace():
    """Three reasoning steps in a clean supporting chain, plus an answer line
    that gets dropped. Premises 0 and 1 support the conclusion (step 2)."""
    trace = build_trace(
        "All birds in this garden are known to be ravens.\n"
        "Every raven that we have observed is black in colour.\n"
        "Therefore all the birds in this garden must be black.\n"
        "The answer is black."
    )
    # After the answer line is dropped, steps are 0,1,2 and 2 is the conclusion.
    relations = RelationResult(
        relations=[
            Relation(source=0, target=2, kind="RA", confidence=0.95),
            Relation(source=1, target=2, kind="RA", confidence=0.95),
        ],
        n_pairs_scored=3,
    )
    return trace, relations


def test_clean_chain_scores_well():
    features = compute(*_chain_trace())
    assert features.attachment == 1.0
    assert features.connectivity == 1.0
    assert features.support_depth > 0.0


def test_disconnected_trace_scores_worse_than_connected():
    trace, relations = _chain_trace()
    connected = compute(trace, relations)
    floating = compute(trace, RelationResult(relations=[], n_pairs_scored=6))
    assert floating.attachment < connected.attachment
    assert floating.connectivity < connected.connectivity


def test_restatement_is_detected_as_padding():
    """Saying the same thing twice is the obvious way to game this reward."""
    trace, relations = _chain_trace()
    padded = RelationResult(
        relations=relations.relations + [Relation(0, 1, "MA", 0.9)],
        n_pairs_scored=6,
    )
    assert compute(trace, padded).restatement_rate > 0.0


# --- the scorer --------------------------------------------------------------


def _calibration(**overrides):
    base = dict(
        weights={
            "support_density": 1.0,
            "attachment": 1.0,
            "connectivity": 1.0,
            "support_depth": 1.0,
            "conflict_rate": 1.0,
            "restatement_rate": 1.0,
        },
        length_target=100.0,
    )
    base.update(overrides)
    return Calibration(**base)


def test_scorer_refuses_unset_weights(tmp_path):
    """The guard that stops a result being invalidated by hand-set weights."""
    path = tmp_path / "cal.yaml"
    path.write_text("weights:\n  attachment: null\nlength_target: 100\n")
    with pytest.raises(ValueError, match="weights"):
        Calibration.load(path)


def test_scorer_refuses_unset_length_target(tmp_path):
    path = tmp_path / "cal.yaml"
    path.write_text("weights:\n  attachment: 1.0\nlength_target: null\n")
    with pytest.raises(ValueError, match="length_target"):
        Calibration.load(path)


def test_real_calibration_file_is_still_frozen():
    """Guards the guard. Flip this test when calibrate.py has run in week 2.

    Until then, an unfrozen calibration file must be un-trainable. When you
    freeze it, this test failing is the intended signal that you passed the
    gate; invert it then.
    """
    with pytest.raises(ValueError):
        Calibration.load()


def test_length_penalty_only_bites_past_the_target():
    assert length_penalty(50, 100) == 1.0
    assert length_penalty(100, 100) == 1.0
    assert length_penalty(200, 100) == pytest.approx(0.5)


def test_padding_the_trace_cannot_raise_the_score():
    """The core anti-gaming property. If this fails, the reward is exploitable."""
    trace, relations = _chain_trace()
    features = compute(trace, relations)
    cal = _calibration()

    short = structural_score(features, n_tokens=80, calibration=cal)
    long = structural_score(features, n_tokens=400, calibration=cal)
    assert long < short


def test_score_stays_in_range():
    features = compute(*_chain_trace())
    score = structural_score(features, n_tokens=50, calibration=_calibration())
    assert 0.0 <= score <= 1.0


def test_ablation_drops_the_named_features():
    """The ablation run must actually change the score, or it proves nothing."""
    trace, relations = _chain_trace()
    padded = RelationResult(
        relations=relations.relations + [Relation(0, 1, "MA", 0.9)],
        n_pairs_scored=6,
    )
    features = compute(trace, padded)
    cal = _calibration()

    full = structural_score(features, 50, cal)
    ablated = structural_score(features, 50, cal, use_restatement=False)
    assert full != ablated
