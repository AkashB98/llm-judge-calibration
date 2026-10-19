"""Scorecards: verdicts, determinism, rendering."""

import json

from judge_lab.corpus import load_corpus
from judge_lab.judges import get_judge
from judge_lab.report import grade_all, render_text, scorecard

ITEMS = load_corpus()


def test_calibrated_passes():
    card = scorecard(get_judge("calibrated"), ITEMS)
    assert card["verdict"] == "PASS"
    assert card["reasons"] == []


def test_noisy_fails():
    card = scorecard(get_judge("noisy"), ITEMS)
    assert card["verdict"] == "FAIL"
    assert card["reasons"]


def test_overconfident_warned_not_failed():
    card = scorecard(get_judge("overconfident"), ITEMS)
    assert card["verdict"] == "WARN"
    assert any("difficulty_drift" in r for r in card["reasons"])


def test_position_biased_warned():
    card = scorecard(get_judge("position_biased"), ITEMS)
    assert card["verdict"] == "WARN"
    assert any("position_bias" in r for r in card["reasons"])


def test_scorecard_shape():
    card = scorecard(get_judge("harsh"), ITEMS)
    assert card["judge"] == "harsh"
    assert set(card) == {"judge", "verdict", "reasons", "agreement",
                         "calibration", "per_category", "probes"}
    assert set(card["per_category"]) == {"instruction_following", "factual_qa",
                                         "summarization", "refusal_boundary"}
    assert card["calibration"]["tolerance"] == 0.10


def test_deterministic_json():
    a = json.dumps(scorecard(get_judge("lenient"), ITEMS), sort_keys=True)
    b = json.dumps(scorecard(get_judge("lenient"), ITEMS), sort_keys=True)
    assert a == b


def test_render_text_contains_verdict():
    card = scorecard(get_judge("calibrated"), ITEMS)
    text = render_text(card)
    assert "calibrated" in text and "PASS" in text
    assert "reliability curve" in text


def test_custom_bands():
    card = scorecard(get_judge("calibrated"), ITEMS,
                     bands={"warn": {"ece": 0.0001}})
    assert card["verdict"] == "WARN"


def test_grade_all_length():
    assert len(grade_all(get_judge("calibrated"), ITEMS)) == 48


def test_lenient_positive_residual():
    card = scorecard(get_judge("lenient"), ITEMS)
    assert card["agreement"]["mean_residual"] > 0.03
    card = scorecard(get_judge("harsh"), ITEMS)
    assert card["agreement"]["mean_residual"] < -0.03
