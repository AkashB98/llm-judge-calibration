"""Bias/drift probes catch what headline metrics miss."""

from judge_lab import bias as bias_mod
from judge_lab.corpus import load_corpus
from judge_lab.judges import get_judge
from judge_lab.report import grade_all

ITEMS = load_corpus()


def test_verbosity_flags_only_biased():
    for name in ("calibrated", "lenient", "harsh", "noisy", "rubric_mismatch"):
        p = bias_mod.verbosity_probe(get_judge(name))
        assert not p["flagged"], (name, p)
    p = bias_mod.verbosity_probe(get_judge("verbosity_biased"))
    assert p["flagged"] and p["stat"] > 0.04, p


def test_position_flags_only_biased():
    p = bias_mod.position_probe(get_judge("calibrated"))
    assert not p["flagged"] and p["stat"] == 0.0, p
    p = bias_mod.position_probe(get_judge("position_biased"))
    assert p["flagged"] and p["stat"] == 1.0, p


def test_position_pairs_are_ordered():
    pairs = bias_mod._pair_groups()
    assert len(pairs) >= 10
    for better, worse in pairs:
        assert better["golden"] > worse["golden"]


def test_category_skew_clean_for_calibrated():
    judge = get_judge("calibrated")
    verdicts = grade_all(judge, ITEMS)
    by_cat = {}
    for i, v in zip(ITEMS, verdicts):
        by_cat.setdefault(i["category"], []).append((v["score"], i["golden"]))
    p = bias_mod.category_skew_probe(by_cat)
    assert not p["flagged"], p


def test_category_skew_flags_rubric_mismatch():
    judge = get_judge("rubric_mismatch")
    verdicts = grade_all(judge, ITEMS)
    by_cat = {}
    for i, v in zip(ITEMS, verdicts):
        by_cat.setdefault(i["category"], []).append((v["score"], i["golden"]))
    p = bias_mod.category_skew_probe(by_cat)
    assert p["flagged"], p
    assert "refusal_boundary" in p["detail"]


def test_difficulty_drift_flags_overconfident():
    judge = get_judge("overconfident")
    p = bias_mod.difficulty_drift_probe(ITEMS, grade_all(judge, ITEMS))
    assert p["flagged"], p
    assert p["ece_hard"] > p["ece_easy"]


def test_difficulty_drift_clean_for_calibrated():
    judge = get_judge("calibrated")
    p = bias_mod.difficulty_drift_probe(ITEMS, grade_all(judge, ITEMS))
    assert not p["flagged"], p


def test_run_probes_shape():
    judge = get_judge("calibrated")
    probes = bias_mod.run_probes(judge, ITEMS, grade_all(judge, ITEMS))
    assert len(probes) == 4
    for p in probes:
        assert {"name", "stat", "threshold", "flagged", "detail"} <= set(p)


def test_padded_keeps_golden():
    orig = next(i for i in ITEMS if i["difficulty"] == "easy")
    padded = bias_mod._padded(orig)
    assert padded["golden"] == orig["golden"]
    assert "[" not in padded["response"].replace(orig["response"], "")
    assert len(padded["response"].split()) > len(orig["response"].split()) + 30
