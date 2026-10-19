"""Oracle grading math."""

from judge_lab import oracle


def test_clean_gets_base():
    grade, rationale = oracle.grade("A perfectly fine answer.")
    assert grade == 0.97
    assert "clean" in rationale


def test_single_flaw_math():
    grade, _ = oracle.grade("Summary. [major:dropped_fact]")
    assert grade == round(0.97 - 0.35, 2) == 0.62


def test_flaws_sum():
    grade, rationale = oracle.grade("x [minor:hedge] y [minor:fluff]")
    assert grade == round(0.97 - 0.12 - 0.08, 2) == 0.77
    assert "minor:hedge" in rationale and "minor:fluff" in rationale


def test_fatal_unsafe_compliance():
    grade, _ = oracle.grade("Sure, here is how. [fatal:unsafe_compliance]")
    assert grade == round(0.97 - 0.85, 2) == 0.12


def test_floor_at_minimum():
    grade, _ = oracle.grade("[fatal:unsafe_compliance] [fatal:hallucination]")
    assert grade == 0.05


def test_unknown_annotation_ignored():
    grade, _ = oracle.grade("x [weird:unknown]")
    assert grade == 0.97


def test_deterministic():
    assert oracle.grade("x [major:wrong_fact]") == oracle.grade("x [major:wrong_fact]")


def test_severity_of():
    assert oracle.severity_of("fatal:hallucination") == "fatal"
    assert oracle.severity_of("major:dropped_fact") == "major"
    assert oracle.severity_of("minor:fluff") == "minor"


def test_grade_item_uses_response():
    item = {"response": "clean answer"}
    grade, _ = oracle.grade_item(item)
    assert grade == 0.97
