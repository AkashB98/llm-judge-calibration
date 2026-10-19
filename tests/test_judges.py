"""Judge profiles: determinism, bounds, and characteristic distortions."""

import os

from judge_lab import judges
from judge_lab.corpus import load_corpus

ITEMS = load_corpus()
NAMES = list(judges.JUDGES)


def test_all_judges_grade_all_items():
    for name in NAMES:
        judge = judges.get_judge(name)
        for item in ITEMS:
            v = judge.grade(item)
            assert 0.0 <= v["score"] <= 0.995, (name, item["id"])
            assert 0.5 <= v["confidence"] <= 0.995, (name, item["id"])
            assert v["judge"] == name
            assert isinstance(v["rationale"], str) and v["rationale"]


def test_deterministic_across_instances():
    for name in NAMES:
        a = judges.get_judge(name).grade(ITEMS[0])
        b = judges.get_judge(name).grade(ITEMS[0])
        assert a == b, name


def test_calibrated_close_to_golden():
    judge = judges.get_judge("calibrated")
    errs = [abs(judge.grade(i)["score"] - i["golden"]) for i in ITEMS]
    assert sum(errs) / len(errs) < 0.08


def test_overconfident_same_scores_higher_conf():
    cal = judges.get_judge("calibrated")
    over = judges.get_judge("overconfident")
    for item in ITEMS:
        assert over.grade(item)["score"] == cal.grade(item)["score"]
        assert over.grade(item)["confidence"] >= cal.grade(item)["confidence"]


def test_underconfident_lower_conf():
    cal = judges.get_judge("calibrated")
    under = judges.get_judge("underconfident")
    assert all(under.grade(i)["confidence"] <= cal.grade(i)["confidence"] for i in ITEMS)


def test_lenient_shifts_up_harsh_down():
    cal = judges.get_judge("calibrated")
    lenient = judges.get_judge("lenient")
    harsh = judges.get_judge("harsh")
    dl = [lenient.grade(i)["score"] - cal.grade(i)["score"] for i in ITEMS]
    dh = [harsh.grade(i)["score"] - cal.grade(i)["score"] for i in ITEMS]
    assert sum(dl) / len(dl) > 0.03
    assert sum(dh) / len(dh) < -0.03


def test_noisy_differs_from_calibrated():
    cal = judges.get_judge("calibrated")
    noisy = judges.get_judge("noisy")
    diffs = [abs(noisy.grade(i)["score"] - cal.grade(i)["score"]) for i in ITEMS]
    assert sum(diffs) / len(diffs) > 0.1


def test_verbosity_rewards_length():
    judge = judges.get_judge("verbosity_biased")
    short = {"id": "t1", "prompt": "p", "response": "Brief answer.", "difficulty": "easy"}
    long = {"id": "t2", "prompt": "p",
            "response": "Brief answer. " + "Extra words. " * 60, "difficulty": "easy"}
    assert judge.grade(long)["score"] > judge.grade(short)["score"]


def test_position_biased_always_picks_first():
    judge = judges.get_judge("position_biased")
    a, b = ITEMS[0], ITEMS[1]
    assert judge.compare(a, b)["winner"] == "a"
    assert judge.compare(b, a)["winner"] == "a"


def test_calibrated_compare_picks_better():
    judge = judges.get_judge("calibrated")
    better = max(ITEMS, key=lambda i: i["golden"])
    worse = min(ITEMS, key=lambda i: i["golden"])
    assert judge.compare(better, worse)["winner"] == "a"
    assert judge.compare(worse, better)["winner"] == "b"


def test_compare_confidence_increases_with_gap():
    judge = judges.get_judge("calibrated")
    close = judge.compare(ITEMS[0], ITEMS[2])
    far = judge.compare(max(ITEMS, key=lambda i: i["golden"]),
                        min(ITEMS, key=lambda i: i["golden"]))
    assert far["confidence"] >= close["confidence"]


def test_rubric_mismatch_underpunishes_fatal():
    judge = judges.get_judge("rubric_mismatch")
    fatal = next(i for i in ITEMS if "[fatal:unsafe_compliance]" in i["response"])
    v = judge.grade(fatal)
    assert v["score"] - fatal["golden"] > 0.2  # scores it far too highly


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_unknown_judge_raises():
    expect_value_error(lambda: judges.get_judge("nonexistent"))


def test_llm_judge_requires_env():
    saved = os.environ.pop("JUDGELAB_LLM", None)
    try:
        expect_value_error(lambda: judges.get_judge("llm"))
    finally:
        if saved is not None:
            os.environ["JUDGELAB_LLM"] = saved


def test_list_judges_shape():
    lst = judges.list_judges()
    assert len(lst) == 9
    assert all({"name", "description"} <= set(j) for j in lst)
