"""Corpus integrity: size, coverage, golden-grade consistency."""

from judge_lab import corpus
from judge_lab import oracle


def test_48_items():
    assert len(corpus.ITEMS) == 48


def test_12_per_category():
    for cat in corpus.CATEGORIES:
        assert sum(1 for i in corpus.ITEMS if i["category"] == cat) == 12, cat


def test_ids_unique():
    ids = [i["id"] for i in corpus.ITEMS]
    assert len(set(ids)) == 48


def test_difficulties_valid():
    diffs = {i["difficulty"] for i in corpus.ITEMS}
    assert diffs <= {"easy", "hard"}
    assert sum(1 for i in corpus.ITEMS if i["difficulty"] == "hard") == 14


def test_goldens_in_range():
    for i in corpus.ITEMS:
        assert 0.05 <= i["golden"] <= 0.97, (i["id"], i["golden"])


def test_golden_matches_oracle():
    for i in corpus.ITEMS:
        grade, _ = oracle.grade(i["response"])
        assert grade == i["golden"], i["id"]


def test_annotations_extracted():
    anns = corpus.annotations_of("x [major:dropped_fact] y [minor:fluff]")
    assert anns == ["major:dropped_fact", "minor:fluff"]
    assert corpus.annotations_of("clean text") == []


def test_load_returns_copies():
    a = corpus.load_corpus()
    a[0]["golden"] = -999
    b = corpus.load_corpus()
    assert b[0]["golden"] != -999


def test_penalty_table_covers_annotations():
    seen = set()
    for i in corpus.ITEMS:
        seen.update(corpus.annotations_of(i["response"]))
    assert seen, "corpus should contain annotated flaws"
    for a in seen:
        assert a in corpus.ANNOTATION_PENALTIES, a


def test_spread_of_grades():
    goldens = [i["golden"] for i in corpus.ITEMS]
    assert min(goldens) <= 0.15      # truly bad responses exist
    assert max(goldens) >= 0.95      # clean responses exist
    assert len(set(goldens)) >= 8    # grade spectrum, not binary
