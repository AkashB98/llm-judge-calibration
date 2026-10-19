"""Heuristic oracle: the ground-truth grader the judges are measured against.

The oracle applies the documented penalty table (corpus.ANNOTATION_PENALTIES)
to the flaw annotations visible in each response and returns a golden grade
plus a human-readable rationale. It is deterministic by construction -- the
same response always grades the same -- which is what makes it a usable
reference for agreement and calibration measurement.
"""

from .corpus import (
    ANNOTATION_PENALTIES,
    BASE_SCORE,
    MIN_SCORE,
    annotations_of,
)


def grade(response):
    """Grade a response text. Returns (golden_grade, rationale)."""
    anns = annotations_of(response)
    if not anns:
        return BASE_SCORE, "clean: no flaw annotations; base score %.2f" % BASE_SCORE
    parts = []
    total = 0.0
    for a in anns:
        p = ANNOTATION_PENALTIES.get(a, 0.0)
        total += p
        parts.append("%s (-%.2f)" % (a, p))
    score = round(max(MIN_SCORE, BASE_SCORE - total), 2)
    rationale = "flaws: %s => %.2f - %.2f = %.2f" % (
        ", ".join(parts), BASE_SCORE, total, score)
    return score, rationale


def grade_item(item):
    """Grade a corpus item dict. Returns (golden_grade, rationale)."""
    return grade(item["response"])


def severity_of(annotation):
    """Return 'fatal' | 'major' | 'minor' for an annotation key."""
    return annotation.split(":", 1)[0]
