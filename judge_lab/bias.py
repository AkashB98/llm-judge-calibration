"""Bias / drift probes: the failure modes headline metrics miss.

  verbosity_probe    controlled length pairs (same golden grade, one padded
                   with fluff): does the judge reward longer responses?
  position_probe     pairwise (better, worse) shown in both orders: does the
                   winner flip with presentation order?
  category_skew      per-category mean(score - golden): is one category
                   systematically mis-graded while others are fine?
  difficulty_drift   ECE on easy vs hard items: does calibration collapse
                   exactly where the judge is most uncertain?

Each probe returns {name, stat, threshold, flagged, detail}. Deterministic.
"""

from . import metrics
from .corpus import load_corpus

FILLER_SENTENCES = [
    "This answer aims to be as complete and helpful as possible for the reader.",
    "Additional context is provided here to ensure nothing important is missed.",
    "The explanation above covers the key points in a thorough and clear manner.",
    "Readers who want more detail can review the product manual for background.",
]

VERBOSITY_THRESHOLD = 0.04   # mean padded-minus-original score gap
POSITION_THRESHOLD = 0.5     # fraction of order-flipped winners
SKEW_THRESHOLD = 0.10        # per-category |mean residual|
SKEW_MARGIN = 0.05           # ...and clear of the next-worst category
DRIFT_THRESHOLD = 0.08       # hard ECE minus easy ECE


def _padded(item):
    dup = dict(item)
    dup["id"] = item["id"] + ":padded"
    dup["response"] = item["response"] + " " + " ".join(FILLER_SENTENCES)
    return dup


def verbosity_probe(judge, n_pairs=14):
    """Same content quality, different length. Unbiased judges score ~equal.

    Uses mid-range golden grades (0.25-0.90) so the word bonus has headroom
    below the score ceiling -- near-perfect items would clamp it away.
    """
    cands = [i for i in load_corpus()
             if i["difficulty"] == "easy" and 0.25 <= i["golden"] <= 0.90]
    pairs = [(i, _padded(i)) for i in sorted(cands, key=lambda x: x["id"])[:n_pairs]]
    assert len(pairs) == n_pairs, "corpus changed: expected %d probe pairs" % n_pairs
    diffs = [judge.grade(p)["score"] - judge.grade(o)["score"] for o, p in pairs]
    stat = metrics.mean(diffs)
    return {
        "name": "verbosity_bias",
        "stat": round(stat, 4),
        "threshold": VERBOSITY_THRESHOLD,
        "flagged": stat > VERBOSITY_THRESHOLD,
        "detail": "mean(padded - original) over %d length-controlled pairs; "
                  "positive = rewards verbosity" % len(pairs),
    }


def _pair_groups():
    groups = {}
    for item in load_corpus():
        groups.setdefault(item["prompt"], []).append(item)
    pairs = []
    for prompt in sorted(groups):
        items = sorted(groups[prompt], key=lambda x: x["golden"])
        if len(items) >= 2 and items[0]["golden"] < items[-1]["golden"]:
            pairs.append((items[-1], items[0]))  # (better, worse)
    return pairs


def position_probe(judge, n_pairs=10):
    """Present (better, worse) in both orders. Winner should not flip."""
    pairs = _pair_groups()[:n_pairs]
    flips = 0
    for better, worse in pairs:
        w1 = judge.compare(better, worse)["winner"]   # better presented first
        w2 = judge.compare(worse, better)["winner"]   # worse presented first
        # content-consistent outcomes: (a, b). A flip means order decided it.
        if (w1, w2) != ("a", "b"):
            flips += 1
    stat = flips / len(pairs)
    return {
        "name": "position_bias",
        "stat": round(stat, 4),
        "threshold": POSITION_THRESHOLD,
        "flagged": stat > POSITION_THRESHOLD,
        "detail": "%d/%d pairs changed winner when order swapped" % (flips, len(pairs)),
    }


def category_skew_probe(verdicts_by_category):
    """verdicts_by_category: {category: [(score, golden), ...]}."""
    skews = {c: metrics.mean_residual([s for s, _ in v], [g for _, g in v])
             for c, v in verdicts_by_category.items()}
    ranked = sorted(skews.items(), key=lambda kv: abs(kv[1]), reverse=True)
    worst_cat, worst = ranked[0]
    second = abs(ranked[1][1]) if len(ranked) > 1 else 0.0
    flagged = abs(worst) > SKEW_THRESHOLD and (abs(worst) - second) > SKEW_MARGIN
    return {
        "name": "category_skew",
        "stat": round(worst, 4),
        "threshold": SKEW_THRESHOLD,
        "flagged": flagged,
        "detail": "worst category %r skew %+.3f (next %+.3f); per-category: %s" % (
            worst_cat, worst, ranked[1][1] if len(ranked) > 1 else 0.0,
            ", ".join("%s=%+.3f" % (c, s) for c, s in sorted(skews.items()))),
    }


def difficulty_drift_probe(items, verdicts):
    """Does calibration collapse on hard items?"""
    easy = [(v["confidence"], c) for i, v, c in zip(items, verdicts, _correct(items, verdicts))
            if i["difficulty"] == "easy"]
    hard = [(v["confidence"], c) for i, v, c in zip(items, verdicts, _correct(items, verdicts))
            if i["difficulty"] == "hard"]
    ece_easy = metrics.ece([c for c, _ in easy], [k for _, k in easy])
    ece_hard = metrics.ece([c for c, _ in hard], [k for _, k in hard])
    drift = ece_hard - ece_easy
    return {
        "name": "difficulty_drift",
        "stat": round(drift, 4),
        "threshold": DRIFT_THRESHOLD,
        "flagged": drift > DRIFT_THRESHOLD,
        "detail": "ECE hard %.3f vs easy %.3f" % (ece_hard, ece_easy),
        "ece_easy": round(ece_easy, 4),
        "ece_hard": round(ece_hard, 4),
    }


def _correct(items, verdicts):
    goldens = [i["golden"] for i in items]
    scores = [v["score"] for v in verdicts]
    return metrics.correctness(scores, goldens)


def run_probes(judge, items, verdicts):
    """Run all probes. Returns list of probe dicts."""
    by_cat = {}
    for item, v in zip(items, verdicts):
        by_cat.setdefault(item["category"], []).append((v["score"], item["golden"]))
    return [
        verbosity_probe(judge),
        position_probe(judge),
        category_skew_probe(by_cat),
        difficulty_drift_probe(items, verdicts),
    ]
