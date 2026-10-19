"""Judge scorecards: agreement + calibration + bias probes -> verdict.

Verdict bands (override via CLI flags):
  FAIL  ece > 0.15 or mce > 0.25 or mae > 0.20 or qwk < 0.50
  WARN  ece > 0.08 or mce > 0.15 or mae > 0.12 or any probe flagged
  PASS  otherwise

A miscalibrated judge is a broken thermometer: every eval that trusts it
inherits its bias. The scorecard exists to answer "can I trust this grader?"
before the grader scores anything real.
"""

from . import bias as bias_mod
from . import metrics
from .corpus import load_corpus

FAIL_BANDS = {"ece": 0.15, "mce": 0.25, "mae": 0.20, "qwk": 0.50}
WARN_BANDS = {"ece": 0.08, "mce": 0.15, "mae": 0.12}


def grade_all(judge, items):
    return [judge.grade(i) for i in items]


def scorecard(judge, items=None, verdicts=None, bands=None):
    items = items if items is not None else load_corpus()
    verdicts = verdicts if verdicts is not None else grade_all(judge, items)
    fail = dict(FAIL_BANDS)
    warn = dict(WARN_BANDS)
    if bands:
        fail.update(bands.get("fail", {}))
        warn.update(bands.get("warn", {}))

    scores = [v["score"] for v in verdicts]
    confs = [v["confidence"] for v in verdicts]
    goldens = [i["golden"] for i in items]
    correct = metrics.correctness(scores, goldens)

    agreement = {
        "n": len(items),
        "spearman": round(metrics.spearman(scores, goldens), 4),
        "mae": round(metrics.mae(scores, goldens), 4),
        "rmse": round(metrics.rmse(scores, goldens), 4),
        "qwk": round(metrics.qwk(scores, goldens), 4),
        "mean_residual": round(metrics.mean_residual(scores, goldens), 4),
    }
    calibration = {
        "tolerance": metrics.TOLERANCE,
        "ece": round(metrics.ece(confs, correct), 4),
        "mce": round(metrics.mce(confs, correct), 4),
        "brier": round(metrics.brier(confs, correct), 4),
        "accuracy": round(metrics.mean(correct), 4),
        "bins": metrics.reliability_bins(confs, correct),
    }
    per_category = {}
    for cat in sorted({i["category"] for i in items}):
        idx = [k for k, i in enumerate(items) if i["category"] == cat]
        cs = [scores[k] for k in idx]
        gs = [goldens[k] for k in idx]
        per_category[cat] = {
            "n": len(idx),
            "mae": round(metrics.mae(cs, gs), 4),
            "mean_residual": round(metrics.mean_residual(cs, gs), 4),
            "spearman": round(metrics.spearman(cs, gs), 4),
        }

    probes = bias_mod.run_probes(judge, items, verdicts)

    reasons = []
    verdict = "PASS"
    if (calibration["ece"] > fail["ece"] or calibration["mce"] > fail["mce"]
            or agreement["mae"] > fail["mae"] or agreement["qwk"] < fail["qwk"]):
        verdict = "FAIL"
    if verdict == "FAIL":
        if calibration["ece"] > fail["ece"]:
            reasons.append("ECE %.3f > %.2f: confidence does not match accuracy"
                           % (calibration["ece"], fail["ece"]))
        if calibration["mce"] > fail["mce"]:
            reasons.append("MCE %.3f > %.2f: a confidence band is badly off"
                           % (calibration["mce"], fail["mce"]))
        if agreement["mae"] > fail["mae"]:
            reasons.append("MAE %.3f > %.2f: grades far from reference"
                           % (agreement["mae"], fail["mae"]))
        if agreement["qwk"] < fail["qwk"]:
            reasons.append("QWK %.3f < %.2f: ranking barely beats chance"
                           % (agreement["qwk"], fail["qwk"]))
    if verdict == "PASS" and (
            calibration["ece"] > warn["ece"] or calibration["mce"] > warn["mce"]
            or agreement["mae"] > warn["mae"]):
        verdict = "WARN"
        if calibration["ece"] > warn["ece"]:
            reasons.append("ECE %.3f > %.2f" % (calibration["ece"], warn["ece"]))
        if calibration["mce"] > warn["mce"]:
            reasons.append("MCE %.3f > %.2f" % (calibration["mce"], warn["mce"]))
        if agreement["mae"] > warn["mae"]:
            reasons.append("MAE %.3f > %.2f" % (agreement["mae"], warn["mae"]))
    for p in probes:
        if p["flagged"]:
            if verdict == "PASS":
                verdict = "WARN"
            reasons.append("probe %s flagged: %s (stat %.3f > %.2f)" % (
                p["name"], p["detail"], p["stat"], p["threshold"]))

    return {
        "judge": judge.name,
        "verdict": verdict,
        "reasons": reasons,
        "agreement": agreement,
        "calibration": calibration,
        "per_category": per_category,
        "probes": probes,
    }


def render_text(card):
    """Human-readable scorecard with an ASCII reliability curve."""
    L = []
    L.append("judge: %s  verdict: %s" % (card["judge"], card["verdict"]))
    for r in card["reasons"]:
        L.append("  ! %s" % r)
    a, c = card["agreement"], card["calibration"]
    L.append("agreement  n=%d  spearman=%.3f  mae=%.3f  qwk=%.3f  mean_resid=%+.3f" % (
        a["n"], a["spearman"], a["mae"], a["qwk"], a["mean_residual"]))
    L.append("calibration  ece=%.3f  mce=%.3f  brier=%.4f  accuracy=%.3f" % (
        c["ece"], c["mce"], c["brier"], c["accuracy"]))
    L.append("reliability curve (confidence bin -> accuracy):")
    for b in c["bins"]:
        bar = "#" * int(40 * b["accuracy"])
        L.append("  %s acc=%.2f conf=%.2f n=%3d |%s" % (
            b["bin"], b["accuracy"], b["avg_confidence"], b["n"], bar))
    L.append("per-category mae:")
    for cat, p in card["per_category"].items():
        L.append("  %-22s mae=%.3f resid=%+.3f" % (cat, p["mae"], p["mean_residual"]))
    L.append("probes:")
    for p in card["probes"]:
        mark = "FLAG" if p["flagged"] else "ok  "
        L.append("  [%s] %-16s stat=%.3f  %s" % (mark, p["name"], p["stat"], p["detail"]))
    return "\n".join(L)
