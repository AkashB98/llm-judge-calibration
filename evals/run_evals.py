#!/usr/bin/env python3
"""Golden evals: pinned agreement/calibration/probe outcomes.

Deterministic by construction (seeded judges, fixed corpus). The report is
written to evals/eval_report.json; its md5 must be byte-identical across runs.
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from judge_lab import bias as bias_mod
from judge_lab import oracle
from judge_lab.corpus import load_corpus
from judge_lab.judges import get_judge
from judge_lab.report import grade_all, scorecard

RESULTS = []


def check(name, passed, **detail):
    RESULTS.append({"name": name, "passed": bool(passed), "detail": detail})
    print(("PASS " if passed else "FAIL ") + name +
          ("  " + json.dumps(detail, sort_keys=True) if detail else ""))
    return passed


def main():
    items = load_corpus()
    ok = True

    ok &= check("corpus_size",
                len(items) == 48 and
                all(sum(1 for i in items if i["category"] == c) == 12
                    for c in ("instruction_following", "factual_qa",
                              "summarization", "refusal_boundary")),
                n=len(items))

    ok &= check("oracle_golden_consistency",
                all(oracle.grade(i["response"])[0] == i["golden"] for i in items))

    cal = scorecard(get_judge("calibrated"), items)
    ok &= check("calibrated_agreement",
                cal["agreement"]["spearman"] > 0.90 and cal["agreement"]["mae"] < 0.06,
                spearman=cal["agreement"]["spearman"], mae=cal["agreement"]["mae"])
    ok &= check("calibrated_calibration",
                cal["verdict"] == "PASS" and cal["calibration"]["ece"] < 0.03,
                ece=cal["calibration"]["ece"], verdict=cal["verdict"])

    over = scorecard(get_judge("overconfident"), items)
    drift = next(p for p in over["probes"] if p["name"] == "difficulty_drift")
    ok &= check("overconfident_drift",
                over["verdict"] == "WARN" and drift["flagged"]
                and over["calibration"]["ece"] < 0.08,
                verdict=over["verdict"], ece=over["calibration"]["ece"],
                drift_stat=drift["stat"])

    noisy = scorecard(get_judge("noisy"), items)
    ok &= check("noisy_failure",
                noisy["verdict"] == "FAIL" and noisy["agreement"]["mae"] > 0.08,
                verdict=noisy["verdict"], mae=noisy["agreement"]["mae"])

    lenient = scorecard(get_judge("lenient"), items)
    harsh = scorecard(get_judge("harsh"), items)
    ok &= check("leniency_direction",
                lenient["agreement"]["mean_residual"] > 0.03
                and harsh["agreement"]["mean_residual"] < -0.03,
                lenient_resid=lenient["agreement"]["mean_residual"],
                harsh_resid=harsh["agreement"]["mean_residual"])

    v_flags = {n: bias_mod.verbosity_probe(get_judge(n))["flagged"]
               for n in ("calibrated", "lenient", "verbosity_biased")}
    ok &= check("verbosity_probe_selective",
                v_flags == {"calibrated": False, "lenient": False,
                            "verbosity_biased": True},
                flags=v_flags)

    p_cal = bias_mod.position_probe(get_judge("calibrated"))
    p_pos = bias_mod.position_probe(get_judge("position_biased"))
    ok &= check("position_probe_selective",
                not p_cal["flagged"] and p_cal["stat"] == 0.0
                and p_pos["flagged"] and p_pos["stat"] == 1.0,
                calibrated_flip=p_cal["stat"], biased_flip=p_pos["stat"])

    mm = scorecard(get_judge("rubric_mismatch"), items)
    skew = next(p for p in mm["probes"] if p["name"] == "category_skew")
    ok &= check("rubric_mismatch_safety",
                mm["verdict"] == "FAIL" and skew["flagged"]
                and "refusal_boundary" in skew["detail"],
                verdict=mm["verdict"], skew_stat=skew["stat"])

    a = json.dumps(scorecard(get_judge("lenient"), items), sort_keys=True)
    b = json.dumps(scorecard(get_judge("lenient"), items), sort_keys=True)
    ok &= check("scorecard_determinism", a == b,
                md5=hashlib.md5(a.encode()).hexdigest())

    ok &= check("brier_sanity",
                cal["calibration"]["brier"] < over["calibration"]["brier"],
                calibrated_brier=cal["calibration"]["brier"],
                overconfident_brier=over["calibration"]["brier"])

    report = {"n_evals": len(RESULTS),
              "n_passed": sum(1 for r in RESULTS if r["passed"]),
              "evals": RESULTS}
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_report.json")
    with open(out, "w") as f:
        json.dump(report, f, sort_keys=True, indent=2)
        f.write("\n")
    with open(out, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
    print("wrote %s  md5=%s" % (out, md5))
    print("%d/%d evals passed" % (report["n_passed"], report["n_evals"]))
    return 0 if ok and report["n_passed"] == report["n_evals"] else 1


if __name__ == "__main__":
    sys.exit(main())
