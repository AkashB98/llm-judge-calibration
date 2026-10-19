#!/usr/bin/env python3
"""End-to-end asserted demo: the harness telling good judges from bad ones."""

import json
import threading
import time
import urllib.request
from http.server import HTTPServer

from judge_lab.api import Handler
from judge_lab.corpus import load_corpus
from judge_lab.judges import get_judge
from judge_lab.report import grade_all, render_text, scorecard


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read().decode())


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def main():
    items = load_corpus()
    print("== corpus: %d items ==" % len(items))
    assert len(items) == 48
    assert len({i["category"] for i in items}) == 4
    goldens = [i["golden"] for i in items]
    assert min(goldens) >= 0.05 and max(goldens) <= 0.97
    print("golden grade range: %.2f .. %.2f" % (min(goldens), max(goldens)))

    print("\n== calibrated judge: expect PASS ==")
    card = scorecard(get_judge("calibrated"), items)
    print(render_text(card))
    assert card["verdict"] == "PASS", card["verdict"]
    assert card["calibration"]["ece"] < 0.08, card["calibration"]
    assert card["agreement"]["spearman"] > 0.90, card["agreement"]
    assert not any(p["flagged"] for p in card["probes"]), card["probes"]

    print("\n== overconfident judge: headline ECE looks fine, drift probe catches it ==")
    card = scorecard(get_judge("overconfident"), items)
    assert card["verdict"] in ("WARN", "FAIL"), card["verdict"]
    drift = next(p for p in card["probes"] if p["name"] == "difficulty_drift")
    assert drift["flagged"], drift
    print("verdict=%s; drift probe: %s" % (card["verdict"], drift["detail"]))

    print("\n== noisy judge: expect FAIL (confident but wrong) ==")
    card = scorecard(get_judge("noisy"), items)
    assert card["verdict"] == "FAIL", card["verdict"]
    print("verdict=%s; mae=%.3f ece=%.3f" % (
        card["verdict"], card["agreement"]["mae"], card["calibration"]["ece"]))

    print("\n== verbosity probe: flags only verbosity_biased ==")
    from judge_lab import bias as bias_mod
    for name in ("calibrated", "lenient", "verbosity_biased"):
        probe = bias_mod.verbosity_probe(get_judge(name))
        print("  %-16s stat=%+.3f flagged=%s" % (name, probe["stat"], probe["flagged"]))
        assert probe["flagged"] == (name == "verbosity_biased"), (name, probe)

    print("\n== position probe: flags only position_biased ==")
    for name in ("calibrated", "position_biased"):
        probe = bias_mod.position_probe(get_judge(name))
        print("  %-16s flip_rate=%.2f flagged=%s" % (name, probe["stat"], probe["flagged"]))
        assert probe["flagged"] == (name == "position_biased"), (name, probe)

    print("\n== live API round-trip ==")
    server = HTTPServer(("127.0.0.1", 8079), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)
    try:
        assert _get("http://127.0.0.1:8079/health") == {"ok": True}
        assert len(_get("http://127.0.0.1:8079/judges")["judges"]) == 9
        res = _post("http://127.0.0.1:8079/grade",
                    {"judge": "harsh", "item_ids": ["IF-01", "RB-02"]})
        assert res["judge"] == "harsh" and len(res["verdicts"]) == 2
        card = _get("http://127.0.0.1:8079/report?judge=lenient")
        assert card["verdict"] in ("WARN", "FAIL"), card["verdict"]
        assert card["agreement"]["mean_residual"] > 0.05, card["agreement"]
        print("health, judges, grade, report: all 200 OK")
    finally:
        server.shutdown()

    print("\nDEMO OK: the harness separates trustworthy graders from broken ones.")


if __name__ == "__main__":
    main()
