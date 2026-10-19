"""JSON API over live localhost (127.0.0.1 only)."""

import json
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer

from judge_lab.api import Handler

BASE = "http://127.0.0.1:8077"


def _ensure_server():
    if getattr(_ensure_server, "started", False):
        return
    httpd = HTTPServer(("127.0.0.1", 8077), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    _ensure_server.started = True


_ensure_server()


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.status, json.loads(r.read().decode())


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, json.loads(r.read().decode())


def expect_http_error(fn, code):
    try:
        fn()
    except urllib.error.HTTPError as e:
        assert e.code == code, e.code
        return
    raise AssertionError("expected HTTP %d" % code)


def test_health():
    status, body = get("/health")
    assert status == 200 and body == {"ok": True}


def test_judges():
    status, body = get("/judges")
    assert status == 200 and len(body["judges"]) == 9


def test_grade_subset():
    status, body = post("/grade", {"judge": "harsh", "item_ids": ["IF-01", "RB-02"]})
    assert status == 200
    assert body["judge"] == "harsh"
    assert [v["item_id"] for v in body["verdicts"]] == ["IF-01", "RB-02"]


def test_grade_defaults_to_calibrated_full_corpus():
    status, body = post("/grade", {})
    assert status == 200 and len(body["verdicts"]) == 48


def test_report_noisy_fails():
    status, body = get("/report?judge=noisy")
    assert status == 200
    assert body["verdict"] == "FAIL"
    assert body["agreement"]["n"] == 48


def test_report_unknown_judge_400():
    expect_http_error(lambda: get("/report?judge=bogus"), 400)


def test_grade_unknown_items_400():
    expect_http_error(
        lambda: post("/grade", {"judge": "calibrated", "item_ids": ["NOPE-1"]}), 400)


def test_unknown_path_404():
    expect_http_error(lambda: get("/nope"), 404)
