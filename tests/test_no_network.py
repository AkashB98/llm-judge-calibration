"""No-network guard: the grading path must never touch the network.

Socket creation is stubbed to raise; grade/scorecard/probes/CLI must all
still work (the live-server tests in test_api.py cover localhost separately).
"""

import contextlib
import io
import socket

import cli
from judge_lab import bias as bias_mod
from judge_lab.corpus import load_corpus
from judge_lab.judges import get_judge
from judge_lab.report import grade_all, scorecard


def offline():
    """Context manager: any socket use raises."""
    real_socket = socket.socket
    real_create = socket.create_connection

    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during offline test")

    class Guard:
        def __enter__(self):
            socket.socket = _blocked
            socket.create_connection = _blocked

        def __exit__(self, *exc):
            socket.socket = real_socket
            socket.create_connection = real_create
            return False

    return Guard()


def test_grade_all_offline():
    with offline():
        verdicts = grade_all(get_judge("noisy"), load_corpus())
    assert len(verdicts) == 48


def test_scorecard_offline():
    with offline():
        card = scorecard(get_judge("overconfident"))
    assert card["verdict"] == "WARN"


def test_probes_offline():
    with offline():
        items = load_corpus()
        judge = get_judge("verbosity_biased")
        probes = bias_mod.run_probes(judge, items, grade_all(judge, items))
    assert any(p["flagged"] for p in probes)


def test_cli_check_offline():
    buf = io.StringIO()
    with offline(), contextlib.redirect_stdout(buf):
        rc = cli.main(["check", "--judge", "calibrated"])
    assert rc == 0
    assert "PASS" in buf.getvalue()


def test_cli_grade_offline():
    buf = io.StringIO()
    with offline(), contextlib.redirect_stdout(buf):
        rc = cli.main(["grade", "--judge", "lenient"])
    assert rc == 0
    assert '"item_id": "IF-01"' in buf.getvalue()
