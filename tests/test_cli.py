"""CLI: subcommands, exit codes, output shape."""

import contextlib
import io
import json
import os
import subprocess
import sys

import cli

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "cli", *args],
                          capture_output=True, text=True, cwd=REPO)


def capture(argv):
    buf = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, buf.getvalue(), err.getvalue()


def test_judges_lists_nine():
    rc, out, _ = capture(["judges"])
    assert rc == 0
    assert "calibrated" in out and "verbosity_biased" in out


def test_check_calibrated_exit_0():
    rc, out, _ = capture(["check", "--judge", "calibrated"])
    assert rc == 0
    assert "PASS" in out


def test_check_overconfident_exit_1():
    rc, out, _ = capture(["check", "--judge", "overconfident"])
    assert rc == 1
    assert "WARN" in out


def test_check_noisy_exit_2():
    rc, out, _ = capture(["check", "--judge", "noisy"])
    assert rc == 2
    assert "FAIL" in out


def test_grade_json_shape():
    rc, out, _ = capture(["grade", "--judge", "harsh"])
    assert rc == 0
    data = json.loads(out)
    assert len(data) == 48
    assert {"item_id", "golden", "score", "confidence", "rationale", "judge"} <= set(data[0])


def test_calibrate_prints_scorecard():
    rc, out, _ = capture(["calibrate", "--judge", "lenient"])
    assert rc == 0
    assert "reliability curve" in out and "probes:" in out


def test_unknown_judge_exit_3():
    rc, _, err = capture(["check", "--judge", "bogus"])
    assert rc == 3
    assert "error" in err


def test_subprocess_check_noisy():
    # exercises the real entry point incl. sys.exit codes
    proc = run_cli("check", "--judge", "noisy")
    assert proc.returncode == 2
    assert "FAIL" in proc.stdout
