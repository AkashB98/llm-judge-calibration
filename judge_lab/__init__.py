"""judge_lab: measure whether your LLM-as-judge grader is trustworthy.

Agreement + calibration + bias probes over a seeded golden corpus,
with per-judge scorecards and PASS/WARN/FAIL verdicts.
"""

from . import bias, corpus, judges, metrics, oracle, report  # noqa: F401

__version__ = "1.0.0"
