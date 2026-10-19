# llm-judge-calibration

**Is your LLM-as-judge grader trustworthy?** A miscalibrated judge is a broken
thermometer: every eval that trusts it inherits its bias. This harness grades
the graders — measuring *agreement* and *calibration* of LLM-as-judge scorers
against a heuristic oracle over seeded golden grades, with bias/drift probes
for the failure modes headline metrics miss.

Stdlib only. Zero network. Fully deterministic (seeded judges, fixed corpus).

## Why this exists

Every team shipping agents runs LLM-as-judge evals in CI: a model scores model
outputs, and the scores gate releases. But judges have pathologies — they are
overconfident, they reward verbosity, they pick whichever candidate is listed
first, they learn subtly wrong rubrics. If you never calibrate the judge, your
"95% pass rate" is fiction. This is the evals-quality layer that backs every
other eval harness: prove the thermometer works before you trust the readings.

## Quickstart

```bash
python3 -m cli judges                  # list the 9 judge profiles
python3 -m cli calibrate --judge calibrated
python3 -m cli check --judge noisy      # CI gate: exit 0 PASS / 1 WARN / 2 FAIL
python3 -m cli demo                    # end-to-end asserted demo
python3 -m cli serve                   # JSON API on 127.0.0.1:8078
python3 tests/run_tests.py             # 93 hermetic tests, stdlib only
python3 evals/run_evals.py             # 12 golden evals -> evals/eval_report.json
```

## How it works

**Corpus** (`judge_lab/corpus.py`) — 48 (prompt, response) items across 4
categories (instruction following, factual QA, summarization, refusal
boundary), each with a golden grade from the oracle. Responses carry synthetic
flaw annotations like `[major:dropped_fact]` — these stand in for the content
features a real judge would extract from text (wrong numbers, missing facts,
format violations, unsafe compliance). All content is fictional (the Helios
Home smart-home brand); no personal data anywhere.

**Heuristic oracle** (`judge_lab/oracle.py`) — deterministic ground-truth
grader: applies the documented penalty table to the visible annotations.
Same response, same grade, always.

**Judge profiles** (`judge_lab/judges.py`) — 9 deterministic grader
personalities, each a realistic pathology. An optional real LLM judge sits
behind `JUDGELAB_LLM=1` (OpenAI-compatible endpoint); it never runs in
tests/evals/demo.

**Metrics** (`judge_lab/metrics.py`) — agreement: Spearman rank correlation,
MAE/RMSE, quadratic-weighted kappa, mean residual. Calibration: a grade counts
as "correct" when within ±0.10 of golden; then ECE / MCE / Brier score over a
reliability curve.

**Bias probes** (`judge_lab/bias.py`) — verbosity (length-controlled pairs),
position (pairwise order swaps), category skew, difficulty drift (ECE on hard
vs easy items).

**Scorecard** (`judge_lab/report.py`) — per-judge agreement + calibration +
per-category breakdown + probe results → PASS / WARN / FAIL with reasons.

## Measured scorecards

| judge | verdict | ECE | MAE | Spearman | what the harness found |
|---|---|---|---|---|---|
| calibrated | PASS | 0.008 | 0.038 | 0.938 | reference good judge |
| overconfident | WARN | 0.042 | 0.038 | 0.938 | headline ECE looks fine — **drift probe** flags hard-item overconfidence |
| underconfident | FAIL | 0.156 | 0.038 | 0.967 | confidence far below accuracy |
| lenient | FAIL | 0.182 | 0.068 | 0.954 | +0.062 systematic inflation, claims 99% confidence |
| harsh | FAIL | 0.640 | 0.124 | 0.926 | −0.124 systematic deflation |
| noisy | FAIL | 0.411 | 0.107 | 0.855 | confident but wrong (jitter it doesn't know about) |
| verbosity_biased | PASS* | 0.027 | 0.032 | 0.941 | *headline passes — **verbosity probe** flags +0.065 length bonus |
| position_biased | WARN | 0.008 | 0.038 | 0.938 | **position probe**: winner flips on 10/10 order swaps |
| rubric_mismatch | FAIL | 0.161 | 0.075 | 0.874 | wrong rubric: **category-skew probe** flags refusal_boundary (+0.133) |

The two starred rows are the point: headline metrics alone clear a
verbosity-biased judge and nearly clear an overconfident one. The probes exist
because calibration is not one number.

## Evals

`python3 evals/run_evals.py` — 12 golden evals pinning the behaviors above
(corpus size, oracle consistency, calibrated agreement/calibration,
overconfident drift, noisy failure, leniency direction, probe selectivity,
rubric-mismatch safety flag, scorecard determinism, Brier sanity).
`evals/eval_report.json` is committed; md5 `27165ebbfb0b6014fedac34237c3e4f0`
byte-identical across runs.

## Dev loop: what the tests caught

Real bugs found while building, not staged:

1. **The "calibrated" judge wasn't.** First version shared coarse
   severity-class rubric weights; systematic residuals on
   `unsafe_compliance`/`over_refusal` items pushed its ECE to ~0.096 — failing
   its own PASS band. Fix: calibrated uses the exact oracle penalty table; the
   wrong-rubric behavior became its own profile (`rubric_mismatch`), which the
   category-skew probe now catches on safety items.
2. **Verbosity probe never fired.** It used near-perfect clean items, and the
   0.995 score ceiling ate the word bonus. Fix: probe on mid-range golden items
   (0.25–0.90) with headroom below the ceiling.
3. **Float equality in tests.** `mae([0.5,0.7],[0.5,0.9]) == 0.1` is
   `0.10000000000000003`. Tests now assert within 1e-12.
4. **Demo over-asserted leniency.** Expected mean residual > 0.08; measured
   0.062 — the 0.99 score cap compresses the +0.12 shift on clean items.
   Assertion corrected with the explanation.
5. **Unhandled CLI errors.** `--judge bogus` died with a traceback; now exits
   3 with a clean stderr message.
6. **No pytest in this environment.** The suite follows the repo's stdlib-only
   `tests/run_tests.py` convention (no fixtures; manual stdout capture).

## Honest notes

- The flaw annotations are synthetic by design: this harness tests *grader
  calibration*, not NLP feature extraction. Plug a real judge in via
  `JUDGELAB_LLM=1` and the same scorecard measures it.
- Verdict bands (`--ece-warn` style) are constants in `report.py`
  (`FAIL_BANDS`/`WARN_BANDS`) and overridable per `scorecard()` call.
- The API binds `127.0.0.1` only. The LLM hook is the only network path and
  is never exercised by tests, evals, or the demo.
