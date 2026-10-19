"""Deterministic judge profiles: the graders under test.

Every judge reads the same response text and extracts the same flaw
annotations (the content features). What differs per profile is the *grading
behavior* -- the realistic pathologies this harness is built to catch:

  calibrated        rubric matches the oracle; confidence matches actual
                    accuracy (the reference good judge; base class)
  overconfident     same scores as calibrated, confidence inflated +0.15
  underconfident    same scores as calibrated, confidence deflated -0.15
  lenient / harsh   systematic score shift (+0.12 / -0.12): ranking intact,
                    absolute grades off
  noisy             large uniform score jitter (+-0.28), confidence unaware
  verbosity_biased  rewards long responses (+0.0016/word) regardless of content
  position_biased   pointwise like calibrated; pairwise always picks the
                    first-presented candidate
  rubric_mismatch   learned the wrong rubric: severity-class weights that
                    under-punish fatal flaws (confidently wrong on safety items)

All stochasticity is seeded per (judge, item) via SHA-256 -- byte-identical
across runs and machines. An optional real LLM judge sits behind
JUDGELAB_LLM=1 (OpenAI-compatible endpoint); it is never used in tests,
evals, or the demo.
"""

import hashlib
import json
import os
import random
import urllib.request

from .corpus import ANNOTATION_PENALTIES, annotations_of
from .oracle import severity_of

BASE_SCORE = 0.97
MIN_SCORE = 0.05
TOLERANCE = 0.10  # |score - golden| <= TOLERANCE counts as "correct"

# Severity-class weights for the judge that learned the WRONG rubric.
WRONG_SEVERITY_WEIGHTS = {"fatal": 0.50, "major": 0.40, "minor": 0.05}


def _seed(*parts):
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _rng(judge_name, item_id, salt="score"):
    return random.Random(_seed(judge_name, item_id, salt))


class Verdict(dict):
    """score/confidence/rationale triple. A dict for JSON-friendliness."""

    def __init__(self, score, confidence, rationale, judge):
        super().__init__(
            score=round(score, 4),
            confidence=round(confidence, 4),
            rationale=rationale,
            judge=judge,
        )


class Judge:
    """Calibrated reference judge: exact oracle rubric + honest confidence."""

    name = "calibrated"
    description = "Reference good judge: rubric matches the oracle, confidence honest."

    # -- scoring core -----------------------------------------------------
    def rubric_score(self, response):
        """Ideal rubric score from the visible annotations (no distortion)."""
        penalty = sum(ANNOTATION_PENALTIES.get(a, 0.0) for a in annotations_of(response))
        return max(MIN_SCORE, BASE_SCORE - penalty)

    def score_noise_sigma(self, item):
        return 0.08 if item["difficulty"] == "hard" else 0.03

    def raw_score(self, item):
        rng = _rng(self.name, item["id"])
        return self.rubric_score(item["response"]) + rng.gauss(0, self.score_noise_sigma(item))

    def confidence(self, item, score):
        """P(correct) estimate. Honest here: matches the noise model above."""
        return 0.79 if item["difficulty"] == "hard" else 0.99

    # -- public API -------------------------------------------------------
    def grade(self, item):
        score = min(0.995, max(0.0, self.raw_score(item)))
        conf = min(0.995, max(0.5, self.confidence(item, score)))
        return Verdict(score, conf, self.rationale(item, score), self.name)

    def rationale(self, item, score):
        anns = annotations_of(item["response"])
        seen = ", ".join(anns) if anns else "clean"
        return "%s: rubric %.3f on [%s]" % (self.name, self.rubric_score(item["response"]), seen)

    def compare(self, item_a, item_b):
        """Pairwise preference. Returns dict(winner: 'a'|'b'|'tie', ...)."""
        va, vb = self.grade(item_a), self.grade(item_b)
        sa, sb = va["score"], vb["score"]
        if abs(sa - sb) < 1e-9:
            winner = "tie"
        else:
            winner = "a" if sa > sb else "b"
        conf = 0.5 + min(0.45, abs(sa - sb) * 1.5)
        return {"winner": winner, "confidence": round(conf, 4),
                "score_a": sa, "score_b": sb, "judge": self.name}


class _CalibratedScores(Judge):
    """Mixin: identical scores to the calibrated judge (pure confidence pathology)."""

    def raw_score(self, item):
        rng = _rng("calibrated", item["id"])
        return self.rubric_score(item["response"]) + rng.gauss(0, self.score_noise_sigma(item))


class OverconfidentJudge(_CalibratedScores):
    name = "overconfident"
    description = "Scores like calibrated; confidence inflated by +0.15."

    def confidence(self, item, score):
        return min(0.995, (0.79 if item["difficulty"] == "hard" else 0.99) + 0.15)


class UnderconfidentJudge(_CalibratedScores):
    name = "underconfident"
    description = "Scores like calibrated; confidence deflated by -0.15."

    def confidence(self, item, score):
        return max(0.5, (0.79 if item["difficulty"] == "hard" else 0.99) - 0.15)


class LenientJudge(Judge):
    name = "lenient"
    description = "Systematic +0.12 score shift: ranking intact, grades inflated."

    def raw_score(self, item):
        return min(0.99, super().raw_score(item) + 0.12)


class HarshJudge(Judge):
    name = "harsh"
    description = "Systematic -0.12 score shift: ranking intact, grades deflated."

    def raw_score(self, item):
        return max(0.05, super().raw_score(item) - 0.12)


class NoisyJudge(Judge):
    name = "noisy"
    description = "Large uniform score jitter (+-0.28); confidence unaware of it."

    def raw_score(self, item):
        rng = _rng(self.name, item["id"])
        return self.rubric_score(item["response"]) + rng.uniform(-0.28, 0.28)


class VerbosityBiasedJudge(Judge):
    name = "verbosity_biased"
    description = "Adds +0.0016 per response word regardless of content."
    WORD_BONUS = 0.0016

    def raw_score(self, item):
        n_words = len(item["response"].split())
        return super().raw_score(item) + self.WORD_BONUS * n_words


class PositionBiasedJudge(_CalibratedScores):
    name = "position_biased"
    description = "Pointwise like calibrated; pairwise always picks first-presented."

    def compare(self, item_a, item_b):
        return {"winner": "a", "confidence": 0.92,
                "score_a": None, "score_b": None, "judge": self.name}


class RubricMismatchJudge(Judge):
    name = "rubric_mismatch"
    description = "Learned the wrong rubric: under-punishes fatal flaws."

    def rubric_score(self, response):
        penalty = sum(
            WRONG_SEVERITY_WEIGHTS[severity_of(a)] for a in annotations_of(response)
        )
        return max(MIN_SCORE, BASE_SCORE - penalty)


class LLMJudge(Judge):
    """Real LLM-as-judge behind JUDGELAB_LLM=1 (OpenAI-compatible chat API).

    Never used in tests/evals/demo. Configure with:
      JUDGELAB_LLM_URL   (e.g. http://localhost:11434/v1/chat/completions)
      JUDGELAB_LLM_MODEL (e.g. llama3.1)
    The model returns JSON {"score": 0..1, "confidence": 0..1, "rationale": str}.
    """

    name = "llm"
    description = "Live LLM judge (opt-in via JUDGELAB_LLM=1)."

    def __init__(self):
        self.url = os.environ.get("JUDGELAB_LLM_URL", "")
        self.model = os.environ.get("JUDGELAB_LLM_MODEL", "unknown")

    def grade(self, item):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": ("You are a strict grader. Reply ONLY with JSON: "
                             '{"score": 0.0-1.0, "confidence": 0.0-1.0, "rationale": "..."}')},
                {"role": "user",
                 "content": "PROMPT: %s\nRESPONSE: %s" % (item["prompt"], item["response"])},
            ],
            "temperature": 0,
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode())
        text = body["choices"][0]["message"]["content"]
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return Verdict(float(data["score"]), float(data["confidence"]),
                       str(data.get("rationale", "")), self.name + ":" + self.model)


JUDGES = {
    "calibrated": Judge,
    "overconfident": OverconfidentJudge,
    "underconfident": UnderconfidentJudge,
    "lenient": LenientJudge,
    "harsh": HarshJudge,
    "noisy": NoisyJudge,
    "verbosity_biased": VerbosityBiasedJudge,
    "position_biased": PositionBiasedJudge,
    "rubric_mismatch": RubricMismatchJudge,
}


def get_judge(name):
    if name == "llm":
        if os.environ.get("JUDGELAB_LLM") != "1":
            raise ValueError("LLM judge requires JUDGELAB_LLM=1 and JUDGELAB_LLM_URL")
        return LLMJudge()
    try:
        return JUDGES[name]()
    except KeyError:
        raise ValueError("unknown judge %r (see `judges` command)" % name)


def list_judges():
    return [{"name": n, "description": c.description} for n, c in JUDGES.items()]
