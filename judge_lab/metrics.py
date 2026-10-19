"""Agreement + calibration metrics. Stdlib only, hand-rolled.

Agreement (does the judge rank/grade like the oracle?):
  spearman  rank correlation between judge scores and golden grades
  mae       mean |score - golden|
  rmse      root mean squared error
  qwk       quadratic-weighted kappa on quintile bins (chance-corrected)
  mean_residual  mean(score - golden): systematic leniency (<0 harshness)

Calibration (does the judge know how sure it is?):
  A grade counts as "correct" when |score - golden| <= TOLERANCE.
  reliability_bins  per confidence bin: (center, accuracy, avg confidence, n)
  ece       expected calibration error = mean |accuracy - confidence|, weighted
  mce       max calibration error over non-empty bins
  brier     mean (confidence - correct)^2
"""

import math

TOLERANCE = 0.10


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def mae(scores, goldens):
    return mean(abs(s - g) for s, g in zip(scores, goldens))


def rmse(scores, goldens):
    return math.sqrt(mean((s - g) ** 2 for s, g in zip(scores, goldens)))


def mean_residual(scores, goldens):
    return mean(s - g for s, g in zip(scores, goldens))


def _ranks(xs):
    """Average ranks (1-based) with ties averaged. Deterministic."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return num / den if den else 0.0


def spearman(scores, goldens):
    """Spearman rank correlation. 1.0 = identical ranking."""
    return _pearson(_ranks(scores), _ranks(goldens))


def _bin_index(v, n_bins=5):
    return min(n_bins - 1, max(0, int(v * n_bins)))


def qwk(scores, goldens, n_bins=5):
    """Quadratic-weighted kappa on binned grades. 1.0 = perfect agreement."""
    n = len(scores)
    if n == 0:
        return 0.0
    conf = [[0] * n_bins for _ in range(n_bins)]
    for s, g in zip(scores, goldens):
        conf[_bin_index(s, n_bins)][_bin_index(g, n_bins)] += 1
    weights = [[((i - j) ** 2) / ((n_bins - 1) ** 2)
                for j in range(n_bins)] for i in range(n_bins)]
    hist_s = [sum(row) for row in conf]
    hist_g = [sum(conf[i][j] for i in range(n_bins)) for j in range(n_bins)]
    num = den = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            expected = hist_s[i] * hist_g[j] / n
            num += weights[i][j] * conf[i][j]
            den += weights[i][j] * expected
    if den == 0:
        return 1.0
    return 1.0 - num / den


def correctness(scores, goldens, tolerance=TOLERANCE):
    """1.0 where |score - golden| <= tolerance, else 0.0."""
    return [1.0 if abs(s - g) <= tolerance else 0.0 for s, g in zip(scores, goldens)]


def reliability_bins(confidences, corrects, n_bins=10):
    """Return non-empty bins as dicts: bin, center, n, accuracy, avg_confidence, gap."""
    bins = []
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences)
               if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = mean(corrects[i] for i in idx)
        avg_c = mean(confidences[i] for i in idx)
        bins.append({
            "bin": "[%.1f, %.1f)" % (lo, hi),
            "center": round((lo + hi) / 2, 3),
            "n": len(idx),
            "accuracy": round(acc, 4),
            "avg_confidence": round(avg_c, 4),
            "gap": round(abs(acc - avg_c), 4),
        })
    return bins


def ece(confidences, corrects, n_bins=10):
    """Expected calibration error."""
    n = len(confidences)
    if n == 0:
        return 0.0
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        idx = [i for i, c in enumerate(confidences)
               if (lo <= c < hi) or (b == n_bins - 1 and c == 1.0)]
        if not idx:
            continue
        acc = mean(corrects[i] for i in idx)
        avg_c = mean(confidences[i] for i in idx)
        total += len(idx) / n * abs(acc - avg_c)
    return total


def mce(confidences, corrects, n_bins=10):
    """Max calibration error over non-empty bins."""
    gaps = [b["gap"] for b in reliability_bins(confidences, corrects, n_bins)]
    return max(gaps) if gaps else 0.0


def brier(confidences, corrects):
    """Brier score: mean (confidence - correct)^2. Lower is better."""
    return mean((c - k) ** 2 for c, k in zip(confidences, corrects))
