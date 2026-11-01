"""Pure statistics for answer evals: Cohen's kappa, accuracy, noise spread, paired CIs, and the
score -> label mapping for faithfulness judges.

Definitions (checked by tests on hand-worked cases):

accuracy        share of items where the two label sequences agree.
Cohen's kappa   (po - pe) / (1 - pe) with po = accuracy and pe = sum over classes of
                P(rater A says c) * P(rater B says c), the agreement expected by chance from the
                two raters' marginals. None when pe == 1 (both raters use one and the same single
                class everywhere: kappa is 0/0, reported as undefined, never as 1.0).
noise           per metric, over the repeats of one commit: mean, min, max and the sample standard
                deviation (n - 1 denominator; 0.0 with a single repeat).
paired CI       percentile bootstrap (default 2000 resamples, fixed seed) of the mean of per-item
                differences between two experiments on the same items.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# Human labels (the labeller keys s / p / n) and the faithfulness-score thresholds that map a
# judge score in [0, 1] (the share of the answer's claims the passages support) to a label.
SUPPORTED = "supported"
PARTLY = "partly"
NOT_SUPPORTED = "not_supported"
LABELS = (SUPPORTED, PARTLY, NOT_SUPPORTED)
KEY_TO_LABEL = {"s": SUPPORTED, "p": PARTLY, "n": NOT_SUPPORTED}
SUPPORTED_MIN = 0.9  # score >= 0.9            -> supported
PARTLY_MIN = 0.5  # 0.5 <= score < 0.9       -> partly; below 0.5 -> not_supported


def score_to_label(score: float) -> str:
    """Map a faithfulness score in [0, 1] to supported / partly / not_supported."""
    if score >= SUPPORTED_MIN:
        return SUPPORTED
    if score >= PARTLY_MIN:
        return PARTLY
    return NOT_SUPPORTED


def to_binary(label: str) -> str:
    """Binary view: supported vs everything else."""
    return SUPPORTED if label == SUPPORTED else NOT_SUPPORTED


def accuracy(a: Sequence[str], b: Sequence[str]) -> float | None:
    if len(a) != len(b):
        raise ValueError("label sequences differ in length")
    if not a:
        return None
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def cohens_kappa(a: Sequence[str], b: Sequence[str]) -> float | None:
    """Cohen's kappa between two label sequences; None when undefined (see module doc)."""
    po = accuracy(a, b)
    if po is None:
        return None
    n = len(a)
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in ca.keys() | cb.keys())
    if math.isclose(pe, 1.0):
        return None
    return (po - pe) / (1 - pe)


@dataclass(frozen=True)
class NoiseStat:
    n: int
    mean: float
    min: float
    max: float
    stdev: float

    @property
    def spread(self) -> float:
        return self.max - self.min


def noise_stats(repeats: Sequence[Mapping[str, float]]) -> dict[str, NoiseStat]:
    """Per-metric mean, min, max and sample stdev across repeats of the same commit.
    A metric missing from some repeats is summarised over the repeats that have it."""
    out: dict[str, NoiseStat] = {}
    for metric in sorted({m for r in repeats for m in r}):
        vals = [float(r[metric]) for r in repeats if metric in r]
        out[metric] = NoiseStat(
            n=len(vals),
            mean=float(np.mean(vals)),
            min=min(vals),
            max=max(vals),
            stdev=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        )
    return out


def paired_bootstrap_ci(
    diffs: Sequence[float], *, resamples: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float] | None:
    """(mean, low, high) of the per-item differences; None for an empty list."""
    if not diffs:
        return None
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(resamples, len(arr)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(arr.mean()), float(lo), float(hi)
