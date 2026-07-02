"""Multiple-testing correction - because twenty metrics is twenty chances to be fooled.

Run one test at ``alpha = 0.05`` and there's a 5% chance of a false positive. Run
*twenty* independent tests and the chance that **at least one** lies to you jumps to
64%. Every real experiment tracks many metrics - revenue, retention, clicks, latency,
a dozen guardrails - so this problem is not academic. It's the difference between
"treatment moved metric #14!" being a discovery and being noise.

InsightFlow offers the two standard corrections, which trade off differently:

* **Bonferroni** controls the *family-wise error rate* (FWER): the chance of **any**
  false positive at all. Dead simple, very strict - divide alpha by the number of
  tests. Right for safety-critical guardrails where one false alarm is unacceptable.

* **Benjamini-Hochberg (BH)** controls the *false discovery rate* (FDR): the expected
  *proportion* of your "discoveries" that are false. Much more powerful when you're
  screening many metrics and can tolerate that, say, 5% of the flagged ones are
  flukes. This is the sensible default for exploratory metric sweeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

Method = Literal["bonferroni", "benjamini-hochberg"]


@dataclass(frozen=True)
class CorrectionResult:
    """The outcome of correcting a family of p-values."""

    method: Method
    alpha: float
    p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    labels: tuple[str, ...] | None = None

    @property
    def n_tests(self) -> int:
        return len(self.p_values)

    @property
    def n_significant(self) -> int:
        return sum(self.rejected)

    def significant(self) -> list[str | int]:
        """The labels (or indices) of the hypotheses that survived correction."""
        keys = self.labels if self.labels is not None else range(self.n_tests)
        return [k for k, r in zip(keys, self.rejected) if r]

    def summary(self) -> str:
        return (
            f"{self.method} @ alpha={self.alpha}: "
            f"{self.n_significant}/{self.n_tests} significant after correction "
            f"-> {self.significant()}"
        )

    def __str__(self) -> str:
        return self.summary()


def _prepare(p_values: Sequence[float], labels: Sequence[str] | None):
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        raise ValueError("No p-values provided.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("All p-values must lie in [0, 1].")
    if labels is not None and len(labels) != p.size:
        raise ValueError("labels must be the same length as p_values.")
    return p


def bonferroni(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> CorrectionResult:
    """Bonferroni FWER correction.

    Reject a hypothesis when its p-value is below ``alpha / m`` (equivalently, when
    ``p * m`` is below ``alpha``). Guarantees the chance of *any* false positive
    across the whole family stays at or under ``alpha``.
    """
    p = _prepare(p_values, labels)
    m = p.size
    adjusted = np.minimum(p * m, 1.0)
    rejected = adjusted < alpha
    return CorrectionResult(
        method="bonferroni",
        alpha=alpha,
        p_values=tuple(float(x) for x in p),
        adjusted_p_values=tuple(float(x) for x in adjusted),
        rejected=tuple(bool(x) for x in rejected),
        labels=tuple(labels) if labels is not None else None,
    )


def benjamini_hochberg(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> CorrectionResult:
    """Benjamini-Hochberg FDR correction (the step-up procedure).

    Sort the ``m`` p-values ascending. Find the largest rank ``k`` for which
    ``p_(k) <= (k / m) * alpha`` and reject every hypothesis up to that rank. The
    adjusted ("BH") p-values are the running minimum of ``p_(i) * m / i`` taken from
    the largest p-value downward, so they stay monotone and interpretable.
    """
    p = _prepare(p_values, labels)
    m = p.size

    order = np.argsort(p)                      # indices that sort p ascending
    ranks = np.arange(1, m + 1)                # 1..m
    p_sorted = p[order]

    # Step-up rejection threshold.
    thresholds = ranks / m * alpha
    below = p_sorted <= thresholds
    if below.any():
        k = np.max(np.nonzero(below))          # largest passing rank (0-indexed)
        reject_sorted = np.arange(m) <= k
    else:
        reject_sorted = np.zeros(m, dtype=bool)

    # BH-adjusted p-values: enforce monotonicity from the top down.
    adjusted_sorted = np.minimum.accumulate((p_sorted * m / ranks)[::-1])[::-1]
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    # Un-sort everything back to the caller's original order.
    adjusted = np.empty(m)
    rejected = np.empty(m, dtype=bool)
    adjusted[order] = adjusted_sorted
    rejected[order] = reject_sorted

    return CorrectionResult(
        method="benjamini-hochberg",
        alpha=alpha,
        p_values=tuple(float(x) for x in p),
        adjusted_p_values=tuple(float(x) for x in adjusted),
        rejected=tuple(bool(x) for x in rejected),
        labels=tuple(labels) if labels is not None else None,
    )


def correct(
    p_values: Sequence[float],
    *,
    method: Method = "benjamini-hochberg",
    alpha: float = 0.05,
    labels: Sequence[str] | None = None,
) -> CorrectionResult:
    """Dispatch to a correction method by name (BH is the default)."""
    if method == "bonferroni":
        return bonferroni(p_values, alpha=alpha, labels=labels)
    if method == "benjamini-hochberg":
        return benjamini_hochberg(p_values, alpha=alpha, labels=labels)
    raise ValueError(f"Unknown method {method!r}; use 'bonferroni' or 'benjamini-hochberg'.")
