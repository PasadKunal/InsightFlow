"""Sample Ratio Mismatch (SRM) detection - the experiment's smoke alarm.

You designed a 50/50 split, but the data shows 5,100 vs 4,900. Is that just
chance, or did something *break*? SRM is the #1 data-quality check at every serious
experimentation platform, because a broken split silently invalidates every result
that follows it. Common causes: a buggy assignment SDK, bot traffic hitting one
arm, logging that drops events for one variant, or a redirect that fails for
treatment only.

The test is a simple **chi-squared goodness-of-fit** of the observed counts against
the expected split. The twist is the *threshold*: we flag at ``p < 0.001``, not the
usual 0.05. SRM checks run constantly, so a lax threshold would cry wolf; a real
SRM produces astronomically small p-values anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy import stats


# Deliberately strict. SRM is a "this experiment is broken" alarm, and true
# mismatches blow past this by many orders of magnitude - so false alarms stay rare.
DEFAULT_SRM_ALPHA = 0.001


@dataclass(frozen=True)
class SRMResult:
    """Outcome of an SRM check."""

    mismatch_detected: bool
    p_value: float
    chi_squared: float
    observed: dict[str, int]
    expected: dict[str, float]
    alpha: float

    def summary(self) -> str:
        if self.mismatch_detected:
            return (
                f"SRM DETECTED (p={self.p_value:.2e} < {self.alpha}). "
                f"Observed {self.observed} vs expected "
                f"{ {k: round(v, 1) for k, v in self.expected.items()} }. "
                f"Do NOT trust downstream results until the assignment pipeline is fixed."
            )
        return (
            f"No SRM (p={self.p_value:.3g} >= {self.alpha}). "
            f"Split looks healthy: observed {self.observed}."
        )

    def __str__(self) -> str:
        return self.summary()


def detect_srm(
    observed: Mapping[str, int],
    expected_ratio: Sequence[float] | Mapping[str, float] | None = None,
    *,
    alpha: float = DEFAULT_SRM_ALPHA,
) -> SRMResult:
    """Check whether observed arm counts match the intended split.

    Parameters
    ----------
    observed:
        Mapping of arm name -> observed user count, e.g. ``{"control": 4900,
        "treatment": 5100}``.
    expected_ratio:
        The intended split. May be:
          * ``None`` (default) -> assume an equal split across all arms;
          * a sequence aligned with ``observed``'s insertion order, e.g. ``[0.5,
            0.5]`` or ``[1, 1]`` (ratios are normalized, so weights are fine);
          * a mapping of arm name -> weight.
        Whatever you pass, the weights are normalized to sum to 1.
    alpha:
        Significance threshold. Defaults to a strict 0.001 (see module docstring).

    Notes
    -----
    Raises ``ValueError`` if fewer than two arms are provided or all counts are
    zero - there is nothing meaningful to test in those cases.
    """
    arms = list(observed.keys())
    if len(arms) < 2:
        raise ValueError("SRM needs at least two arms to compare.")

    observed_counts = np.array([observed[a] for a in arms], dtype=float)
    total = observed_counts.sum()
    if total == 0:
        raise ValueError("Total observed count is zero - no data to test.")

    # Resolve the expected split into normalized probabilities aligned to `arms`.
    if expected_ratio is None:
        weights = np.ones(len(arms))
    elif isinstance(expected_ratio, Mapping):
        missing = set(arms) - set(expected_ratio)
        if missing:
            raise ValueError(f"expected_ratio is missing arms: {sorted(missing)}")
        weights = np.array([expected_ratio[a] for a in arms], dtype=float)
    else:
        if len(expected_ratio) != len(arms):
            raise ValueError("expected_ratio length must match the number of arms.")
        weights = np.array(expected_ratio, dtype=float)

    if np.any(weights < 0) or weights.sum() == 0:
        raise ValueError("expected_ratio weights must be non-negative and not all zero.")
    probs = weights / weights.sum()
    expected_counts = probs * total

    chi2, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)

    return SRMResult(
        mismatch_detected=bool(p_value < alpha),
        p_value=float(p_value),
        chi_squared=float(chi2),
        observed={a: int(observed[a]) for a in arms},
        expected={a: float(e) for a, e in zip(arms, expected_counts)},
        alpha=alpha,
    )
