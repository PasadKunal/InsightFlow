"""Experiment design: power analysis and sample-size planning.

The single most common way experiments go wrong is being *underpowered* - too
few users to detect the effect you care about, so you ship "no significant
difference" and never learn anything. This module answers the two questions you
must settle **before** an experiment starts:

1. "How many users per arm do I need?"  -> :func:`sample_size_for_proportion`,
   :func:`sample_size_for_mean`.
2. "Given the sample I can realistically get, what power do I actually have?"
   -> :func:`power_for_proportion`.

Everything here is built on the classic normal-approximation formulas and the
convention that **power = 1 - beta** (probability of catching a real effect).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class SampleSizeResult:
    """The answer to 'how many users do I need?', with the inputs kept for context."""

    per_arm: int
    total: int
    minimum_detectable_effect: float
    alpha: float
    power: float

    def __str__(self) -> str:
        return (
            f"Need {self.per_arm:,} per arm ({self.total:,} total) to detect an effect "
            f"of {self.minimum_detectable_effect:.4g} at alpha={self.alpha}, "
            f"power={self.power}."
        )


def sample_size_for_proportion(
    baseline_rate: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    relative: bool = True,
) -> SampleSizeResult:
    """Users per arm needed to detect a change in a conversion rate.

    Parameters
    ----------
    baseline_rate:
        The control conversion rate you expect, e.g. 0.10 for 10%.
    minimum_detectable_effect (MDE):
        The smallest lift worth detecting. If ``relative=True`` (default) this is
        a fraction of the baseline - 0.05 means "a 5% relative lift", i.e. 0.10 ->
        0.105. If ``relative=False`` it is an absolute change in rate.
    alpha:
        Significance level (two-sided).
    power:
        Desired probability of detecting the effect if it is real.
    """
    if not 0 < baseline_rate < 1:
        raise ValueError("baseline_rate must be strictly between 0 and 1.")
    if minimum_detectable_effect <= 0:
        raise ValueError("minimum_detectable_effect must be positive.")

    p1 = baseline_rate
    absolute_mde = baseline_rate * minimum_detectable_effect if relative else minimum_detectable_effect
    p2 = p1 + absolute_mde
    if not 0 < p2 < 1:
        raise ValueError(
            f"The implied treatment rate {p2:.4g} is outside (0, 1); reduce the MDE."
        )

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2

    # Standard two-proportion sample-size formula (pooled + unpooled variance terms).
    numerator = (
        z_alpha * np.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    n_per_arm = int(np.ceil(numerator / (absolute_mde**2)))

    return SampleSizeResult(
        per_arm=n_per_arm,
        total=2 * n_per_arm,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=alpha,
        power=power,
    )


def sample_size_for_mean(
    std: float,
    minimum_detectable_effect: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
) -> SampleSizeResult:
    """Users per arm needed to detect a shift in a continuous metric's mean.

    Parameters
    ----------
    std:
        The (assumed common) standard deviation of the metric.
    minimum_detectable_effect:
        The absolute change in mean you want to be able to detect, in the metric's
        own units (e.g. "$1.50 more revenue per user").
    """
    if std <= 0:
        raise ValueError("std must be positive.")
    if minimum_detectable_effect <= 0:
        raise ValueError("minimum_detectable_effect must be positive.")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    # Two-sample formula: n = 2 * (z_a + z_b)^2 * sigma^2 / delta^2.
    n_per_arm = int(
        np.ceil(2 * (z_alpha + z_beta) ** 2 * std**2 / minimum_detectable_effect**2)
    )

    return SampleSizeResult(
        per_arm=n_per_arm,
        total=2 * n_per_arm,
        minimum_detectable_effect=minimum_detectable_effect,
        alpha=alpha,
        power=power,
    )


def power_for_proportion(
    baseline_rate: float,
    minimum_detectable_effect: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
    relative: bool = True,
) -> float:
    """The statistical power you actually have for a *fixed* sample size.

    The inverse of :func:`sample_size_for_proportion`: given the users you can
    realistically enroll, how likely are you to detect the effect if it is real?
    Returns a value in [0, 1].
    """
    if n_per_arm <= 0:
        raise ValueError("n_per_arm must be positive.")

    p1 = baseline_rate
    absolute_mde = baseline_rate * minimum_detectable_effect if relative else minimum_detectable_effect
    p2 = p1 + absolute_mde
    if not 0 < p2 < 1:
        raise ValueError(f"The implied treatment rate {p2:.4g} is outside (0, 1).")

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    se_null = np.sqrt(2 * ((p1 + p2) / 2) * (1 - (p1 + p2) / 2) / n_per_arm)
    se_alt = np.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)

    # Probability the observed difference clears the critical value under H1.
    z = (abs(absolute_mde) - z_alpha * se_null) / se_alt
    return float(stats.norm.cdf(z))
