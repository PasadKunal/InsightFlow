"""Tests for power analysis and sample-size planning.

Two kinds of checks:
  1. Internal consistency — sample_size and power are inverses of each other.
  2. Empirical validation — if we simulate experiments at the computed sample
     size, the fraction that come back significant really does hit ~80% power.
"""

import numpy as np
import pytest

from insightflow.core import (
    power_for_proportion,
    proportion_ztest,
    sample_size_for_mean,
    sample_size_for_proportion,
)


def test_sample_size_proportion_basic():
    res = sample_size_for_proportion(0.10, 0.20, alpha=0.05, power=0.80)
    assert res.per_arm > 0
    assert res.total == 2 * res.per_arm


def test_larger_effect_needs_fewer_users():
    small = sample_size_for_proportion(0.10, 0.05)  # detect a 5% relative lift
    large = sample_size_for_proportion(0.10, 0.30)  # detect a 30% relative lift
    assert large.per_arm < small.per_arm


def test_more_power_needs_more_users():
    p80 = sample_size_for_proportion(0.10, 0.10, power=0.80)
    p95 = sample_size_for_proportion(0.10, 0.10, power=0.95)
    assert p95.per_arm > p80.per_arm


def test_sample_size_and_power_are_inverses():
    # Compute n for 80% power, then confirm power_for_proportion agrees.
    ss = sample_size_for_proportion(0.10, 0.15, alpha=0.05, power=0.80)
    recovered = power_for_proportion(0.10, 0.15, ss.per_arm, alpha=0.05)
    assert recovered == pytest.approx(0.80, abs=0.02)


def test_sample_size_mean():
    res = sample_size_for_mean(std=15, minimum_detectable_effect=2, power=0.80)
    assert res.per_arm > 0
    # Halving the detectable effect roughly quadruples the sample (n ∝ 1/delta^2).
    finer = sample_size_for_mean(std=15, minimum_detectable_effect=1, power=0.80)
    assert finer.per_arm == pytest.approx(4 * res.per_arm, rel=0.05)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        sample_size_for_proportion(0.0, 0.1)  # baseline out of (0,1)
    with pytest.raises(ValueError):
        sample_size_for_proportion(0.10, -0.1)  # negative MDE
    with pytest.raises(ValueError):
        sample_size_for_mean(std=-1, minimum_detectable_effect=1)


@pytest.mark.parametrize("target_power", [0.80, 0.90])
def test_empirical_power_matches_design(target_power):
    """The gold-standard check: simulate at the designed n and measure real power."""
    rng = np.random.default_rng(7)
    baseline, mde = 0.10, 0.30  # true treatment rate = 0.13
    treat_rate = baseline * (1 + mde)

    ss = sample_size_for_proportion(baseline, mde, alpha=0.05, power=target_power)
    n = ss.per_arm

    significant = 0
    trials = 400
    for _ in range(trials):
        c = rng.binomial(n, baseline)
        t = rng.binomial(n, treat_rate)
        if proportion_ztest(c, n, t, n).significant:
            significant += 1

    empirical_power = significant / trials
    # Monte-Carlo noise over 400 trials — allow a reasonable band around target.
    assert empirical_power == pytest.approx(target_power, abs=0.06)
