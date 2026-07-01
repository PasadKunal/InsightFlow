"""Tests for the frequentist hypothesis tests.

Strategy: build data with a *known* answer (a real effect or a true null), then
assert the test lands on the right verdict, sign, and effect size — and that it
agrees with SciPy's own reference implementations where they overlap.
"""

import numpy as np
import pytest
from scipy import stats

from insightflow.core import (
    chi_squared_test,
    mann_whitney_u,
    proportion_ztest,
    two_sample_ttest,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ── t-test ───────────────────────────────────────────────────────────────────
def test_ttest_detects_real_effect(rng):
    control = rng.normal(100, 15, size=2000)
    treatment = rng.normal(105, 15, size=2000)  # +5 mean shift
    res = two_sample_ttest(control, treatment)

    assert res.significant
    assert res.p_value < 0.001
    assert res.effect_size > 0  # treatment is higher
    assert res.extra["mean_difference"] == pytest.approx(5, abs=1.5)
    # CI on the difference should exclude 0 when the effect is real.
    assert not res.confidence_interval.contains(0)


def test_ttest_true_null_usually_not_significant(rng):
    control = rng.normal(50, 10, size=1000)
    treatment = rng.normal(50, 10, size=1000)  # identical distributions
    res = two_sample_ttest(control, treatment)
    assert not res.significant
    assert res.confidence_interval.contains(0)


def test_ttest_matches_scipy(rng):
    c = rng.normal(0, 1, 500)
    t = rng.normal(0.3, 1, 500)
    res = two_sample_ttest(c, t, equal_var=False)
    ref_stat, ref_p = stats.ttest_ind(t, c, equal_var=False)
    assert res.statistic == pytest.approx(ref_stat)
    assert res.p_value == pytest.approx(ref_p)


def test_ttest_cohens_d_sign_and_magnitude(rng):
    # A one-SD shift should give Cohen's d near 1.0.
    control = rng.normal(0, 1, 5000)
    treatment = rng.normal(1, 1, 5000)
    res = two_sample_ttest(control, treatment)
    assert res.effect_size == pytest.approx(1.0, abs=0.1)


def test_ttest_empty_input_raises():
    with pytest.raises(ValueError):
        two_sample_ttest([], [1, 2, 3])


# ── proportion z-test ────────────────────────────────────────────────────────
def test_proportion_ztest_detects_lift():
    # 10% vs 13% on 5k users each — a clear, real lift.
    res = proportion_ztest(500, 5000, 650, 5000)
    assert res.significant
    assert res.extra["rate_control"] == pytest.approx(0.10)
    assert res.extra["rate_treatment"] == pytest.approx(0.13)
    assert res.effect_size == pytest.approx(0.30, abs=0.01)  # 30% relative lift


def test_proportion_ztest_no_difference():
    res = proportion_ztest(500, 5000, 505, 5000)
    assert not res.significant
    assert res.confidence_interval.contains(0)


def test_proportion_ztest_validates_bounds():
    with pytest.raises(ValueError):
        proportion_ztest(6000, 5000, 100, 5000)  # more conversions than users
    with pytest.raises(ValueError):
        proportion_ztest(10, 0, 5, 100)  # zero sample size


# ── chi-squared ──────────────────────────────────────────────────────────────
def test_chi_squared_agrees_with_ztest_on_significance():
    # For a 2x2 table the chi-squared and z-test give the same p-value.
    z = proportion_ztest(500, 5000, 650, 5000)
    c = chi_squared_test(500, 5000, 650, 5000)
    assert c.p_value == pytest.approx(z.p_value, abs=1e-6)
    assert c.significant == z.significant


def test_chi_squared_cramers_v_in_range():
    res = chi_squared_test(500, 5000, 650, 5000)
    assert 0 <= res.effect_size <= 1


# ── Mann-Whitney U ───────────────────────────────────────────────────────────
def test_mann_whitney_detects_shift_in_skewed_data(rng):
    # Log-normal data: means are unstable, so this is the right tool.
    control = rng.lognormal(0.0, 1.0, size=2000)
    treatment = rng.lognormal(0.5, 1.0, size=2000)
    res = mann_whitney_u(control, treatment)
    assert res.significant
    assert res.extra["median_treatment"] > res.extra["median_control"]


def test_mann_whitney_true_null(rng):
    control = rng.lognormal(0, 1, 1500)
    treatment = rng.lognormal(0, 1, 1500)
    res = mann_whitney_u(control, treatment)
    assert not res.significant


def test_mann_whitney_effect_size_bounds(rng):
    control = rng.lognormal(0, 1, 500)
    treatment = rng.lognormal(1, 1, 500)
    res = mann_whitney_u(control, treatment)
    assert -1 <= res.effect_size <= 1


# ── shared result behavior ───────────────────────────────────────────────────
def test_result_summary_is_readable(rng):
    res = two_sample_ttest(rng.normal(0, 1, 100), rng.normal(1, 1, 100))
    text = res.summary()
    assert "t-test" in text
    assert "p=" in text
    assert "Cohen's d" in text
