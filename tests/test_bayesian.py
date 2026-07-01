"""Tests for the Beta-Binomial Bayesian test."""

import numpy as np
import pytest

from insightflow.core import beta_binomial_test


def test_posterior_update_is_correct():
    # Beta(1,1) prior + 30 conversions in 100 -> Beta(31, 71).
    res = beta_binomial_test(30, 100, 30, 100)
    assert res.control.alpha == 31
    assert res.control.beta == 71
    assert res.control.mean == pytest.approx(31 / 102)


def test_clear_winner_has_high_probability():
    # 10% vs 15% on 5k users each is a decisive treatment win.
    res = beta_binomial_test(500, 5000, 750, 5000)
    assert res.prob_treatment_best > 0.99
    assert res.expected_absolute_uplift > 0
    assert res.recommendation == "SHIP treatment"


def test_clear_loser_has_low_probability():
    res = beta_binomial_test(750, 5000, 500, 5000)  # treatment is worse
    assert res.prob_treatment_best < 0.01
    assert res.recommendation == "KEEP control"


def test_tie_is_inconclusive():
    res = beta_binomial_test(500, 5000, 505, 5000)
    assert 0.2 < res.prob_treatment_best < 0.8
    assert res.recommendation == "KEEP RUNNING (not yet conclusive)"


def test_probability_is_symmetric():
    # P(treatment best) from one framing == 1 - P(...) from the swapped framing.
    a = beta_binomial_test(500, 5000, 600, 5000)
    b = beta_binomial_test(600, 5000, 500, 5000)
    assert a.prob_treatment_best == pytest.approx(1 - b.prob_treatment_best, abs=0.01)


def test_expected_loss_small_for_clear_winner():
    # When treatment clearly wins, the risk of shipping it is tiny.
    res = beta_binomial_test(500, 5000, 750, 5000)
    assert res.expected_loss < 1e-3


def test_credible_interval_contains_true_rate():
    # With lots of data the 95% credible interval should bracket the true rate.
    rng = np.random.default_rng(3)
    true_rate = 0.2
    n = 20_000
    conv = rng.binomial(n, true_rate)
    res = beta_binomial_test(conv, n, conv, n)
    ci = res.control.credible_interval(0.95)
    assert ci.contains(true_rate)


def test_reproducible_with_seed():
    a = beta_binomial_test(500, 5000, 600, 5000, seed=123)
    b = beta_binomial_test(500, 5000, 600, 5000, seed=123)
    assert a.prob_treatment_best == b.prob_treatment_best


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        beta_binomial_test(200, 100, 50, 100)      # more conversions than users
    with pytest.raises(ValueError):
        beta_binomial_test(50, 100, 50, 100, prior_alpha=0)  # bad prior
    with pytest.raises(ValueError):
        beta_binomial_test(50, 0, 50, 100)         # zero sample
