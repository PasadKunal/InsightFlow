"""Tests for the SPRT sequential test.

The two properties that matter: (1) it makes the *right* call when there's a real
effect vs. a true null, and (2) it stops *early* - the entire reason to use it.
"""

import numpy as np
import pytest

from insightflow.core import SequentialTest, run_sprt


def _stream(rng, rate, n):
    return list(rng.random(n) < rate)


def test_rejects_null_when_effect_is_real():
    rng = np.random.default_rng(0)
    # True rate 0.15, testing p0=0.10 vs p1=0.15 -> should reject the null.
    data = _stream(rng, 0.15, 5000)
    res = run_sprt(data, p0=0.10, p1=0.15, alpha=0.05, beta=0.20)
    assert res.decision == "reject_null"
    assert res.stopped


def test_accepts_null_when_no_effect():
    rng = np.random.default_rng(1)
    data = _stream(rng, 0.10, 5000)  # truly at the null rate
    res = run_sprt(data, p0=0.10, p1=0.15, alpha=0.05, beta=0.20)
    assert res.decision == "accept_null"
    assert res.stopped


def test_stops_early_relative_to_fixed_sample():
    # The headline benefit: a strong effect gets caught in far fewer observations.
    rng = np.random.default_rng(2)
    data = _stream(rng, 0.20, 5000)  # much higher than p1
    res = run_sprt(data, p0=0.10, p1=0.15, alpha=0.05, beta=0.20)
    assert res.decision == "reject_null"
    assert res.stopped_at < 1000  # nowhere near the full 5000


def test_boundaries_have_expected_order():
    test = SequentialTest(0.10, 0.15, alpha=0.05, beta=0.20)
    assert test.upper_boundary > 0 > test.lower_boundary


def test_latches_after_decision():
    # Once stopped, further observations must not change the verdict.
    rng = np.random.default_rng(4)
    test = SequentialTest(0.10, 0.15)
    final = None
    for converted in _stream(rng, 0.20, 5000):
        res = test.observe(converted)
        if res.stopped and final is None:
            final = res
    # Feeding more data after stopping keeps the same decision & stop point.
    after = test.observe(True)
    assert after.decision == final.decision
    assert after.stopped_at == final.stopped_at


def test_type_i_error_is_controlled():
    """Under a true null, false-positive rate should stay near alpha, not balloon."""
    rng = np.random.default_rng(7)
    alpha = 0.05
    false_positives = 0
    trials = 300
    for _ in range(trials):
        data = _stream(rng, 0.10, 3000)  # null is true
        res = run_sprt(data, p0=0.10, p1=0.15, alpha=alpha, beta=0.20)
        if res.decision == "reject_null":
            false_positives += 1
    # SPRT controls error at the boundaries; allow Monte-Carlo slack.
    assert false_positives / trials <= alpha + 0.03


def test_invalid_config_raises():
    with pytest.raises(ValueError):
        SequentialTest(0.5, 0.5)          # p0 == p1
    with pytest.raises(ValueError):
        SequentialTest(0.0, 0.5)          # p0 out of range
    with pytest.raises(ValueError):
        SequentialTest(0.1, 0.2, alpha=0)  # bad alpha
