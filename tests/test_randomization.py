"""Tests for deterministic and stratified randomization."""

import numpy as np
import pytest

from insightflow.core import assign, assign_many, stratified_assign, summarize


def test_assignment_is_deterministic():
    # Same user + same experiment -> same arm, every single time.
    first = assign("user-123", experiment_id="exp-checkout")
    for _ in range(100):
        assert assign("user-123", experiment_id="exp-checkout") == first


def test_same_user_differs_across_experiments():
    # Independent salting means no correlation between concurrent experiments.
    arms = {
        assign("user-123", experiment_id=f"exp-{i}") for i in range(50)
    }
    # Across many experiments the user should see both arms at least once.
    assert arms == {"control", "treatment"}


def test_split_is_approximately_balanced():
    ids = [f"user-{i}" for i in range(20_000)]
    assignments = assign_many(ids, experiment_id="exp-balance", treatment_fraction=0.5)
    summ = summarize(assignments)
    assert summ.total == 20_000
    assert summ.treatment_fraction == pytest.approx(0.5, abs=0.02)


def test_custom_treatment_fraction():
    ids = [f"u{i}" for i in range(20_000)]
    assignments = assign_many(ids, experiment_id="exp-90", treatment_fraction=0.9)
    summ = summarize(assignments)
    assert summ.treatment_fraction == pytest.approx(0.9, abs=0.02)


def test_assign_rejects_bad_fraction():
    with pytest.raises(ValueError):
        assign("u1", experiment_id="e", treatment_fraction=0.0)
    with pytest.raises(ValueError):
        assign("u1", experiment_id="e", treatment_fraction=1.0)


# ── stratified ───────────────────────────────────────────────────────────────
def test_stratified_balances_within_every_stratum():
    rng = np.random.default_rng(0)
    n = 12_000
    ids = [f"user-{i}" for i in range(n)]
    strata = rng.choice(["ios", "android", "web"], size=n, p=[0.2, 0.5, 0.3])

    assignments = stratified_assign(
        ids, list(strata), experiment_id="exp-strat", treatment_fraction=0.5
    )

    # Each stratum individually should be ~50/50, not just the overall population.
    for label in ["ios", "android", "web"]:
        members = [ids[i] for i in range(n) if strata[i] == label]
        treated = sum(assignments[m] == "treatment" for m in members)
        assert treated / len(members) == pytest.approx(0.5, abs=0.02)


def test_stratified_is_deterministic():
    ids = [f"u{i}" for i in range(1000)]
    strata = ["a" if i % 2 else "b" for i in range(1000)]
    a1 = stratified_assign(ids, strata, experiment_id="e")
    a2 = stratified_assign(ids, strata, experiment_id="e")
    assert a1 == a2


def test_stratified_length_mismatch_raises():
    with pytest.raises(ValueError):
        stratified_assign(["a", "b"], ["x"], experiment_id="e")
