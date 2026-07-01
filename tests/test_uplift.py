"""Tests for the uplift / CATE modeling layer.

Because true individual treatment effects are unobservable in real data, we validate
on *synthetic* data where we control the effect function exactly — then check that the
X-Learner recovers it, that segment ranking is monotone, and that SHAP surfaces the
features we actually baked the effect into.
"""

import numpy as np
import pytest

from insightflow.uplift import (
    XLearner,
    explain_cate,
    make_synthetic_uplift_data,
    rank_by_quantile,
    rank_segments,
    validate_xlearner,
)


@pytest.fixture(scope="module")
def data():
    return make_synthetic_uplift_data(n_samples=3000, n_features=5, seed=0)


@pytest.fixture(scope="module")
def fitted(data):
    learner = XLearner(random_state=0)
    cate = learner.fit_predict_cate(data.X, data.treatment, data.y)
    return learner, cate


# ── X-Learner ────────────────────────────────────────────────────────────────
def test_recovers_ate(data, fitted):
    _, cate = fitted
    assert np.mean(cate) == pytest.approx(np.mean(data.true_cate), abs=0.2)


def test_cate_correlates_with_truth(data, fitted):
    _, cate = fitted
    corr = np.corrcoef(cate, data.true_cate)[0, 1]
    assert corr > 0.85  # strong recovery of the heterogeneous structure


def test_predict_requires_fit():
    learner = XLearner()
    with pytest.raises(RuntimeError):
        learner.predict_cate(np.zeros((3, 5)))


def test_rejects_single_arm(data):
    with pytest.raises(ValueError):
        XLearner().fit(data.X, np.zeros_like(data.treatment), data.y)  # all control


def test_rejects_non_binary_treatment(data):
    bad = data.treatment.copy()
    bad[0] = 2
    with pytest.raises(ValueError):
        XLearner().fit(data.X, bad, data.y)


# ── validation harness ───────────────────────────────────────────────────────
def test_validation_report_is_accurate():
    report = validate_xlearner(n_samples=3000, seed=1)
    assert report.pehe < 0.6                       # low error vs true CATE
    assert report.cate_correlation > 0.85
    assert report.estimated_ate == pytest.approx(report.true_ate, abs=0.2)
    assert report.top_quintile_true_lift > 1.0     # targeting the top helps


# ── segment analysis ─────────────────────────────────────────────────────────
def test_quantile_ranking_is_monotone(fitted):
    _, cate = fitted
    report = rank_by_quantile(cate, n_quantiles=5)
    means = report.table["mean_cate"].to_numpy()
    assert np.all(np.diff(means) > 0)              # each quantile responds more
    assert report.top_quantile_lift > 1.0
    assert report.table["n_users"].sum() == len(cate)


def test_segment_ranking_picks_the_best(fitted):
    _, cate = fitted
    # Build a segment label correlated with the effect; "high" must rank first.
    segments = np.where(cate > np.median(cate), "high", "low")
    report = rank_segments(cate, segments)
    assert report.best_segment == "high"
    assert report.table.iloc[0]["mean_cate"] > report.table.iloc[-1]["mean_cate"]


def test_segment_length_mismatch_raises(fitted):
    _, cate = fitted
    with pytest.raises(ValueError):
        rank_segments(cate, ["only", "three", "labels"])


# ── SHAP explanation ─────────────────────────────────────────────────────────
def test_shap_identifies_true_drivers(data, fitted):
    learner, _ = fitted
    explanation = explain_cate(learner, data.X)
    # We built the effect from feature_0 (dominant) and feature_1; feature_0 should top.
    top = [name for name, _ in explanation.top_features(2)]
    assert top[0] == "feature_0"
    assert "feature_1" in top
    assert explanation.shap_values.shape == data.X.shape
    assert np.isfinite(explanation.base_value)
