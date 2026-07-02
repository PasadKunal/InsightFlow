"""Validating the uplift model against a known ground truth.

CATE estimation has a famous problem: you can *never* observe the true individual
effect in real data, because you only ever see a user under treatment **or** control,
never both. So how do you know your X-Learner is any good?

The answer used everywhere in the causal-ML literature: **synthetic data with a known
effect function.** Here we generate users whose true per-user effect τ(x) we control
exactly, fit the X-Learner as if we were blind to it, and then grade the estimates
against the truth. The headline metric is **PEHE** (Precision in Estimation of
Heterogeneous Effects) - the root-mean-square error between estimated and true CATE.
Lower is better. We also report the correlation and how much true uplift you'd capture
by targeting the model's top quintile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .x_learner import XLearner


@dataclass(frozen=True)
class SyntheticData:
    """A synthetic uplift dataset with the true per-user effect exposed."""

    X: np.ndarray
    treatment: np.ndarray
    y: np.ndarray
    true_cate: np.ndarray
    feature_names: list[str]


def make_synthetic_uplift_data(
    n_samples: int = 4000,
    n_features: int = 5,
    *,
    treatment_fraction: float = 0.5,
    noise: float = 1.0,
    seed: int = 0,
) -> SyntheticData:
    """Generate data with a deliberately *heterogeneous* treatment effect.

    The true effect is driven mostly by the first feature::

        τ(x) = 1 + 2 · sigmoid(3 · x0) + 0.5 · x1

    so users with a high ``x0`` respond far more strongly than the average - exactly
    the kind of structure a good CATE model should recover and a segment analysis
    should surface.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_samples, n_features))
    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Randomized assignment -> propensity is constant (the clean experimental case).
    treatment = (rng.random(n_samples) < treatment_fraction).astype(int)

    # Baseline outcome (present with or without treatment).
    baseline = X[:, 0] + 0.5 * X[:, 1] - 0.5 * X[:, 2]

    # The true, heterogeneous per-user effect.
    true_cate = 1.0 + 2.0 / (1.0 + np.exp(-3.0 * X[:, 0])) + 0.5 * X[:, 1]

    y = baseline + treatment * true_cate + rng.normal(0, noise, size=n_samples)

    return SyntheticData(
        X=X, treatment=treatment, y=y, true_cate=true_cate, feature_names=feature_names
    )


@dataclass(frozen=True)
class UpliftValidationReport:
    """How well the estimated CATE matches the known truth."""

    pehe: float                 # sqrt(mean((tau_hat - tau_true)^2)); lower is better
    cate_correlation: float     # Pearson corr(tau_hat, tau_true); higher is better
    true_ate: float
    estimated_ate: float
    top_quintile_true_lift: float  # true effect in the model's top quintile / overall true ATE
    n_samples: int

    def summary(self) -> str:
        return (
            f"Uplift validation on {self.n_samples:,} synthetic users:\n"
            f"  PEHE (RMSE vs true CATE): {self.pehe:.3f}\n"
            f"  CATE correlation:         {self.cate_correlation:.3f}\n"
            f"  ATE  true / estimated:    {self.true_ate:.3f} / {self.estimated_ate:.3f}\n"
            f"  Targeting the model's top quintile captures "
            f"{self.top_quintile_true_lift:.2f}x the overall ATE"
        )

    def __str__(self) -> str:
        return self.summary()


def validate_xlearner(
    data: SyntheticData | None = None,
    *,
    seed: int = 0,
    **data_kwargs,
) -> UpliftValidationReport:
    """Fit an X-Learner on synthetic data and grade it against the true effects."""
    if data is None:
        data = make_synthetic_uplift_data(seed=seed, **data_kwargs)

    learner = XLearner(random_state=seed)
    estimated = learner.fit_predict_cate(data.X, data.treatment, data.y)
    true = data.true_cate

    pehe = float(np.sqrt(np.mean((estimated - true) ** 2)))
    correlation = float(np.corrcoef(estimated, true)[0, 1])

    # Rank users by the *estimated* effect, then measure the *true* effect we'd capture.
    top_cutoff = np.quantile(estimated, 0.80)
    top_mask = estimated >= top_cutoff
    overall_true_ate = float(np.mean(true))
    top_quintile_true_lift = float(np.mean(true[top_mask]) / overall_true_ate)

    return UpliftValidationReport(
        pehe=pehe,
        cate_correlation=correlation,
        true_ate=overall_true_ate,
        estimated_ate=float(np.mean(estimated)),
        top_quintile_true_lift=top_quintile_true_lift,
        n_samples=data.X.shape[0],
    )
