"""The X-Learner — estimating *who* responds to treatment, not just the average.

A normal A/B test gives you one number: the Average Treatment Effect (ATE). "The new
checkout lifts conversion by 2%." But that 2% is an average over everyone — and it
almost always hides a richer story: maybe new users get +8% and power users get -1%.
The **Conditional** Average Treatment Effect (CATE) is that per-user story: τ(x) = the
expected lift for a user with features x. Knowing it lets you roll out to the people
it helps and spare the people it hurts.

The **X-Learner** (Künzel et al., 2019) is a meta-learner that estimates CATE well
even when treatment and control groups are unbalanced. It works in four moves:

1. **Outcome models.** Fit μ₀(x) on the control group and μ₁(x) on the treated group
   — two separate models of "what outcome do we expect here?".
2. **Impute individual effects.** For each treated user, ``Y − μ₀(x)`` estimates the
   effect they personally got. For each control user, ``μ₁(x) − Y`` does the same.
3. **CATE models.** Fit τ₁(x) on the treated imputations and τ₀(x) on the control
   ones — two direct models of the effect itself.
4. **Blend by propensity.** Combine them as ``τ(x) = e(x)·τ₀(x) + (1−e(x))·τ₁(x)``,
   where e(x) is the propensity score. This leans on whichever group is better
   represented at x, which is exactly why the X-Learner shines on imbalanced data.

Base models are gradient-boosted trees (XGBoost by default, scikit-learn as a
fallback), so the learner captures non-linear, interacting effects out of the box.
"""

from __future__ import annotations

import numpy as np

try:  # Prefer XGBoost; fall back to scikit-learn so the module always imports.
    from xgboost import XGBClassifier, XGBRegressor

    _HAS_XGB = True
except ImportError:  # pragma: no cover - exercised only when xgboost is absent
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )

    _HAS_XGB = False


def _default_regressor(random_state: int):
    if _HAS_XGB:
        return XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=random_state, n_jobs=1,
        )
    return HistGradientBoostingRegressor(random_state=random_state)


def _default_classifier(random_state: int):
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=random_state,
            eval_metric="logloss", n_jobs=1,
        )
    return HistGradientBoostingClassifier(random_state=random_state)


def _as_matrix(X) -> tuple[np.ndarray, list[str] | None]:
    """Return (float matrix, feature names or None). Accepts DataFrames or arrays."""
    if hasattr(X, "columns"):  # pandas DataFrame
        return X.to_numpy(dtype=float), list(X.columns)
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr, None


class XLearner:
    """X-Learner CATE estimator with gradient-boosted base models.

    Example
    -------
    >>> learner = XLearner(random_state=0).fit(X, treatment, y)
    >>> cate = learner.predict_cate(X_new)   # per-user treatment effect
    >>> learner.ate                          # average of the estimated effects
    """

    def __init__(self, *, base_regressor_factory=None, propensity_factory=None, random_state: int = 0):
        self.random_state = random_state
        self._make_regressor = base_regressor_factory or (lambda: _default_regressor(random_state))
        self._make_classifier = propensity_factory or (lambda: _default_classifier(random_state))
        self.feature_names_: list[str] | None = None
        self._fitted = False

    def fit(self, X, treatment, y) -> "XLearner":
        """Fit the four-stage X-Learner.

        Parameters
        ----------
        X: array-like or DataFrame of shape (n_samples, n_features)
        treatment: 1-D array of 0/1 (0 = control, 1 = treatment)
        y: 1-D array of outcomes (continuous or 0/1)
        """
        Xm, names = _as_matrix(X)
        w = np.asarray(treatment).astype(int)
        y = np.asarray(y, dtype=float)
        if not set(np.unique(w)).issubset({0, 1}):
            raise ValueError("treatment must contain only 0 (control) and 1 (treatment).")
        if Xm.shape[0] != w.shape[0] or Xm.shape[0] != y.shape[0]:
            raise ValueError("X, treatment, and y must all have the same number of rows.")

        treated, control = w == 1, w == 0
        if treated.sum() == 0 or control.sum() == 0:
            raise ValueError("Both a treatment and a control group are required.")

        self.feature_names_ = names

        # Stage 1 — outcome models, one per arm.
        self.mu0_ = self._make_regressor().fit(Xm[control], y[control])
        self.mu1_ = self._make_regressor().fit(Xm[treated], y[treated])

        # Stage 2 — impute the individual treatment effect for every user.
        imputed_treated = y[treated] - self.mu0_.predict(Xm[treated])
        imputed_control = self.mu1_.predict(Xm[control]) - y[control]

        # Stage 3 — model the effect directly within each arm.
        self.tau1_ = self._make_regressor().fit(Xm[treated], imputed_treated)
        self.tau0_ = self._make_regressor().fit(Xm[control], imputed_control)

        # Stage 4 — propensity model e(x) = P(treated | x) for the final blend.
        self.propensity_ = self._make_classifier().fit(Xm, w)

        self._fitted = True
        return self

    def predict_cate(self, X) -> np.ndarray:
        """Predict the per-user treatment effect τ(x)."""
        if not self._fitted:
            raise RuntimeError("Call fit() before predict_cate().")
        Xm, _ = _as_matrix(X)
        e = self.propensity_.predict_proba(Xm)[:, 1]
        e = np.clip(e, 0.05, 0.95)  # guard against divide-by-confidence at the edges
        return e * self.tau0_.predict(Xm) + (1 - e) * self.tau1_.predict(Xm)

    @property
    def ate(self) -> float:
        """The ATE implied by the model, averaged over the training data."""
        if not hasattr(self, "_train_cate"):
            raise RuntimeError("ATE is available after fit(); call predict_cate on training data.")
        return float(np.mean(self._train_cate))

    def fit_predict_cate(self, X, treatment, y) -> np.ndarray:
        """Convenience: fit, then return the CATE on the same X (and cache the ATE)."""
        self.fit(X, treatment, y)
        cate = self.predict_cate(X)
        self._train_cate = cate
        return cate
