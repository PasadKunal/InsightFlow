"""Explaining the CATE with SHAP — *why* does this user respond?

A CATE model tells you a user has a high treatment effect. SHAP tells you **which of
their features drive it**. That's the difference between "target this cohort" and
"target this cohort *because they're new, on mobile, and price-sensitive*" — the
second is what actually informs product decisions.

The X-Learner blends several trees, which makes it awkward to explain directly. The
clean, widely-used trick is a **surrogate**: fit one gradient-boosted tree to
reproduce the X-Learner's CATE predictions (``x → τ̂(x)``), then run SHAP's exact
``TreeExplainer`` on that. Because the surrogate mimics the full learner faithfully,
its SHAP attributions explain what drives the estimated treatment effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .x_learner import XLearner, _as_matrix, _default_regressor


@dataclass(frozen=True)
class ShapExplanation:
    """Global and per-user SHAP attributions for the estimated treatment effect."""

    feature_names: list[str]
    shap_values: np.ndarray            # shape (n_users, n_features)
    base_value: float                  # model's average CATE prediction
    global_importance: dict[str, float]  # feature -> mean |SHAP|, sorted desc

    def top_features(self, k: int = 5) -> list[tuple[str, float]]:
        return list(self.global_importance.items())[:k]

    def summary(self) -> str:
        lines = ["Feature importance for the treatment effect (mean |SHAP|):"]
        for name, imp in self.top_features():
            lines.append(f"  {name:<18} {imp:.4f}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class CateExplainer:
    """SHAP explainer for an X-Learner's CATE, via a faithful surrogate model."""

    def __init__(self, learner: XLearner, *, random_state: int = 0):
        if not getattr(learner, "_fitted", False):
            raise RuntimeError("Fit the XLearner before explaining it.")
        self.learner = learner
        self.random_state = random_state
        self._surrogate = None
        self._explainer = None
        self.feature_names_: list[str] | None = None

    def fit(self, X) -> "CateExplainer":
        """Train the surrogate on the learner's CATE predictions over ``X``."""
        import shap

        Xm, names = _as_matrix(X)
        target = self.learner.predict_cate(X)

        self.feature_names_ = (
            names or self.learner.feature_names_ or [f"feature_{i}" for i in range(Xm.shape[1])]
        )
        self._surrogate = _default_regressor(self.random_state).fit(Xm, target)
        self._explainer = shap.TreeExplainer(self._surrogate)
        return self

    def explain(self, X) -> ShapExplanation:
        """Compute SHAP values for ``X`` and roll them up into global importances."""
        if self._explainer is None:
            raise RuntimeError("Call fit() before explain().")
        Xm, _ = _as_matrix(X)
        shap_values = np.asarray(self._explainer.shap_values(Xm))
        base_value = float(np.mean(self.learner.predict_cate(X)))

        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        global_importance = {
            self.feature_names_[i]: float(mean_abs[i]) for i in order
        }

        return ShapExplanation(
            feature_names=list(self.feature_names_),
            shap_values=shap_values,
            base_value=base_value,
            global_importance=global_importance,
        )


def explain_cate(learner: XLearner, X, *, random_state: int = 0) -> ShapExplanation:
    """One-shot helper: build the surrogate explainer and return the explanation."""
    return CateExplainer(learner, random_state=random_state).fit(X).explain(X)
