"""Uplift modeling demo - who responds best to the treatment?

Run it with:

    pip install -e ".[uplift]"
    ifvenv/bin/python examples/uplift_demo.py

It fits an X-Learner on synthetic data with a *known* heterogeneous effect, validates
the estimates against ground truth, ranks users into responder quintiles, and uses
SHAP to explain which features drive the treatment effect.
"""

from insightflow.uplift import (
    XLearner,
    explain_cate,
    make_synthetic_uplift_data,
    rank_by_quantile,
    validate_xlearner,
)


def rule(title: str) -> None:
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")


# 1. VALIDATE - prove the estimator is correct against a known truth ─────────
rule("1. Validate the X-Learner against a known ground-truth effect")
print(validate_xlearner(n_samples=5000, seed=0))

# 2. FIT - estimate a per-user treatment effect (CATE) ──────────────────────
rule("2. Fit the X-Learner and estimate each user's treatment effect")
data = make_synthetic_uplift_data(n_samples=5000, seed=1)
learner = XLearner(random_state=1)
cate = learner.fit_predict_cate(data.X, data.treatment, data.y)
print(f"Estimated ATE across all users: {cate.mean():.3f}")
print(f"But effects range from {cate.min():.2f} to {cate.max():.2f} - highly heterogeneous.")

# 3. SEGMENT - find the high-responder cohort to target ─────────────────────
rule("3. Rank users into responder quintiles (who should we roll out to?)")
report = rank_by_quantile(cate, n_quantiles=5)
print(report)
print(f"\n=> Targeting only the top quintile delivers "
      f"{report.top_quantile_lift:.2f}x the average treatment effect.")

# 4. EXPLAIN - which features drive the treatment effect? ────────────────────
rule("4. SHAP: which user features drive the treatment response?")
print(explain_cate(learner, data.X))

print("\nDone. Phase 4: X-Learner CATE + segment ranking + SHAP - validated on known effects.\n")
