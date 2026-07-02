"""InsightFlow quickstart - the whole Phase 1 engine in one runnable script.

Run it with:

    ifvenv/bin/python examples/quickstart.py

It walks through the life of a single experiment: design it, randomize users,
sanity-check the split, then analyze the result three different ways.
"""

import numpy as np

from insightflow.core import (
    assign_many,
    benjamini_hochberg,
    beta_binomial_test,
    detect_srm,
    mann_whitney_u,
    proportion_ztest,
    run_sprt,
    sample_size_for_proportion,
    stratified_assign,
    summarize,
    two_sample_ttest,
)


def rule(title: str) -> None:
    print(f"\n{'─' * 70}\n{title}\n{'─' * 70}")


# 1. DESIGN - how many users do we need? ─────────────────────────────────────
rule("1. Experiment design: how big does this experiment need to be?")
plan = sample_size_for_proportion(
    baseline_rate=0.10,           # current checkout conversion is 10%
    minimum_detectable_effect=0.10,  # we care about a 10% relative lift (10% -> 11%)
    alpha=0.05,
    power=0.80,
)
print(plan)

# 2. RANDOMIZE - assign users, stratified by device ──────────────────────────
rule("2. Randomization: assign users to control / treatment (stratified by device)")
rng = np.random.default_rng(1)
n_users = plan.per_arm * 2
user_ids = [f"user-{i}" for i in range(n_users)]
devices = rng.choice(["ios", "android", "web"], size=n_users, p=[0.3, 0.5, 0.2])

assignments = stratified_assign(
    user_ids, list(devices), experiment_id="checkout-redesign-2026"
)
summary = summarize(assignments)
print(f"Assigned {summary.total:,} users -> "
      f"{summary.control:,} control / {summary.treatment:,} treatment "
      f"({summary.treatment_fraction:.1%} treatment)")

# 3. GUARD RAIL - did the split come out clean? ──────────────────────────────
rule("3. Data-quality guard: Sample Ratio Mismatch (SRM) check")
srm = detect_srm({"control": summary.control, "treatment": summary.treatment})
print(srm)

# 4. ANALYZE - simulate results and test them ────────────────────────────────
rule("4. Analysis: a conversion-rate experiment with a real +12% relative lift")
control_conv = rng.binomial(summary.control, 0.10)
treatment_conv = rng.binomial(summary.treatment, 0.112)  # true lift baked in
result = proportion_ztest(control_conv, summary.control,
                          treatment_conv, summary.treatment)
print(result)
print(f"   Control rate:   {result.extra['rate_control']:.3%}")
print(f"   Treatment rate: {result.extra['rate_treatment']:.3%}")
print(f"   Recommendation: {'SHIP IT ✅' if result.significant and result.effect_size > 0 else 'do not ship ❌'}")

# 5. The same idea for continuous & skewed metrics ───────────────────────────
rule("5. The engine also handles continuous and skewed metrics")
control_rev = rng.normal(50, 20, summary.control)
treatment_rev = rng.normal(52, 20, summary.treatment)
print("Revenue per user (t-test):", two_sample_ttest(control_rev, treatment_rev).summary())

control_time = rng.lognormal(2, 1, summary.control)
treatment_time = rng.lognormal(2.1, 1, summary.treatment)
print("Session length (Mann-Whitney):", mann_whitney_u(control_time, treatment_time).summary())

# 6. BAYESIAN - the same experiment, but "probability treatment is best" ──────
rule("6. Bayesian view: what a PM actually wants to hear")
bayes = beta_binomial_test(control_conv, summary.control, treatment_conv, summary.treatment)
print(bayes)

# 7. SEQUENTIAL - stop early, validly, the moment evidence is decisive ────────
rule("7. Sequential testing (SPRT): stop early without cheating")
stream = list(rng.random(50_000) < 0.13)   # true rate 13% vs a 10% null
sprt = run_sprt(stream, p0=0.10, p1=0.12, alpha=0.05, beta=0.20)
print(sprt)
print(f"   Reached a valid decision after only {sprt.stopped_at:,} users "
      f"(a fixed test here would have needed ~{sample_size_for_proportion(0.10, 0.20).per_arm:,}).")

# 8. MULTIPLE TESTING - don't get fooled by testing many metrics ─────────────
rule("8. Multiple-testing correction: 6 metrics, which lifts are real?")
metrics = ["revenue", "retention", "clicks", "signups", "shares", "latency"]
raw_p = [0.002, 0.03, 0.04, 0.20, 0.55, 0.80]
correction = benjamini_hochberg(raw_p, labels=metrics)
print(correction)
print("   (Raw p<0.05 would have flagged 3 metrics; after FDR control, fewer survive.)")

print("\nDone. Phases 1-2: frequentist + Bayesian + sequential + multiple-testing - all tested.\n")
