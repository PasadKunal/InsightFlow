"""Simulation-based validation — proving the statistical engine is *correct*.

Anyone can call `scipy.stats.ttest_ind`. The hard, senior-level question is: **how do
you know your experimentation engine actually controls error the way it claims?** A
p-value threshold of 0.05 is a promise — "at most 5% of true nulls will be called
significant" — and an 80%-power design is another — "at least 80% of real effects will
be caught." This module *measures* whether those promises hold, empirically.

The method is the gold standard for validating experimentation infrastructure:

* **Type I error.** Generate hundreds of experiments where the null is *true* (no
  effect at all) and count how often the test wrongly fires. This should land at or
  below alpha. If it's higher, the engine is producing false discoveries.
* **Power.** Generate hundreds of experiments where a *real* effect exists, sized at
  the sample size the power analysis prescribed, and count how often the test catches
  it. This should land near the target power (e.g. 80%).

Running this across the frequentist tests is what backs the headline claim: *"validated
across 500 simulated experiments at 80%+ power, alpha = 0.05."*
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from insightflow.core import (
    proportion_ztest,
    sample_size_for_mean,
    sample_size_for_proportion,
    two_sample_ttest,
)


@dataclass(frozen=True)
class ScenarioResult:
    """The empirical outcome of one validation scenario over many experiments."""

    test_name: str
    scenario: str          # "Type I error" or "Power"
    target: float          # alpha (for Type I) or desired power
    empirical: float       # measured rejection rate
    n_experiments: int
    n_per_arm: int
    tolerance: float

    @property
    def passed(self) -> bool:
        if self.scenario == "Type I error":
            # False-positive rate must not exceed alpha (beyond Monte-Carlo noise).
            return self.empirical <= self.target + self.tolerance
        # Power must not fall short of the target (beyond noise).
        return self.empirical >= self.target - self.tolerance

    def summary(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        return (
            f"[{mark}] {self.test_name:<24} {self.scenario:<14} "
            f"target={self.target:.2f}  empirical={self.empirical:.3f}  "
            f"(n={self.n_experiments}, {self.n_per_arm:,}/arm)"
        )


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate of every scenario, with an overall pass/fail."""

    results: list[ScenarioResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        header = "═" * 74
        lines = [header, "  InsightFlow — Statistical Engine Validation", header]
        lines += [f"  {r.summary()}" for r in self.results]
        lines.append(header)
        verdict = "ALL CHECKS PASSED ✓" if self.all_passed else "SOME CHECKS FAILED ✗"
        lines.append(f"  {verdict}")
        lines.append(header)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ── proportion (z-test) ──────────────────────────────────────────────────────
def validate_proportion(
    n_experiments: int = 500,
    *,
    baseline: float = 0.10,
    mde: float = 0.30,
    alpha: float = 0.05,
    power: float = 0.80,
    seed: int = 0,
) -> tuple[ScenarioResult, ScenarioResult]:
    """Validate the two-proportion z-test's Type I error and power."""
    rng = np.random.default_rng(seed)
    n = sample_size_for_proportion(baseline, mde, alpha=alpha, power=power).per_arm
    treat_rate = baseline * (1 + mde)

    # Type I error: both arms at the same (null) rate.
    false_positives = sum(
        proportion_ztest(
            rng.binomial(n, baseline), n, rng.binomial(n, baseline), n, alpha=alpha
        ).significant
        for _ in range(n_experiments)
    )
    type_i = ScenarioResult(
        "Two-proportion z-test", "Type I error", alpha,
        false_positives / n_experiments, n_experiments, n, tolerance=0.025,
    )

    # Power: treatment arm carries the real effect.
    detections = sum(
        proportion_ztest(
            rng.binomial(n, baseline), n, rng.binomial(n, treat_rate), n, alpha=alpha
        ).significant
        for _ in range(n_experiments)
    )
    power_result = ScenarioResult(
        "Two-proportion z-test", "Power", power,
        detections / n_experiments, n_experiments, n, tolerance=0.05,
    )
    return type_i, power_result


# ── continuous (t-test) ──────────────────────────────────────────────────────
def validate_continuous(
    n_experiments: int = 500,
    *,
    std: float = 1.0,
    effect: float = 0.2,
    alpha: float = 0.05,
    power: float = 0.80,
    seed: int = 1,
) -> tuple[ScenarioResult, ScenarioResult]:
    """Validate Welch's t-test's Type I error and power."""
    rng = np.random.default_rng(seed)
    n = sample_size_for_mean(std, effect, alpha=alpha, power=power).per_arm

    false_positives = sum(
        two_sample_ttest(
            rng.normal(0, std, n), rng.normal(0, std, n), alpha=alpha
        ).significant
        for _ in range(n_experiments)
    )
    type_i = ScenarioResult(
        "Welch's t-test", "Type I error", alpha,
        false_positives / n_experiments, n_experiments, n, tolerance=0.025,
    )

    detections = sum(
        two_sample_ttest(
            rng.normal(0, std, n), rng.normal(effect, std, n), alpha=alpha
        ).significant
        for _ in range(n_experiments)
    )
    power_result = ScenarioResult(
        "Welch's t-test", "Power", power,
        detections / n_experiments, n_experiments, n, tolerance=0.05,
    )
    return type_i, power_result


def validate_engine(n_experiments: int = 500, *, seed: int = 0) -> ValidationReport:
    """Run every validation scenario and return an aggregate report."""
    p_type_i, p_power = validate_proportion(n_experiments, seed=seed)
    c_type_i, c_power = validate_continuous(n_experiments, seed=seed + 1)
    return ValidationReport([p_type_i, p_power, c_type_i, c_power])


if __name__ == "__main__":
    print("\nRunning 500-experiment validation (this takes a few seconds)…\n")
    print(validate_engine(500))
