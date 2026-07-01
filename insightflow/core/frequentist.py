"""Frequentist hypothesis tests for A/B experiments.

This module is the workhorse of InsightFlow. It covers the three shapes of data
you actually meet in product experiments:

* **Continuous metrics** (revenue per user, session length) -> two-sample t-test.
* **Proportions / conversion rates** (signup rate, click rate) -> two-proportion
  z-test and its chi-squared twin.
* **Skewed / non-normal metrics** (where the mean lies) -> Mann-Whitney U.

Every function returns a :class:`TestResult` carrying an effect size, because a
p-value alone never answers the real question: *is this difference big enough to
care about?*
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from .results import ConfidenceInterval, TestResult

# A type alias for "anything that looks like a 1-D sequence of numbers".
ArrayLike = np.ndarray | list | tuple


def _as_array(x: ArrayLike, name: str) -> np.ndarray:
    """Coerce input to a clean 1-D float array, dropping NaNs, with clear errors."""
    arr = np.asarray(x, dtype=float).ravel()
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        raise ValueError(f"'{name}' is empty after removing NaNs — nothing to test.")
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# Continuous metrics: two-sample t-test
# ─────────────────────────────────────────────────────────────────────────────
def two_sample_ttest(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    alpha: float = 0.05,
    equal_var: bool = False,
) -> TestResult:
    """Compare the means of two independent groups.

    By default this runs **Welch's t-test** (``equal_var=False``), which does not
    assume the two groups share a variance. Welch is the sane default for real
    experiments where treatment can change spread, not just center.

    The effect size is **Cohen's d** (mean difference in pooled-standard-deviation
    units): ~0.2 small, ~0.5 medium, ~0.8 large.

    Parameters
    ----------
    control, treatment:
        Per-user metric values for each arm.
    alpha:
        Significance threshold for the verdict.
    equal_var:
        If True, run Student's pooled-variance t-test instead of Welch's.
    """
    c = _as_array(control, "control")
    t = _as_array(treatment, "treatment")

    statistic, p_value = stats.ttest_ind(t, c, equal_var=equal_var)

    mean_c, mean_t = c.mean(), t.mean()
    mean_diff = mean_t - mean_c

    # Cohen's d with a pooled standard deviation.
    n_c, n_t = c.size, t.size
    var_c, var_t = c.var(ddof=1), t.var(ddof=1)
    pooled_sd = np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2))
    cohens_d = mean_diff / pooled_sd if pooled_sd > 0 else 0.0

    # Confidence interval for the difference in means (Welch standard error).
    se_diff = np.sqrt(var_c / n_c + var_t / n_t)
    if equal_var:
        df = n_c + n_t - 2
    else:
        # Welch–Satterthwaite degrees of freedom.
        df = (var_c / n_c + var_t / n_t) ** 2 / (
            (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
        )
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci = ConfidenceInterval(
        lower=mean_diff - t_crit * se_diff,
        upper=mean_diff + t_crit * se_diff,
        confidence=1 - alpha,
    )

    relative_lift = mean_diff / mean_c if mean_c != 0 else float("nan")

    return TestResult(
        test_name="Welch's two-sample t-test" if not equal_var else "Student's two-sample t-test",
        statistic=float(statistic),
        p_value=float(p_value),
        effect_size=float(cohens_d),
        effect_size_name="Cohen's d",
        confidence_interval=ci,
        alpha=alpha,
        n_control=n_c,
        n_treatment=n_t,
        extra={
            "mean_control": float(mean_c),
            "mean_treatment": float(mean_t),
            "mean_difference": float(mean_diff),
            "relative_lift": float(relative_lift),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Proportions: two-proportion z-test
# ─────────────────────────────────────────────────────────────────────────────
def proportion_ztest(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Compare two conversion rates with a two-proportion z-test.

    This is the test for "did the treatment change the *rate* at which users do
    X?" — click, sign up, purchase. The effect size reported is **relative lift**
    (the metric stakeholders actually ask about), and the confidence interval is
    on the absolute difference in rates.
    """
    if not (0 <= conversions_control <= n_control):
        raise ValueError("conversions_control must be between 0 and n_control.")
    if not (0 <= conversions_treatment <= n_treatment):
        raise ValueError("conversions_treatment must be between 0 and n_treatment.")
    if n_control == 0 or n_treatment == 0:
        raise ValueError("Sample sizes must be positive.")

    p_c = conversions_control / n_control
    p_t = conversions_treatment / n_treatment
    diff = p_t - p_c

    # Pooled proportion under the null (rates are equal) for the z-statistic.
    p_pool = (conversions_control + conversions_treatment) / (n_control + n_treatment)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # Unpooled SE for the confidence interval on the observed difference.
    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = ConfidenceInterval(
        lower=diff - z_crit * se_unpooled,
        upper=diff + z_crit * se_unpooled,
        confidence=1 - alpha,
    )

    relative_lift = diff / p_c if p_c != 0 else float("nan")

    return TestResult(
        test_name="Two-proportion z-test",
        statistic=float(z),
        p_value=float(p_value),
        effect_size=float(relative_lift),
        effect_size_name="relative lift",
        confidence_interval=ci,
        alpha=alpha,
        n_control=n_control,
        n_treatment=n_treatment,
        extra={
            "rate_control": float(p_c),
            "rate_treatment": float(p_t),
            "absolute_difference": float(diff),
        },
    )


def chi_squared_test(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Independence test on a 2x2 (arm x converted?) contingency table.

    Mathematically equivalent to the two-proportion z-test for large samples, but
    framed as a chi-squared test — the form many analysts expect to see, and the
    one that generalizes to more than two categories. The effect size is
    **Cramér's V** (0 = no association, 1 = perfect).
    """
    if n_control == 0 or n_treatment == 0:
        raise ValueError("Sample sizes must be positive.")

    # Rows = arm, columns = [converted, not converted].
    table = np.array(
        [
            [conversions_control, n_control - conversions_control],
            [conversions_treatment, n_treatment - conversions_treatment],
        ]
    )
    chi2, p_value, dof, _expected = stats.chi2_contingency(table, correction=False)

    n_total = n_control + n_treatment
    cramers_v = np.sqrt(chi2 / n_total)  # min(rows,cols)-1 == 1 for a 2x2 table

    return TestResult(
        test_name="Chi-squared test of independence",
        statistic=float(chi2),
        p_value=float(p_value),
        effect_size=float(cramers_v),
        effect_size_name="Cramér's V",
        alpha=alpha,
        n_control=n_control,
        n_treatment=n_treatment,
        extra={
            "degrees_of_freedom": int(dof),
            "rate_control": conversions_control / n_control,
            "rate_treatment": conversions_treatment / n_treatment,
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Non-normal metrics: Mann-Whitney U
# ─────────────────────────────────────────────────────────────────────────────
def mann_whitney_u(
    control: ArrayLike,
    treatment: ArrayLike,
    *,
    alpha: float = 0.05,
) -> TestResult:
    """Rank-based test for whether one group tends to produce larger values.

    Use this when the metric is heavily skewed or has fat tails (revenue, time on
    page) and the *mean* is a misleading summary. It tests distributions, not
    means, and needs no normality assumption. The effect size is the **rank-biserial
    correlation** (-1..1): the probability that a random treatment value exceeds a
    random control value, rescaled.
    """
    c = _as_array(control, "control")
    t = _as_array(treatment, "treatment")

    u_statistic, p_value = stats.mannwhitneyu(t, c, alternative="two-sided")

    # Rank-biserial correlation from the U statistic.
    n_c, n_t = c.size, t.size
    rank_biserial = 1 - (2 * u_statistic) / (n_c * n_t)

    return TestResult(
        test_name="Mann-Whitney U test",
        statistic=float(u_statistic),
        p_value=float(p_value),
        effect_size=float(rank_biserial),
        effect_size_name="rank-biserial correlation",
        alpha=alpha,
        n_control=n_c,
        n_treatment=n_t,
        extra={
            "median_control": float(np.median(c)),
            "median_treatment": float(np.median(t)),
        },
    )
