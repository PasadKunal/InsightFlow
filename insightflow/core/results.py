"""Shared result containers for the statistical engine.

Every test in InsightFlow returns a small, self-describing dataclass instead of a
bare number. That way a caller (an API route, a report generator, or a human
reading the REPL) always gets the *full picture*: the estimate, how uncertain we
are about it, and a plain-English verdict — not just a p-value floating in space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ConfidenceInterval:
    """A two-sided confidence interval for an estimate."""

    lower: float
    upper: float
    confidence: float = 0.95  # e.g. 0.95 for a 95% interval

    def contains(self, value: float) -> bool:
        """True if `value` (often 0, "no effect") falls inside the interval."""
        return self.lower <= value <= self.upper

    def __str__(self) -> str:
        pct = int(round(self.confidence * 100))
        return f"{pct}% CI [{self.lower:.4g}, {self.upper:.4g}]"


@dataclass(frozen=True)
class TestResult:
    """The outcome of a single frequentist hypothesis test.

    Attributes
    ----------
    test_name:
        Human-readable name, e.g. "Welch's two-sample t-test".
    statistic:
        The test statistic (t, chi-squared, U, z, ...).
    p_value:
        Two-sided p-value unless the test is inherently one-sided.
    effect_size:
        A standardized or business-facing measure of *how big* the difference is
        (Cohen's d, relative lift, ...). Significance without effect size is a trap.
    effect_size_name:
        What `effect_size` actually measures, so nobody has to guess.
    confidence_interval:
        Interval around the primary effect estimate, when one is defined.
    alpha:
        The significance threshold the verdict was made against.
    n_control, n_treatment:
        Sample sizes that went into the test — essential context for trust.
    extra:
        Free-form bag for test-specific extras (means, rates, etc.).
    """

    test_name: str
    statistic: float
    p_value: float
    effect_size: Optional[float] = None
    effect_size_name: Optional[str] = None
    confidence_interval: Optional[ConfidenceInterval] = None
    alpha: float = 0.05
    n_control: Optional[int] = None
    n_treatment: Optional[int] = None
    extra: dict = field(default_factory=dict)

    @property
    def significant(self) -> bool:
        """Did we reject the null hypothesis at the chosen alpha?"""
        return self.p_value < self.alpha

    def summary(self) -> str:
        """A one-line, human-readable summary of the result."""
        verdict = "SIGNIFICANT" if self.significant else "not significant"
        parts = [f"{self.test_name}: {verdict} (p={self.p_value:.4g}, alpha={self.alpha})"]
        if self.effect_size is not None:
            parts.append(f"{self.effect_size_name}={self.effect_size:.4g}")
        if self.confidence_interval is not None:
            parts.append(str(self.confidence_interval))
        return " | ".join(parts)

    def __str__(self) -> str:
        return self.summary()
