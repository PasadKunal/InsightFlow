"""Turning raw statistics into a decision-ready experiment report.

The output of the engine is a pile of numbers: p-values, credible intervals, effect
sizes, an SRM check. A stakeholder doesn't want the pile - they want **the decision**
and the few facts that justify it. This module assembles everything into a single
``ExperimentReport``: one clear recommendation (SHIP / DO NOT SHIP / EXTEND / INVALID)
plus the supporting evidence, ready to be rendered as text, JSON, a chart, or a PDF.

It is deliberately decoupled from the web and ML layers - it consumes the plain result
objects from ``insightflow.core`` (and, optionally, a segment breakdown), so it can be
unit-tested on its own with no database or model training in sight.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from insightflow.core.bayesian import BayesianResult
from insightflow.core.results import TestResult
from insightflow.core.srm_detector import SRMResult


# The four decisions a report can reach, most-actionable first.
SHIP = "SHIP"
DO_NOT_SHIP = "DO NOT SHIP"
EXTEND = "EXTEND"
INVALID = "INVALID"


@dataclass
class ExperimentReport:
    """A complete, decision-ready summary of one experiment."""

    name: str
    metric_type: str
    status: str
    n_control: int
    n_treatment: int

    recommendation: str
    headline: str  # one-sentence plain-English takeaway

    primary_test_name: str
    p_value: float
    significant: bool
    effect_size: float | None
    effect_size_name: str | None
    ci_lower: float | None
    ci_upper: float | None

    all_tests: list[dict] = field(default_factory=list)
    bayesian: dict | None = None
    srm: dict | None = None
    segment_breakdown: list[dict] | None = None
    top_quintile_lift: float | None = None
    required_sample_size_per_arm: int | None = None

    narrative: str | None = None  # filled in by the LLM insight generator, if used
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── serialization / rendering ────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["generated_at"] = self.generated_at.isoformat()
        return d

    def text_summary(self) -> str:
        """A clean plain-text report - used for the console, PDFs, and LLM prompts."""
        lines = [
            f"EXPERIMENT REPORT - {self.name}",
            f"Status: {self.status} | Metric: {self.metric_type} | "
            f"n = {self.n_control:,} control / {self.n_treatment:,} treatment",
            "",
            f"RECOMMENDATION: {self.recommendation}",
            f"  {self.headline}",
            "",
            "Primary test:",
            f"  {self.primary_test_name}: p = {self.p_value:.4g} "
            f"({'significant' if self.significant else 'not significant'})",
        ]
        if self.effect_size is not None:
            lines.append(f"  {self.effect_size_name} = {self.effect_size:.4g}")
        if self.ci_lower is not None:
            lines.append(f"  95% CI = [{self.ci_lower:.4g}, {self.ci_upper:.4g}]")
        if self.bayesian:
            lines += [
                "",
                "Bayesian view:",
                f"  P(treatment best) = {self.bayesian['prob_treatment_best']:.1%}",
                f"  expected uplift   = {self.bayesian['expected_relative_uplift']:+.1%} (relative)",
                f"  expected loss     = {self.bayesian['expected_loss']:.4g}",
            ]
        if self.srm:
            state = "MISMATCH DETECTED" if self.srm["mismatch_detected"] else "healthy"
            lines += ["", f"Data quality (SRM): {state} (p = {self.srm['p_value']:.3g})"]
        if self.top_quintile_lift is not None:
            lines += [
                "",
                f"Uplift: top-responder quintile shows {self.top_quintile_lift:.2f}x "
                f"the overall treatment effect.",
            ]
        if self.narrative:
            lines += ["", "Summary:", self.narrative]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.text_summary()


def _decide(
    srm: SRMResult | None,
    primary: TestResult,
    n_per_arm_min: int,
    required_sample_size_per_arm: int | None,
) -> tuple[str, str]:
    """Return (recommendation, one-sentence headline) from the evidence."""
    if srm is not None and srm.mismatch_detected:
        return (
            INVALID,
            "A Sample Ratio Mismatch was detected - the assignment split is broken, so "
            "these results cannot be trusted. Fix the pipeline and rerun.",
        )

    if primary.significant:
        if (primary.effect_size or 0) > 0:
            return (
                SHIP,
                f"Treatment shows a statistically significant improvement "
                f"(p = {primary.p_value:.3g}); recommend shipping.",
            )
        return (
            DO_NOT_SHIP,
            f"Treatment is significantly worse than control "
            f"(p = {primary.p_value:.3g}); do not ship.",
        )

    # Not significant: extend if we haven't reached the planned sample size yet.
    if required_sample_size_per_arm and n_per_arm_min < required_sample_size_per_arm:
        return (
            EXTEND,
            f"No significant difference yet, and the experiment is underpowered "
            f"({n_per_arm_min:,} of {required_sample_size_per_arm:,} needed per arm) - "
            f"keep collecting data.",
        )
    return (
        DO_NOT_SHIP,
        "No statistically significant difference at the planned sample size; "
        "treatment does not beat control.",
    )


def generate_report(
    *,
    name: str,
    metric_type: str,
    status: str,
    frequentist: list[TestResult],
    srm: SRMResult | None = None,
    bayesian: BayesianResult | None = None,
    segment_report: Any | None = None,
    required_sample_size_per_arm: int | None = None,
) -> ExperimentReport:
    """Assemble an :class:`ExperimentReport` from core result objects.

    ``frequentist[0]`` is treated as the primary test (the one the ship/hold decision
    hinges on); any further tests are included as supporting evidence. ``segment_report``
    is duck-typed: anything exposing ``.table`` (a DataFrame) and ``.top_quantile_lift``
    works, so the uplift layer plugs in without this module importing it.
    """
    if not frequentist:
        raise ValueError("At least one frequentist test is required for a report.")

    primary = frequentist[0]
    n_c = primary.n_control or 0
    n_t = primary.n_treatment or 0
    n_min = min(n_c, n_t)

    recommendation, headline = _decide(srm, primary, n_min, required_sample_size_per_arm)

    all_tests = [
        {
            "test_name": t.test_name,
            "p_value": t.p_value,
            "significant": t.significant,
            "effect_size": t.effect_size,
            "effect_size_name": t.effect_size_name,
        }
        for t in frequentist
    ]

    bayesian_dict = None
    if bayesian is not None:
        bayesian_dict = {
            "prob_treatment_best": bayesian.prob_treatment_best,
            "expected_relative_uplift": bayesian.expected_relative_uplift,
            "expected_loss": bayesian.expected_loss,
            "recommendation": bayesian.recommendation,
        }

    srm_dict = None
    if srm is not None:
        srm_dict = {
            "mismatch_detected": srm.mismatch_detected,
            "p_value": srm.p_value,
            "observed": srm.observed,
        }

    segment_breakdown = None
    top_quintile_lift = None
    if segment_report is not None and hasattr(segment_report, "table"):
        segment_breakdown = segment_report.table.to_dict(orient="records")
        top_quintile_lift = getattr(segment_report, "top_quantile_lift", None)

    ci = primary.confidence_interval
    return ExperimentReport(
        name=name,
        metric_type=metric_type,
        status=status,
        n_control=n_c,
        n_treatment=n_t,
        recommendation=recommendation,
        headline=headline,
        primary_test_name=primary.test_name,
        p_value=primary.p_value,
        significant=primary.significant,
        effect_size=primary.effect_size,
        effect_size_name=primary.effect_size_name,
        ci_lower=ci.lower if ci else None,
        ci_upper=ci.upper if ci else None,
        all_tests=all_tests,
        bayesian=bayesian_dict,
        srm=srm_dict,
        segment_breakdown=segment_breakdown,
        top_quintile_lift=top_quintile_lift,
        required_sample_size_per_arm=required_sample_size_per_arm,
    )
