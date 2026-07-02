"""Tests for the reporting layer: report generation, insights, charts, PDF, scheduling."""

import time

import pytest

from insightflow.core import (
    beta_binomial_test,
    detect_srm,
    proportion_ztest,
    two_sample_ttest,
)
from insightflow.reporting import (
    ExperimentReport,
    ReportScheduler,
    TemplateProvider,
    generate_insight,
    generate_report,
    get_provider,
    report_to_pdf_bytes,
)


# ── report generation ────────────────────────────────────────────────────────
def _shipping_report() -> ExperimentReport:
    ztest = proportion_ztest(500, 5000, 650, 5000)      # a real +30% lift
    bayes = beta_binomial_test(500, 5000, 650, 5000)
    srm = detect_srm({"control": 5000, "treatment": 5000})
    return generate_report(
        name="Checkout redesign", metric_type="proportion", status="running",
        frequentist=[ztest], srm=srm, bayesian=bayes,
    )


def test_report_recommends_ship_on_real_lift():
    report = _shipping_report()
    assert report.recommendation == "SHIP"
    assert report.significant
    assert report.bayesian["prob_treatment_best"] > 0.95
    assert "ship" in report.headline.lower()


def test_report_flags_invalid_on_srm():
    ztest = proportion_ztest(500, 5000, 650, 5000)
    srm = detect_srm({"control": 6000, "treatment": 4000})  # broken split
    report = generate_report(
        name="Broken", metric_type="proportion", status="running",
        frequentist=[ztest], srm=srm,
    )
    assert report.recommendation == "INVALID"


def test_report_extends_when_underpowered():
    # Not significant + far below required sample size -> EXTEND.
    ztest = proportion_ztest(100, 1000, 105, 1000)
    report = generate_report(
        name="Early", metric_type="proportion", status="running",
        frequentist=[ztest], required_sample_size_per_arm=14000,
    )
    assert not report.significant
    assert report.recommendation == "EXTEND"


def test_report_do_not_ship_on_regression():
    ztest = proportion_ztest(650, 5000, 500, 5000)  # treatment worse
    report = generate_report(
        name="Regression", metric_type="proportion", status="running", frequentist=[ztest],
    )
    assert report.recommendation == "DO NOT SHIP"


def test_report_requires_a_test():
    with pytest.raises(ValueError):
        generate_report(name="x", metric_type="proportion", status="draft", frequentist=[])


def test_report_serializes_and_renders():
    report = _shipping_report()
    d = report.to_dict()
    assert d["recommendation"] == "SHIP"
    assert isinstance(report.text_summary(), str)
    assert "RECOMMENDATION" in report.text_summary()


# ── insight generation (LLM layer, free/offline path) ────────────────────────
def test_default_provider_is_template():
    assert isinstance(get_provider(), TemplateProvider)


def test_template_insight_mentions_key_facts():
    report = _shipping_report()
    text = generate_insight(report)  # defaults to template - no network, no key
    assert report.narrative == text
    assert len(text) > 40
    assert "ship" in text.lower()


def test_insight_falls_back_when_provider_fails():
    # A Groq provider with no key must not raise - it falls back to the template.
    from insightflow.reporting.insight_generator import GroqProvider

    report = _shipping_report()
    text = generate_insight(report, provider=GroqProvider())
    assert isinstance(text, str) and len(text) > 40


# ── charts ───────────────────────────────────────────────────────────────────
def test_visualizations_build_figures():
    import plotly.graph_objects as go

    from insightflow.reporting import visualizations as viz

    assert isinstance(viz.confidence_interval_plot(0.02, 0.01, 0.03), go.Figure)
    assert isinstance(viz.conversion_bar(0.10, 0.13), go.Figure)
    assert isinstance(viz.sprt_trace([0.0, 0.5, 1.2, 2.9], upper=2.7, lower=-1.5), go.Figure)
    assert isinstance(viz.posterior_plot(501, 4501, 651, 4351), go.Figure)
    fig = viz.uplift_quantile_bar([1, 2, 3, 4, 5], [0.5, 1.0, 1.5, 2.0, 3.0])
    assert isinstance(fig, go.Figure)
    assert viz.figure_to_json(fig).startswith("{")


# ── PDF export ───────────────────────────────────────────────────────────────
def test_pdf_export_produces_valid_bytes():
    report = _shipping_report()
    generate_insight(report)
    pdf = report_to_pdf_bytes(report)
    assert pdf[:5] == b"%PDF-"     # PDF magic number
    assert len(pdf) > 1500          # non-trivial document


def test_pdf_works_for_continuous_metric():
    ttest = two_sample_ttest([1, 2, 3, 4, 5] * 40, [3, 4, 5, 6, 7] * 40)
    report = generate_report(
        name="Revenue", metric_type="continuous", status="completed", frequentist=[ttest],
    )
    assert report_to_pdf_bytes(report)[:5] == b"%PDF-"


# ── scheduler ────────────────────────────────────────────────────────────────
def test_scheduler_registers_and_runs_job():
    scheduler = ReportScheduler()
    hits = []
    scheduler.schedule_interval(lambda: hits.append(1), job_id="digest", seconds=1)
    assert len(scheduler.jobs) == 1

    scheduler.start()
    try:
        # Give the background thread a moment to fire the 1-second job at least once.
        deadline = time.time() + 4
        while not hits and time.time() < deadline:
            time.sleep(0.1)
    finally:
        scheduler.shutdown(wait=False)
    assert hits, "scheduled job did not run"


def test_weekly_schedule_registers():
    scheduler = ReportScheduler()
    scheduler.schedule_weekly(lambda: None, job_id="weekly", day_of_week="mon", hour=8)
    assert len(scheduler.jobs) == 1
