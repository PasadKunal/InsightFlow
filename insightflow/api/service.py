"""Service layer — business logic that sits between the HTTP routes and the database.

Routes stay thin (parse, call, return); the real work lives here. Most importantly,
this is where stored rows get handed to the Phase 1-2 statistical engine to produce
an analysis. The web layer never does statistics itself — it delegates to ``core``.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from insightflow.core import (
    beta_binomial_test,
    chi_squared_test,
    detect_srm,
    mann_whitney_u,
    proportion_ztest,
    sample_size_for_proportion,
    two_sample_ttest,
)
from insightflow.core.randomization import assign as deterministic_assign

from . import models, schemas


# ── Experiment lifecycle ─────────────────────────────────────────────────────
def create_experiment(db: Session, payload: schemas.ExperimentCreate) -> models.Experiment:
    """Persist a new experiment, running a power analysis when we have the inputs."""
    required_n = None
    if (
        payload.metric_type == models.MetricType.PROPORTION
        and payload.baseline_rate is not None
        and payload.minimum_detectable_effect is not None
    ):
        plan = sample_size_for_proportion(
            payload.baseline_rate,
            payload.minimum_detectable_effect,
            alpha=payload.alpha,
            power=payload.power,
        )
        required_n = plan.per_arm

    experiment = models.Experiment(
        name=payload.name,
        hypothesis=payload.hypothesis,
        metric_type=payload.metric_type,
        treatment_fraction=payload.treatment_fraction,
        baseline_rate=payload.baseline_rate,
        minimum_detectable_effect=payload.minimum_detectable_effect,
        alpha=payload.alpha,
        power=payload.power,
        required_sample_size_per_arm=required_n,
    )
    db.add(experiment)
    db.commit()
    db.refresh(experiment)
    return experiment


def get_experiment(db: Session, experiment_id: str) -> models.Experiment | None:
    return db.get(models.Experiment, experiment_id)


def list_experiments(db: Session) -> list[models.Experiment]:
    return list(db.scalars(select(models.Experiment).order_by(models.Experiment.created_at.desc())))


def set_status(
    db: Session, experiment: models.Experiment, status: models.ExperimentStatus
) -> models.Experiment:
    experiment.status = status
    db.commit()
    db.refresh(experiment)
    return experiment


def delete_experiment(db: Session, experiment: models.Experiment) -> None:
    db.delete(experiment)
    db.commit()


# ── Assignment & observation ─────────────────────────────────────────────────
def assign_user(db: Session, experiment: models.Experiment, user_id: str) -> models.Assignment:
    """Deterministically assign a user to an arm; idempotent (returns any existing row).

    The arm is decided by the same hash-based function used everywhere else in the
    platform, so an assignment is reproducible and independent of call order.
    """
    existing = db.scalar(
        select(models.Assignment).where(
            models.Assignment.experiment_id == experiment.id,
            models.Assignment.user_id == user_id,
        )
    )
    if existing is not None:
        return existing

    variant = deterministic_assign(
        user_id,
        experiment_id=experiment.id,
        treatment_fraction=experiment.treatment_fraction,
    )
    assignment = models.Assignment(
        experiment_id=experiment.id, user_id=user_id, variant=variant
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return assignment


def record_observation(
    db: Session, experiment: models.Experiment, user_id: str, value: float
) -> models.Observation:
    """Record one metric value, auto-assigning the user first if needed."""
    assignment = assign_user(db, experiment, user_id)
    observation = models.Observation(
        experiment_id=experiment.id,
        user_id=user_id,
        variant=assignment.variant,
        value=value,
    )
    db.add(observation)
    db.commit()
    db.refresh(observation)
    return observation


def bulk_record(
    db: Session, experiment: models.Experiment, items: list[schemas.ObserveRequest]
) -> schemas.IngestSummary:
    """Efficiently ingest many observations (used for seeding and demos)."""
    control = treatment = 0
    for item in items:
        variant = deterministic_assign(
            item.user_id,
            experiment_id=experiment.id,
            treatment_fraction=experiment.treatment_fraction,
        )
        # Ensure an assignment row exists (ignore if already there).
        exists = db.scalar(
            select(models.Assignment.id).where(
                models.Assignment.experiment_id == experiment.id,
                models.Assignment.user_id == item.user_id,
            )
        )
        if exists is None:
            db.add(
                models.Assignment(
                    experiment_id=experiment.id, user_id=item.user_id, variant=variant
                )
            )
        db.add(
            models.Observation(
                experiment_id=experiment.id,
                user_id=item.user_id,
                variant=variant,
                value=item.value,
            )
        )
        if variant == "treatment":
            treatment += 1
        else:
            control += 1
    db.commit()
    return schemas.IngestSummary(ingested=len(items), control=control, treatment=treatment)


# ── Analysis ─────────────────────────────────────────────────────────────────
def _assignment_counts(db: Session, experiment_id: str) -> dict[str, int]:
    rows = db.execute(
        select(models.Assignment.variant, func.count())
        .where(models.Assignment.experiment_id == experiment_id)
        .group_by(models.Assignment.variant)
    ).all()
    counts = {"control": 0, "treatment": 0}
    for variant, n in rows:
        counts[variant] = n
    return counts


def _values_by_variant(db: Session, experiment_id: str) -> dict[str, list[float]]:
    rows = db.execute(
        select(models.Observation.variant, models.Observation.value).where(
            models.Observation.experiment_id == experiment_id
        )
    ).all()
    out: dict[str, list[float]] = {"control": [], "treatment": []}
    for variant, value in rows:
        out.setdefault(variant, []).append(value)
    return out


@dataclass
class _AnalysisBundle:
    """Raw core result objects for an experiment — shared by results, reports & charts."""

    srm: object
    frequentist: list          # list of core TestResult objects (primary first)
    bayesian: object | None    # core BayesianResult or None
    n_control: int
    n_treatment: int


def _run_analysis(db: Session, experiment: models.Experiment) -> _AnalysisBundle:
    """The single analysis path: stored rows -> core statistical result objects.

    Every consumer (the JSON results endpoint, the report, the charts) goes through
    here, so the numbers can never disagree between surfaces. Raises ValueError if
    either arm has no data yet.
    """
    values = _values_by_variant(db, experiment.id)
    control_vals = values.get("control", [])
    treatment_vals = values.get("treatment", [])
    n_c, n_t = len(control_vals), len(treatment_vals)
    if n_c == 0 or n_t == 0:
        raise ValueError("Both arms need at least one observation before analysis.")

    # Guardrail — is the assignment split what we designed for?
    counts = _assignment_counts(db, experiment.id)
    tf = experiment.treatment_fraction
    srm = detect_srm(
        {"control": counts["control"], "treatment": counts["treatment"]},
        expected_ratio={"control": 1 - tf, "treatment": tf},
    )

    frequentist: list = []
    bayesian = None
    if experiment.metric_type == models.MetricType.PROPORTION:
        conv_c, conv_t = int(sum(control_vals)), int(sum(treatment_vals))
        frequentist = [
            proportion_ztest(conv_c, n_c, conv_t, n_t, alpha=experiment.alpha),
            chi_squared_test(conv_c, n_c, conv_t, n_t, alpha=experiment.alpha),
        ]
        bayesian = beta_binomial_test(conv_c, n_c, conv_t, n_t)
    else:
        frequentist = [
            two_sample_ttest(control_vals, treatment_vals, alpha=experiment.alpha),
            mann_whitney_u(control_vals, treatment_vals, alpha=experiment.alpha),
        ]

    return _AnalysisBundle(
        srm=srm, frequentist=frequentist, bayesian=bayesian, n_control=n_c, n_treatment=n_t
    )


def analyze(db: Session, experiment: models.Experiment) -> schemas.ResultsOut:
    """Turn stored rows into a full statistical read-out (SRM + tests + Bayesian + verdict)."""
    bundle = _run_analysis(db, experiment)
    srm = bundle.srm
    primary = bundle.frequentist[0]

    srm_out = schemas.SRMOut(
        mismatch_detected=srm.mismatch_detected,
        p_value=srm.p_value,
        observed=srm.observed,
        expected=srm.expected,
        message=srm.summary(),
    )
    bayesian_out = None
    if bundle.bayesian is not None:
        bayesian_out = schemas.BayesianOut(
            prob_treatment_best=bundle.bayesian.prob_treatment_best,
            expected_relative_uplift=bundle.bayesian.expected_relative_uplift,
            expected_loss=bundle.bayesian.expected_loss,
            recommendation=bundle.bayesian.recommendation,
        )

    return schemas.ResultsOut(
        experiment_id=experiment.id,
        metric_type=experiment.metric_type,
        status=experiment.status,
        n_control=bundle.n_control,
        n_treatment=bundle.n_treatment,
        srm=srm_out,
        frequentist=[_to_frequentist_out(t) for t in bundle.frequentist],
        bayesian=bayesian_out,
        recommendation=_recommend(srm.mismatch_detected, primary),
    )


def build_report(db: Session, experiment: models.Experiment):
    """Build a full ExperimentReport (with a natural-language narrative) for an experiment.

    Reporting deps are imported lazily so the base API still runs without them.
    """
    from insightflow.reporting import generate_insight, generate_report  # lazy import

    bundle = _run_analysis(db, experiment)
    report = generate_report(
        name=experiment.name,
        metric_type=experiment.metric_type.value,
        status=experiment.status.value,
        frequentist=bundle.frequentist,
        srm=bundle.srm,
        bayesian=bundle.bayesian,
        required_sample_size_per_arm=experiment.required_sample_size_per_arm,
    )
    generate_insight(report)  # attaches report.narrative (free template by default)
    return report


def build_charts(db: Session, experiment: models.Experiment) -> dict:
    """Build Plotly figures for an experiment, returned as JSON-ready dicts."""
    import json

    from insightflow.reporting import visualizations as viz  # lazy import

    bundle = _run_analysis(db, experiment)
    primary = bundle.frequentist[0]
    charts: dict[str, dict] = {}

    if experiment.metric_type == models.MetricType.PROPORTION:
        charts["conversion_rate"] = json.loads(
            viz.conversion_bar(
                primary.extra["rate_control"], primary.extra["rate_treatment"]
            ).to_json()
        )
        if bundle.bayesian is not None:
            b = bundle.bayesian
            charts["posteriors"] = json.loads(
                viz.posterior_plot(
                    b.control.alpha, b.control.beta, b.treatment.alpha, b.treatment.beta
                ).to_json()
            )
    else:
        ci = primary.confidence_interval
        charts["effect_ci"] = json.loads(
            viz.confidence_interval_plot(
                primary.extra["mean_difference"], ci.lower, ci.upper, label="mean diff"
            ).to_json()
        )
    return charts


def _to_frequentist_out(res) -> schemas.FrequentistOut:
    ci = res.confidence_interval
    return schemas.FrequentistOut(
        test_name=res.test_name,
        statistic=res.statistic,
        p_value=res.p_value,
        significant=res.significant,
        effect_size=res.effect_size,
        effect_size_name=res.effect_size_name,
        ci_lower=ci.lower if ci else None,
        ci_upper=ci.upper if ci else None,
        extra=res.extra,
    )


def _recommend(srm_detected: bool, primary) -> str:
    """Distil the statistics into a single, human decision."""
    if srm_detected:
        return "INVALID — Sample Ratio Mismatch detected. Fix the assignment pipeline; do not trust these results."
    if not primary.significant:
        return "INCONCLUSIVE — no significant difference yet. Keep running or increase sample size."
    # Significant: direction comes from the effect size (positive = treatment better).
    if (primary.effect_size or 0) > 0:
        return "SHIP — treatment shows a statistically significant improvement."
    return "DO NOT SHIP — treatment is significantly worse than control."
