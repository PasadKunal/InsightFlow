"""Service layer — business logic that sits between the HTTP routes and the database.

Routes stay thin (parse, call, return); the real work lives here. Most importantly,
this is where stored rows get handed to the Phase 1-2 statistical engine to produce
an analysis. The web layer never does statistics itself — it delegates to ``core``.
"""

from __future__ import annotations

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


def analyze(db: Session, experiment: models.Experiment) -> schemas.ResultsOut:
    """Turn stored rows into a full statistical read-out for the experiment.

    Runs, in order: an SRM guardrail on the assignment split, the frequentist test(s)
    appropriate to the metric type, and (for proportions) the Bayesian view — then
    distills everything into a single ship / hold recommendation. Raises ValueError
    if either arm has no data yet.
    """
    values = _values_by_variant(db, experiment.id)
    control_vals = values.get("control", [])
    treatment_vals = values.get("treatment", [])
    n_c, n_t = len(control_vals), len(treatment_vals)
    if n_c == 0 or n_t == 0:
        raise ValueError("Both arms need at least one observation before analysis.")

    # 1. Guardrail — is the assignment split what we designed for?
    counts = _assignment_counts(db, experiment.id)
    tf = experiment.treatment_fraction
    srm = detect_srm(
        {"control": counts["control"], "treatment": counts["treatment"]},
        expected_ratio={"control": 1 - tf, "treatment": tf},
    )
    srm_out = schemas.SRMOut(
        mismatch_detected=srm.mismatch_detected,
        p_value=srm.p_value,
        observed=srm.observed,
        expected=srm.expected,
        message=srm.summary(),
    )

    # 2. Frequentist test(s), chosen by metric type.
    frequentist: list[schemas.FrequentistOut] = []
    primary = None
    if experiment.metric_type == models.MetricType.PROPORTION:
        conv_c, conv_t = int(sum(control_vals)), int(sum(treatment_vals))
        ztest = proportion_ztest(conv_c, n_c, conv_t, n_t, alpha=experiment.alpha)
        chi = chi_squared_test(conv_c, n_c, conv_t, n_t, alpha=experiment.alpha)
        primary = ztest
        for res in (ztest, chi):
            frequentist.append(_to_frequentist_out(res))
    else:
        ttest = two_sample_ttest(control_vals, treatment_vals, alpha=experiment.alpha)
        mwu = mann_whitney_u(control_vals, treatment_vals, alpha=experiment.alpha)
        primary = ttest
        for res in (ttest, mwu):
            frequentist.append(_to_frequentist_out(res))

    # 3. Bayesian view — only meaningful for conversion-style metrics.
    bayesian_out = None
    if experiment.metric_type == models.MetricType.PROPORTION:
        conv_c, conv_t = int(sum(control_vals)), int(sum(treatment_vals))
        bayes = beta_binomial_test(conv_c, n_c, conv_t, n_t)
        bayesian_out = schemas.BayesianOut(
            prob_treatment_best=bayes.prob_treatment_best,
            expected_relative_uplift=bayes.expected_relative_uplift,
            expected_loss=bayes.expected_loss,
            recommendation=bayes.recommendation,
        )

    recommendation = _recommend(srm.mismatch_detected, primary)

    return schemas.ResultsOut(
        experiment_id=experiment.id,
        metric_type=experiment.metric_type,
        status=experiment.status,
        n_control=n_c,
        n_treatment=n_t,
        srm=srm_out,
        frequentist=frequentist,
        bayesian=bayesian_out,
        recommendation=recommendation,
    )


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
