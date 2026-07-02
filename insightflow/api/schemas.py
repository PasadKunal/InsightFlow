"""Pydantic schemas - the validated request/response contracts.

Keeping these separate from the ORM models is deliberate: the database shape and the
API shape are allowed to differ, and FastAPI turns these classes into the interactive
Swagger docs at ``/docs`` for free.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from .models import ExperimentStatus, MetricType


# ── Experiments ──────────────────────────────────────────────────────────────
class ExperimentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200, examples=["Checkout redesign"])
    hypothesis: str | None = Field(
        None, examples=["The new checkout increases purchase conversion."]
    )
    metric_type: MetricType = MetricType.PROPORTION
    treatment_fraction: float = Field(0.5, gt=0, lt=1)

    # Optional design parameters - if given for a proportion metric, we auto-compute
    # the required sample size via a power analysis.
    baseline_rate: float | None = Field(None, gt=0, lt=1, examples=[0.10])
    minimum_detectable_effect: float | None = Field(
        None, gt=0, description="Relative lift to detect, e.g. 0.10 = a 10% lift", examples=[0.10]
    )
    alpha: float = Field(0.05, gt=0, lt=1)
    power: float = Field(0.80, gt=0, lt=1)


class StatusUpdate(BaseModel):
    status: ExperimentStatus


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    hypothesis: str | None
    metric_type: MetricType
    status: ExperimentStatus
    treatment_fraction: float
    baseline_rate: float | None
    minimum_detectable_effect: float | None
    alpha: float
    power: float
    required_sample_size_per_arm: int | None
    created_at: datetime
    updated_at: datetime


# ── Assignment & observation ─────────────────────────────────────────────────
class AssignRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=200)


class AssignmentOut(BaseModel):
    experiment_id: str
    user_id: str
    variant: str


class ObserveRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=200)
    value: float = Field(..., description="0/1 for conversion metrics, any real number otherwise")


class BulkObserveRequest(BaseModel):
    observations: list[ObserveRequest] = Field(..., min_length=1)


class IngestSummary(BaseModel):
    ingested: int
    control: int
    treatment: int


class SimulateRequest(BaseModel):
    """Generate synthetic observations so an experiment is instantly demoable."""

    n_users: int = Field(4000, gt=0, le=200_000)
    # Proportion metrics:
    control_rate: float = Field(0.10, ge=0, le=1)
    treatment_rate: float = Field(0.12, ge=0, le=1)
    # Continuous metrics:
    control_mean: float = 50.0
    treatment_mean: float = 52.0
    std: float = Field(20.0, gt=0)


# ── Results ──────────────────────────────────────────────────────────────────
class SRMOut(BaseModel):
    mismatch_detected: bool
    p_value: float
    observed: dict[str, int]
    expected: dict[str, float]
    message: str


class FrequentistOut(BaseModel):
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float | None
    effect_size_name: str | None
    ci_lower: float | None
    ci_upper: float | None
    extra: dict


class BayesianOut(BaseModel):
    prob_treatment_best: float
    expected_relative_uplift: float
    expected_loss: float
    recommendation: str


class ResultsOut(BaseModel):
    experiment_id: str
    metric_type: MetricType
    status: ExperimentStatus
    n_control: int
    n_treatment: int
    srm: SRMOut
    frequentist: list[FrequentistOut]
    bayesian: BayesianOut | None
    recommendation: str
