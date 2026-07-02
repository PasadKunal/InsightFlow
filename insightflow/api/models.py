"""ORM models - the persistent shape of an experiment.

Three tables, one clean story:

* **Experiment** - the design: what we're testing, on which metric, at what split,
  and the pre-registered power parameters (so nobody moves the goalposts later).
* **Assignment** - which arm each user landed in. One row per (experiment, user);
  the unique constraint makes double-assignment impossible at the database level.
* **Observation** - a measured metric value for a user (0/1 for conversion metrics,
  a real number for continuous ones). This is the raw material every analysis reads.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _uuid() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class MetricType(str, enum.Enum):
    """What kind of metric this experiment measures - it decides which tests run."""

    PROPORTION = "proportion"    # conversion-style: each observation is 0 or 1
    CONTINUOUS = "continuous"    # revenue, session length, ... : any real number


class ExperimentStatus(str, enum.Enum):
    DRAFT = "draft"        # created, not yet collecting data
    RUNNING = "running"    # actively assigning + observing
    STOPPED = "stopped"    # halted early (e.g. SPRT fired, or a guardrail tripped)
    COMPLETED = "completed"


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    hypothesis: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    metric_type: Mapped[MetricType] = mapped_column(Enum(MetricType), nullable=False)
    status: Mapped[ExperimentStatus] = mapped_column(
        Enum(ExperimentStatus), nullable=False, default=ExperimentStatus.DRAFT
    )

    treatment_fraction: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    # Pre-registered design parameters (used for power analysis at creation time).
    baseline_rate: Mapped[float | None] = mapped_column(Float, nullable=True)
    minimum_detectable_effect: Mapped[float | None] = mapped_column(Float, nullable=True)
    alpha: Mapped[float] = mapped_column(Float, nullable=False, default=0.05)
    power: Mapped[float] = mapped_column(Float, nullable=False, default=0.80)

    # Computed once at creation from the design parameters above.
    required_sample_size_per_arm: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

    assignments: Mapped[list["Assignment"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )
    observations: Mapped[list["Observation"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class Assignment(Base):
    __tablename__ = "assignments"
    __table_args__ = (
        # A user can be assigned to a given experiment exactly once.
        UniqueConstraint("experiment_id", "user_id", name="uq_assignment_experiment_user"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    variant: Mapped[str] = mapped_column(String(20), nullable=False)  # "control" | "treatment"
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    experiment: Mapped["Experiment"] = relationship(back_populates="assignments")


class Observation(Base):
    __tablename__ = "observations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user_id: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    variant: Mapped[str] = mapped_column(String(20), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    experiment: Mapped["Experiment"] = relationship(back_populates="observations")
