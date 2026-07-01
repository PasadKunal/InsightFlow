"""Scheduled report delivery — the "weekly digest" without heavy infrastructure.

Teams want experiment reports to *arrive*, not to be fetched. A weekly digest that
lands every Monday morning is the difference between a dashboard people forget and one
that drives decisions. We use APScheduler's in-process ``BackgroundScheduler`` rather
than Celery + a broker: at this scale a cron-like thread is all you need, and it keeps
the deployment a single process.

This is a thin, well-behaved wrapper — you hand it a callable and a cadence; it runs
the callable on that cadence. What the callable *does* (build a report, render a PDF,
email it) is left to the caller, so the scheduler stays testable and dependency-free.
"""

from __future__ import annotations

from typing import Callable

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger


class ReportScheduler:
    """A small facade over APScheduler for recurring report jobs."""

    def __init__(self):
        self._scheduler = BackgroundScheduler(daemon=True)

    def schedule_weekly(
        self,
        func: Callable[[], None],
        *,
        job_id: str,
        day_of_week: str = "mon",
        hour: int = 8,
        minute: int = 0,
    ) -> str:
        """Run ``func`` every week (default: Mondays at 08:00). Returns the job id."""
        self._scheduler.add_job(
            func,
            trigger=CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute),
            id=job_id,
            replace_existing=True,
        )
        return job_id

    def schedule_interval(self, func: Callable[[], None], *, job_id: str, seconds: int) -> str:
        """Run ``func`` every ``seconds`` — handy for demos and tests."""
        self._scheduler.add_job(func, trigger="interval", seconds=seconds, id=job_id, replace_existing=True)
        return job_id

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self, wait: bool = False) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=wait)

    @property
    def jobs(self) -> list:
        return self._scheduler.get_jobs()
