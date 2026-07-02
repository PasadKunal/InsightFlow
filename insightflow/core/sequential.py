"""Sequential testing with the SPRT - valid early stopping.

Here is the trap that ruins real experiments: you run an A/B test, glance at it on
day 3, see ``p = 0.04``, and stop. That p-value is a **lie**. A fixed-sample t-test
is only valid if you look *once*, at the end. Every peek is another roll of the dice,
and repeated peeking inflates your false-positive rate far above the 5% you think
you're controlling.

Wald's **Sequential Probability Ratio Test (SPRT)** is the principled fix. Instead of
a fixed sample size, you accumulate evidence one observation at a time and stop the
*moment* the evidence is decisive - with the error rates you asked for still intact.

How it works, in one breath: pick two hypotheses about the conversion rate, a null
``p0`` and an alternative ``p1``. After each user, update the **log-likelihood ratio**
(how much more likely the data is under ``p1`` than ``p0``). Two boundaries, derived
from your target error rates, bracket the "keep going" zone:

    upper = log((1 - beta) / alpha)      cross it  -> reject H0 (treatment wins)
    lower = log(beta / (1 - alpha))      cross it  -> accept H0 (no effect)

Stay between them and you collect one more observation. Because the boundaries are
built from ``alpha`` and ``beta`` directly, stopping at *any* crossing keeps those
error rates valid - that's the whole point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

Decision = Literal["reject_null", "accept_null", "continue"]


@dataclass(frozen=True)
class SPRTResult:
    """A snapshot of a sequential test after some number of observations."""

    decision: Decision
    log_likelihood_ratio: float
    upper_boundary: float
    lower_boundary: float
    n_observations: int
    stopped_at: int | None  # observation index where a boundary was crossed

    @property
    def stopped(self) -> bool:
        return self.decision != "continue"

    def summary(self) -> str:
        if self.decision == "reject_null":
            verdict = f"REJECT null - treatment effect detected (stopped at n={self.stopped_at})"
        elif self.decision == "accept_null":
            verdict = f"ACCEPT null - no effect (stopped at n={self.stopped_at})"
        else:
            verdict = f"INCONCLUSIVE - still between boundaries after n={self.n_observations}"
        return (
            f"SPRT: {verdict} | log-LR={self.log_likelihood_ratio:.3f} "
            f"in [{self.lower_boundary:.3f}, {self.upper_boundary:.3f}]"
        )

    def __str__(self) -> str:
        return self.summary()


class SequentialTest:
    """An SPRT for Bernoulli (converted / not-converted) observations.

    Use it in a streaming loop::

        test = SequentialTest(p0=0.10, p1=0.12, alpha=0.05, beta=0.20)
        for converted in event_stream:
            if test.observe(converted).stopped:
                break

    ``p0`` is the null conversion rate (usually your current baseline) and ``p1`` is
    the smallest lift you'd care to detect. ``alpha`` bounds false positives; ``beta``
    bounds false negatives (so power is ``1 - beta``).
    """

    def __init__(self, p0: float, p1: float, *, alpha: float = 0.05, beta: float = 0.20):
        if not 0 < p0 < 1 or not 0 < p1 < 1:
            raise ValueError("p0 and p1 must both be strictly between 0 and 1.")
        if p0 == p1:
            raise ValueError("p0 and p1 must differ - there is nothing to distinguish.")
        if not 0 < alpha < 1 or not 0 < beta < 1:
            raise ValueError("alpha and beta must be strictly between 0 and 1.")

        self.p0 = p0
        self.p1 = p1
        self.alpha = alpha
        self.beta = beta

        # Wald's boundaries on the log-likelihood-ratio scale.
        self.upper_boundary = math.log((1 - beta) / alpha)
        self.lower_boundary = math.log(beta / (1 - alpha))

        # Per-observation log-LR increments, precomputed for speed and clarity.
        self._llr_success = math.log(p1 / p0)
        self._llr_failure = math.log((1 - p1) / (1 - p0))

        self.log_likelihood_ratio = 0.0
        self.n_observations = 0
        self._decision: Decision = "continue"
        self._stopped_at: int | None = None

    def observe(self, converted: bool) -> SPRTResult:
        """Feed in one observation and get the current decision back.

        Once a boundary is crossed the test latches: further observations are
        ignored and the terminal decision is returned unchanged.
        """
        if self._decision == "continue":
            self.n_observations += 1
            self.log_likelihood_ratio += self._llr_success if converted else self._llr_failure

            if self.log_likelihood_ratio >= self.upper_boundary:
                self._decision = "reject_null"
                self._stopped_at = self.n_observations
            elif self.log_likelihood_ratio <= self.lower_boundary:
                self._decision = "accept_null"
                self._stopped_at = self.n_observations

        return self._snapshot()

    def _snapshot(self) -> SPRTResult:
        return SPRTResult(
            decision=self._decision,
            log_likelihood_ratio=self.log_likelihood_ratio,
            upper_boundary=self.upper_boundary,
            lower_boundary=self.lower_boundary,
            n_observations=self.n_observations,
            stopped_at=self._stopped_at,
        )


def run_sprt(
    observations: Iterable[bool],
    *,
    p0: float,
    p1: float,
    alpha: float = 0.05,
    beta: float = 0.20,
) -> SPRTResult:
    """Run an SPRT over a whole sequence and return where it stopped.

    A convenience wrapper around :class:`SequentialTest` for when you already have
    the data in hand (e.g. replaying a log or validating on simulated streams).
    Stops early the instant a boundary is crossed - the returned ``stopped_at``
    tells you how few observations that took.
    """
    test = SequentialTest(p0, p1, alpha=alpha, beta=beta)
    result = test._snapshot()
    for converted in observations:
        result = test.observe(converted)
        if result.stopped:
            break
    return result
