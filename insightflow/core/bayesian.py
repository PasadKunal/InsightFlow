"""Bayesian A/B testing with the Beta-Binomial model.

Frequentist tests answer "if there were no effect, how surprising is this data?"
Bayesian tests answer the question people *actually* ask: **"what is the probability
that treatment is better?"** - a statement you can hand to a non-technical PM without
a p-value footnote.

For conversion-rate experiments the math is exceptionally clean. If we start with a
``Beta(alpha, beta)`` prior belief about a rate and then observe ``k`` conversions in
``n`` users, the posterior is simply ``Beta(alpha + k, beta + n - k)``. (Beta is the
*conjugate* prior for the binomial - the update is pure arithmetic, no sampling
needed for the posterior itself.)

From those two posteriors we compute, by Monte-Carlo sampling:

* **P(treatment is best)** - the headline number.
* **Expected uplift** - how much better, on average.
* **Expected loss** - the risk you take by shipping treatment if you're wrong. This
  is the quantity mature teams actually threshold on ("ship when expected loss <
  0.1% of the metric").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from .results import CredibleInterval


@dataclass(frozen=True)
class PosteriorArm:
    """The posterior belief about one arm's conversion rate."""

    name: str
    conversions: int
    n: int
    alpha: float  # posterior Beta alpha
    beta: float   # posterior Beta beta

    @property
    def mean(self) -> float:
        """Posterior mean rate = alpha / (alpha + beta)."""
        return self.alpha / (self.alpha + self.beta)

    def credible_interval(self, credibility: float = 0.95) -> CredibleInterval:
        tail = (1 - credibility) / 2
        return CredibleInterval(
            lower=float(stats.beta.ppf(tail, self.alpha, self.beta)),
            upper=float(stats.beta.ppf(1 - tail, self.alpha, self.beta)),
            credibility=credibility,
        )


@dataclass(frozen=True)
class BayesianResult:
    """The full Bayesian read on a two-arm conversion experiment."""

    control: PosteriorArm
    treatment: PosteriorArm
    prob_treatment_best: float       # P(rate_treatment > rate_control)
    expected_absolute_uplift: float  # E[rate_treatment - rate_control]
    expected_relative_uplift: float  # E[(rate_t - rate_c) / rate_c]
    expected_loss: float             # risk (in rate units) of shipping treatment
    credibility: float

    @property
    def recommendation(self) -> str:
        """A simple ship/keep-running heuristic from the posterior."""
        if self.prob_treatment_best >= 0.95:
            return "SHIP treatment"
        if self.prob_treatment_best <= 0.05:
            return "KEEP control"
        return "KEEP RUNNING (not yet conclusive)"

    def summary(self) -> str:
        ci_t = self.treatment.credible_interval(self.credibility)
        return (
            f"Bayesian Beta-Binomial: P(treatment best) = {self.prob_treatment_best:.1%} | "
            f"expected uplift = {self.expected_relative_uplift:+.1%} relative | "
            f"expected loss = {self.expected_loss:.4g} | "
            f"treatment {ci_t} | -> {self.recommendation}"
        )

    def __str__(self) -> str:
        return self.summary()


def beta_binomial_test(
    conversions_control: int,
    n_control: int,
    conversions_treatment: int,
    n_treatment: int,
    *,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    credibility: float = 0.95,
    samples: int = 100_000,
    seed: int | None = 0,
) -> BayesianResult:
    """Run a Bayesian analysis of a two-arm conversion experiment.

    Parameters
    ----------
    conversions_* , n_*:
        Successes and totals for each arm.
    prior_alpha, prior_beta:
        The Beta prior, shared by both arms. The default ``Beta(1, 1)`` is the
        uniform "I know nothing" prior - every rate in [0, 1] equally likely.
    credibility:
        Width of the reported credible intervals (0.95 = 95%).
    samples:
        Monte-Carlo draws used to compare the two posteriors. More = smoother
        probabilities; 100k gives ~0.1% resolution.
    seed:
        Fixes the sampler so results are reproducible (important for tests and for
        reports that shouldn't wobble between refreshes). Pass ``None`` for entropy.
    """
    if prior_alpha <= 0 or prior_beta <= 0:
        raise ValueError("Prior alpha and beta must be positive.")
    for k, n, label in [
        (conversions_control, n_control, "control"),
        (conversions_treatment, n_treatment, "treatment"),
    ]:
        if n <= 0:
            raise ValueError(f"{label} sample size must be positive.")
        if not 0 <= k <= n:
            raise ValueError(f"{label} conversions must be between 0 and n.")

    control = PosteriorArm(
        name="control",
        conversions=conversions_control,
        n=n_control,
        alpha=prior_alpha + conversions_control,
        beta=prior_beta + n_control - conversions_control,
    )
    treatment = PosteriorArm(
        name="treatment",
        conversions=conversions_treatment,
        n=n_treatment,
        alpha=prior_alpha + conversions_treatment,
        beta=prior_beta + n_treatment - conversions_treatment,
    )

    # Draw from each posterior and compare draw-by-draw. This Monte-Carlo approach
    # generalizes cleanly to >2 arms and to any summary you want to read off.
    rng = np.random.default_rng(seed)
    draws_c = rng.beta(control.alpha, control.beta, size=samples)
    draws_t = rng.beta(treatment.alpha, treatment.beta, size=samples)

    diff = draws_t - draws_c
    prob_treatment_best = float(np.mean(draws_t > draws_c))
    expected_absolute_uplift = float(np.mean(diff))
    expected_relative_uplift = float(np.mean(diff / draws_c))

    # Expected loss of shipping treatment: how much rate we'd give up, on average,
    # in the worlds where control is actually the better arm.
    expected_loss = float(np.mean(np.maximum(draws_c - draws_t, 0.0)))

    return BayesianResult(
        control=control,
        treatment=treatment,
        prob_treatment_best=prob_treatment_best,
        expected_absolute_uplift=expected_absolute_uplift,
        expected_relative_uplift=expected_relative_uplift,
        expected_loss=expected_loss,
        credibility=credibility,
    )
