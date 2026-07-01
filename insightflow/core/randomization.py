"""Assigning users to experiment arms — correctly.

Randomization sounds trivial ("flip a coin") but it is where a lot of real
experiments quietly break. Two things matter:

* **Determinism.** The *same* user must always land in the *same* arm, even across
  service restarts and re-runs. We get this by hashing ``user_id + experiment
  seed`` instead of calling a random number generator. This is exactly how
  production assignment services (feature-flag platforms) work.

* **Balance on the things that matter.** If "device type" strongly affects your
  metric, a lucky-unlucky split can put more iPhone users in treatment and fool
  you. **Stratified randomization** fixes this by randomizing *within* each
  subgroup, so every stratum is split evenly and variance drops.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Hashable, Iterable, Optional, Sequence


# Bucket resolution for deterministic hashing. 10,000 buckets lets us express
# split ratios down to 0.01% precision, which is far finer than any real split.
_BUCKETS = 10_000


def _hash_to_unit(user_id: str, salt: str) -> float:
    """Map (user_id, salt) deterministically to a float in [0, 1).

    We use a stable cryptographic hash (not Python's built-in ``hash``, which is
    randomized per-process) so assignments are reproducible everywhere, forever.
    """
    digest = hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()
    bucket = int(digest, 16) % _BUCKETS
    return bucket / _BUCKETS


def assign(
    user_id: Hashable,
    *,
    experiment_id: str,
    treatment_fraction: float = 0.5,
) -> str:
    """Deterministically assign one user to ``"treatment"`` or ``"control"``.

    Parameters
    ----------
    user_id:
        Stable identifier for the user (anything ``str()``-able).
    experiment_id:
        Salts the hash so the *same* user gets an *independent* assignment in each
        experiment — no accidental correlation between concurrent tests.
    treatment_fraction:
        Share of users sent to treatment, e.g. 0.5 for a 50/50 split.
    """
    if not 0 < treatment_fraction < 1:
        raise ValueError("treatment_fraction must be strictly between 0 and 1.")
    u = _hash_to_unit(str(user_id), salt=experiment_id)
    return "treatment" if u < treatment_fraction else "control"


@dataclass(frozen=True)
class AssignmentSummary:
    """Counts of who landed where, for a quick balance sanity-check."""

    control: int
    treatment: int

    @property
    def total(self) -> int:
        return self.control + self.treatment

    @property
    def treatment_fraction(self) -> float:
        return self.treatment / self.total if self.total else 0.0


def assign_many(
    user_ids: Iterable[Hashable],
    *,
    experiment_id: str,
    treatment_fraction: float = 0.5,
) -> dict[str, str]:
    """Assign a whole population at once. Returns ``{user_id: arm}``."""
    return {
        str(uid): assign(
            uid, experiment_id=experiment_id, treatment_fraction=treatment_fraction
        )
        for uid in user_ids
    }


def stratified_assign(
    user_ids: Sequence[Hashable],
    strata: Sequence[Hashable],
    *,
    experiment_id: str,
    treatment_fraction: float = 0.5,
) -> dict[str, str]:
    """Assign users while keeping each stratum internally balanced.

    ``strata[i]`` is the subgroup label for ``user_ids[i]`` (e.g. "ios", "android",
    "web"). Within every stratum we still hash-assign each user deterministically,
    but we *nudge* the per-stratum split toward the target fraction so no subgroup
    ends up lopsided. The result is lower variance in your treatment-effect
    estimate — you have controlled for the covariate by construction.

    Parameters
    ----------
    user_ids, strata:
        Parallel sequences of equal length.
    """
    if len(user_ids) != len(strata):
        raise ValueError("user_ids and strata must be the same length.")
    if not 0 < treatment_fraction < 1:
        raise ValueError("treatment_fraction must be strictly between 0 and 1.")

    # Group user indices by stratum label.
    groups: dict[Hashable, list[int]] = {}
    for i, s in enumerate(strata):
        groups.setdefault(s, []).append(i)

    assignments: dict[str, str] = {}
    for stratum, indices in groups.items():
        # Deterministic within-stratum ordering by hash: reproducible, and
        # independent of the order users happened to arrive in.
        ordered = sorted(
            indices,
            key=lambda i: _hash_to_unit(str(user_ids[i]), salt=f"{experiment_id}:{stratum}"),
        )
        # The first `treatment_fraction` share of the ordered stratum -> treatment.
        # Rounding is done per-stratum so totals stay close to the target overall.
        n_treatment = round(len(ordered) * treatment_fraction)
        for rank, idx in enumerate(ordered):
            assignments[str(user_ids[idx])] = "treatment" if rank < n_treatment else "control"

    return assignments


def summarize(assignments: dict[str, str]) -> AssignmentSummary:
    """Tally an assignment dict into control/treatment counts."""
    counts = Counter(assignments.values())
    return AssignmentSummary(
        control=counts.get("control", 0),
        treatment=counts.get("treatment", 0),
    )
