"""Turning per-user CATE estimates into targeting decisions.

A vector of 100,000 individual treatment effects isn't actionable on its own. What a
growth team actually wants to hear is: *"target these people."* This module answers
that two ways:

* **By quantile** - sort users by estimated effect and bucket them (quintiles by
  default). The top bucket is your high-responder segment, and we report how many
  times the overall ATE its effect is worth - the "target the top 20%" story.
* **By known segment** - if you already have a categorical dimension (country, plan,
  acquisition channel), average the estimated effect within each and rank them, so
  you can say "enterprise users respond 3x more than free users."
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuantileReport:
    """Estimated effect broken down by CATE quantile (e.g. quintiles)."""

    table: pd.DataFrame          # columns: quantile, n_users, mean_cate, lift_vs_ate
    overall_ate: float
    top_quantile_lift: float     # top bucket's mean effect / overall ATE

    def summary(self) -> str:
        return (
            f"Overall ATE = {self.overall_ate:.4g}. "
            f"Top quantile responds {self.top_quantile_lift:.2f}x the average.\n"
            f"{self.table.to_string(index=False)}"
        )

    def __str__(self) -> str:
        return self.summary()


def rank_by_quantile(cate: np.ndarray, *, n_quantiles: int = 5) -> QuantileReport:
    """Bucket users by estimated effect and measure each bucket's uplift.

    Bucket 1 is the lowest-responding group, bucket ``n_quantiles`` the highest.
    """
    cate = np.asarray(cate, dtype=float)
    if cate.size < n_quantiles:
        raise ValueError("Need at least as many users as quantiles.")

    overall_ate = float(np.mean(cate))
    # Rank -> quantile bucket 1..n_quantiles. Using ranks avoids ties collapsing bins.
    ranks = cate.argsort().argsort()
    buckets = np.floor(ranks / len(cate) * n_quantiles).astype(int) + 1
    buckets = np.clip(buckets, 1, n_quantiles)

    rows = []
    for q in range(1, n_quantiles + 1):
        mask = buckets == q
        mean_cate = float(np.mean(cate[mask]))
        rows.append(
            {
                "quantile": q,
                "n_users": int(mask.sum()),
                "mean_cate": mean_cate,
                "lift_vs_ate": mean_cate / overall_ate if overall_ate != 0 else float("nan"),
            }
        )
    table = pd.DataFrame(rows)
    top_lift = float(table.loc[table["quantile"] == n_quantiles, "lift_vs_ate"].iloc[0])

    return QuantileReport(table=table, overall_ate=overall_ate, top_quantile_lift=top_lift)


@dataclass(frozen=True)
class SegmentReport:
    """Estimated effect ranked across a known categorical dimension."""

    table: pd.DataFrame          # columns: segment, n_users, mean_cate, lift_vs_ate
    overall_ate: float

    @property
    def best_segment(self) -> str:
        return str(self.table.iloc[0]["segment"])

    def summary(self) -> str:
        return (
            f"Overall ATE = {self.overall_ate:.4g}. "
            f"Best-responding segment: {self.best_segment!r}.\n"
            f"{self.table.to_string(index=False)}"
        )

    def __str__(self) -> str:
        return self.summary()


def rank_segments(cate: np.ndarray, segments) -> SegmentReport:
    """Average the estimated effect within each segment label and rank, best first."""
    cate = np.asarray(cate, dtype=float)
    segments = np.asarray(segments)
    if cate.shape[0] != segments.shape[0]:
        raise ValueError("cate and segments must be the same length.")

    overall_ate = float(np.mean(cate))
    rows = []
    for label in pd.unique(segments):
        mask = segments == label
        mean_cate = float(np.mean(cate[mask]))
        rows.append(
            {
                "segment": label,
                "n_users": int(mask.sum()),
                "mean_cate": mean_cate,
                "lift_vs_ate": mean_cate / overall_ate if overall_ate != 0 else float("nan"),
            }
        )
    table = (
        pd.DataFrame(rows)
        .sort_values("mean_cate", ascending=False)
        .reset_index(drop=True)
    )
    return SegmentReport(table=table, overall_ate=overall_ate)
