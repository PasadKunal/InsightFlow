"""Plotly charts for experiment results.

Each function returns a Plotly ``Figure`` - not an image, not HTML. That's deliberate:
a Figure can be serialized to JSON and handed straight to the React dashboard (Phase
5b), rendered to a PNG for a PDF, or shown inline in a notebook. One source of truth,
many surfaces.

The visual language is intentionally restrained: a calm dark-on-light palette, no
chart-junk, clear reference lines at "no effect". Professional, not flashy.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import plotly.graph_objects as go

# A small, consistent palette shared across every chart.
CONTROL_COLOR = "#94a3b8"    # slate - neutral
TREATMENT_COLOR = "#0f3460"  # deep navy - the brand color
ACCENT = "#2ea44f"           # green - positive / ship
WARN = "#dc2626"             # red - regression / SRM


def _layout(fig: go.Figure, title: str, **kwargs) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#1a1a2e")),
        template="plotly_white",
        font=dict(family="Inter, Arial, sans-serif", color="#1a202c"),
        margin=dict(l=60, r=30, t=60, b=50),
        **kwargs,
    )
    return fig


def confidence_interval_plot(
    point: float, lower: float, upper: float, *, label: str = "Effect", title: str = "Treatment effect"
) -> go.Figure:
    """A single effect estimate with its confidence interval and a 'no effect' line.

    The whole story of a frequentist result in one glance: does the interval clear zero?
    """
    crosses_zero = lower <= 0 <= upper
    color = WARN if crosses_zero else ACCENT
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[point], y=[label],
            error_x=dict(type="data", symmetric=False, array=[upper - point], arrayminus=[point - lower]),
            mode="markers", marker=dict(size=12, color=color),
            name="estimate",
        )
    )
    fig.add_vline(x=0, line_dash="dash", line_color="#64748b", annotation_text="no effect")
    return _layout(fig, title, xaxis_title="effect size", showlegend=False, height=260)


def conversion_bar(
    rate_control: float, rate_treatment: float, *,
    ci_control: tuple[float, float] | None = None,
    ci_treatment: tuple[float, float] | None = None,
    title: str = "Conversion rate by arm",
) -> go.Figure:
    """Side-by-side conversion rates with optional error bars."""
    arms = ["control", "treatment"]
    rates = [rate_control, rate_treatment]
    errors = None
    if ci_control and ci_treatment:
        errors = dict(
            type="data", symmetric=False,
            array=[ci_control[1] - rate_control, ci_treatment[1] - rate_treatment],
            arrayminus=[rate_control - ci_control[0], rate_treatment - ci_treatment[0]],
        )
    fig = go.Figure(
        go.Bar(x=arms, y=rates, error_y=errors, marker_color=[CONTROL_COLOR, TREATMENT_COLOR])
    )
    fig.update_yaxes(tickformat=".1%")
    return _layout(fig, title, yaxis_title="conversion rate", showlegend=False, height=340)


def sprt_trace(
    log_lr: Sequence[float], upper: float, lower: float, *, title: str = "Sequential test (SPRT) trace"
) -> go.Figure:
    """The running log-likelihood ratio against its stopping boundaries.

    A picture of *why* the test stopped when it did - the line walks until it pokes
    through a boundary.
    """
    log_lr = list(log_lr)
    x = list(range(1, len(log_lr) + 1))
    fig = go.Figure()
    fig.add_hrect(y0=lower, y1=upper, fillcolor="#f1f5f9", line_width=0)
    fig.add_hline(y=upper, line_dash="dash", line_color=ACCENT, annotation_text="reject null")
    fig.add_hline(y=lower, line_dash="dash", line_color=CONTROL_COLOR, annotation_text="accept null")
    fig.add_trace(go.Scatter(x=x, y=log_lr, mode="lines", line=dict(color=TREATMENT_COLOR, width=2)))
    return _layout(fig, title, xaxis_title="observations", yaxis_title="log-likelihood ratio", showlegend=False, height=340)


def posterior_plot(
    alpha_c: float, beta_c: float, alpha_t: float, beta_t: float, *, title: str = "Posterior conversion rates"
) -> go.Figure:
    """Overlaid Beta posteriors for the two arms - the Bayesian view of separation."""
    from scipy import stats

    xs = np.linspace(0, 1, 500)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=stats.beta.pdf(xs, alpha_c, beta_c),
                             mode="lines", name="control", line=dict(color=CONTROL_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=xs, y=stats.beta.pdf(xs, alpha_t, beta_t),
                             mode="lines", name="treatment", line=dict(color=TREATMENT_COLOR, width=2), fill="tozeroy"))
    # Focus the x-axis on where the mass actually is.
    lo = max(0, min(stats.beta.ppf(0.001, alpha_c, beta_c), stats.beta.ppf(0.001, alpha_t, beta_t)))
    hi = min(1, max(stats.beta.ppf(0.999, alpha_c, beta_c), stats.beta.ppf(0.999, alpha_t, beta_t)))
    fig.update_xaxes(range=[lo, hi], tickformat=".1%")
    return _layout(fig, title, xaxis_title="conversion rate", yaxis_title="density", height=340)


def uplift_quantile_bar(quantiles: Sequence[int], mean_cate: Sequence[float], *,
                        title: str = "Treatment effect by responder quintile") -> go.Figure:
    """Bar chart of estimated effect per responder bucket - the targeting story."""
    fig = go.Figure(go.Bar(x=[f"Q{q}" for q in quantiles], y=list(mean_cate), marker_color=TREATMENT_COLOR))
    fig.add_hline(y=float(np.mean(mean_cate)), line_dash="dash", line_color=ACCENT, annotation_text="overall ATE")
    return _layout(fig, title, xaxis_title="responder quintile (low → high)", yaxis_title="mean CATE", showlegend=False, height=340)


def figure_to_json(fig: go.Figure) -> str:
    """Serialize a figure to JSON for the frontend (Plotly.react consumes this directly)."""
    return fig.to_json()
