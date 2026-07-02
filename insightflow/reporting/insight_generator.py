"""Natural-language experiment summaries - with a strictly free, pluggable backend.

The goal: after an experiment finishes, produce a short paragraph a non-technical PM
can read and act on. The design constraint here is deliberate - **no paid APIs**. So
this module ships three interchangeable providers, chosen by environment variables:

* **template** (default) - no LLM at all. A deterministic, rule-based paragraph built
  straight from the report's numbers. Always available, needs no key, never fails.
* **groq** - Groq's free API (blazing-fast Llama 3.3 70B). Set ``INSIGHTFLOW_LLM=groq``
  and ``GROQ_API_KEY``. Uses the OpenAI-compatible endpoint over plain HTTP.
* **ollama** - a fully local model (e.g. Llama 3.2) via the Ollama daemon. Set
  ``INSIGHTFLOW_LLM=ollama``. 100% offline, zero cost, no signup.

Whatever you pick, if the call errors (no key, network down, daemon off) we fall back
to the template - a report always gets *a* summary. The narrative never blocks a
decision.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import httpx

from .report_generator import ExperimentReport

_SYSTEM_PROMPT = (
    "You are a senior data scientist writing a concise, plain-English summary of an "
    "A/B test for a non-technical product manager. Be direct and specific. State the "
    "recommendation, the key numbers that justify it, and any caveat. 3-5 sentences, "
    "no markdown, no headers."
)


def _build_prompt(report: ExperimentReport) -> str:
    """The user-side prompt: the raw facts, handed over as clean text."""
    return (
        "Write the summary for this experiment.\n\n"
        f"{report.text_summary()}\n\n"
        "Summary:"
    )


class LLMProvider(ABC):
    """Interface every provider implements: text prompt in, text summary out."""

    name: str

    @abstractmethod
    def generate(self, prompt: str) -> str: ...


class TemplateProvider(LLMProvider):
    """Rule-based narrative - no model, no key, always works."""

    name = "template"

    def generate(self, prompt: str) -> str:  # prompt is ignored by design
        raise NotImplementedError("TemplateProvider renders from the report, not a prompt.")

    def render(self, report: ExperimentReport) -> str:
        parts: list[str] = []
        verb = {
            "SHIP": "We recommend shipping the treatment.",
            "DO NOT SHIP": "We recommend not shipping the treatment.",
            "EXTEND": "We recommend extending the experiment before deciding.",
            "INVALID": "This experiment is invalid and must be rerun.",
        }.get(report.recommendation, "")
        parts.append(verb)

        if report.significant:
            direction = "an improvement" if (report.effect_size or 0) > 0 else "a regression"
            parts.append(
                f"The {report.primary_test_name.lower()} found a statistically significant "
                f"result (p = {report.p_value:.3g}), indicating {direction}."
            )
        else:
            parts.append(
                f"The difference was not statistically significant (p = {report.p_value:.3g})."
            )

        if report.bayesian:
            parts.append(
                f"There is a {report.bayesian['prob_treatment_best']:.0%} probability the "
                f"treatment is better, with an expected relative uplift of "
                f"{report.bayesian['expected_relative_uplift']:+.1%}."
            )
        if report.srm and report.srm["mismatch_detected"]:
            parts.append(
                "Warning: a Sample Ratio Mismatch was detected, so these numbers cannot "
                "be trusted until the assignment pipeline is fixed."
            )
        if report.top_quintile_lift and report.top_quintile_lift > 1.1:
            parts.append(
                f"Effects are heterogeneous: the top-responding user quintile shows "
                f"{report.top_quintile_lift:.1f}x the average effect, a candidate for targeted rollout."
            )
        return " ".join(p for p in parts if p)


class GroqProvider(LLMProvider):
    """Groq free API via the OpenAI-compatible chat endpoint."""

    name = "groq"

    def __init__(self, model: str | None = None, timeout: float = 30.0):
        self.api_key = os.environ.get("GROQ_API_KEY")
        self.model = model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set.")
        resp = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.3,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


class OllamaProvider(LLMProvider):
    """Local model through the Ollama daemon (default http://localhost:11434)."""

    name = "ollama"

    def __init__(self, model: str | None = None, host: str | None = None, timeout: float = 60.0):
        self.model = model or os.environ.get("OLLAMA_MODEL", "llama3.2")
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        resp = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


def get_provider(name: str | None = None) -> LLMProvider:
    """Select a provider from ``name`` or the ``INSIGHTFLOW_LLM`` env var (default: template)."""
    choice = (name or os.environ.get("INSIGHTFLOW_LLM", "template")).lower()
    if choice == "groq":
        return GroqProvider()
    if choice == "ollama":
        return OllamaProvider()
    return TemplateProvider()


def generate_insight(report: ExperimentReport, *, provider: LLMProvider | None = None) -> str:
    """Produce a natural-language summary, always falling back to the template on error.

    Returns the text and (as a side effect) attaches it to ``report.narrative``.
    """
    provider = provider or get_provider()
    template = TemplateProvider()
    try:
        if isinstance(provider, TemplateProvider):
            text = template.render(report)
        else:
            text = provider.generate(_build_prompt(report))
            if not text:  # empty response -> use the reliable fallback
                text = template.render(report)
    except Exception:
        # Any failure (missing key, network, daemon down) -> deterministic template.
        text = template.render(report)

    report.narrative = text
    return text
