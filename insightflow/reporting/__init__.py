"""InsightFlow reporting.

Turns raw statistics into decisions people can act on: a structured experiment report,
a natural-language summary (free, pluggable LLM backends), Plotly charts, a shareable
PDF, and a scheduler for recurring digests.
"""

from .insight_generator import (
    GroqProvider,
    LLMProvider,
    OllamaProvider,
    TemplateProvider,
    generate_insight,
    get_provider,
)
from .pdf_exporter import report_to_pdf, report_to_pdf_bytes
from .report_generator import (
    DO_NOT_SHIP,
    EXTEND,
    INVALID,
    SHIP,
    ExperimentReport,
    generate_report,
)
from .scheduler import ReportScheduler

__all__ = [
    # report
    "generate_report",
    "ExperimentReport",
    "SHIP",
    "DO_NOT_SHIP",
    "EXTEND",
    "INVALID",
    # insight / LLM
    "generate_insight",
    "get_provider",
    "LLMProvider",
    "TemplateProvider",
    "GroqProvider",
    "OllamaProvider",
    # export & schedule
    "report_to_pdf",
    "report_to_pdf_bytes",
    "ReportScheduler",
]
