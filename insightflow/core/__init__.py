"""InsightFlow statistical core.

The framework-independent heart of the platform: hypothesis tests, power analysis,
randomization, and data-quality guards. No web server, no database — just numbers
in, trustworthy answers out. Everything here is unit-tested in isolation.
"""

from .frequentist import (
    chi_squared_test,
    mann_whitney_u,
    proportion_ztest,
    two_sample_ttest,
)
from .power_analysis import (
    SampleSizeResult,
    power_for_proportion,
    sample_size_for_mean,
    sample_size_for_proportion,
)
from .randomization import (
    AssignmentSummary,
    assign,
    assign_many,
    stratified_assign,
    summarize,
)
from .results import ConfidenceInterval, TestResult
from .srm_detector import SRMResult, detect_srm

__all__ = [
    # results
    "TestResult",
    "ConfidenceInterval",
    # frequentist tests
    "two_sample_ttest",
    "proportion_ztest",
    "chi_squared_test",
    "mann_whitney_u",
    # power / design
    "sample_size_for_proportion",
    "sample_size_for_mean",
    "power_for_proportion",
    "SampleSizeResult",
    # randomization
    "assign",
    "assign_many",
    "stratified_assign",
    "summarize",
    "AssignmentSummary",
    # data quality
    "detect_srm",
    "SRMResult",
]
