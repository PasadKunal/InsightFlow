"""InsightFlow statistical core.

The framework-independent heart of the platform: hypothesis tests, power analysis,
randomization, and data-quality guards. No web server, no database - just numbers
in, trustworthy answers out. Everything here is unit-tested in isolation.
"""

from .bayesian import BayesianResult, PosteriorArm, beta_binomial_test
from .frequentist import (
    chi_squared_test,
    mann_whitney_u,
    proportion_ztest,
    two_sample_ttest,
)
from .multiple_testing import (
    CorrectionResult,
    benjamini_hochberg,
    bonferroni,
    correct,
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
from .results import ConfidenceInterval, CredibleInterval, TestResult
from .sequential import SequentialTest, SPRTResult, run_sprt
from .srm_detector import SRMResult, detect_srm

__all__ = [
    # results
    "TestResult",
    "ConfidenceInterval",
    "CredibleInterval",
    # frequentist tests
    "two_sample_ttest",
    "proportion_ztest",
    "chi_squared_test",
    "mann_whitney_u",
    # bayesian
    "beta_binomial_test",
    "BayesianResult",
    "PosteriorArm",
    # sequential (SPRT)
    "SequentialTest",
    "run_sprt",
    "SPRTResult",
    # multiple testing
    "bonferroni",
    "benjamini_hochberg",
    "correct",
    "CorrectionResult",
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
