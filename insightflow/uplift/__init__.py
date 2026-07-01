"""InsightFlow uplift modeling.

Heterogeneous treatment-effect estimation: figure out *who* a treatment helps, not
just whether it helps on average. Built on the X-Learner meta-learner with SHAP-based
explanations and segment ranking for targeted rollouts.
"""

from .segment_analyzer import (
    QuantileReport,
    SegmentReport,
    rank_by_quantile,
    rank_segments,
)
from .shap_analysis import CateExplainer, ShapExplanation, explain_cate
from .synthetic_validator import (
    SyntheticData,
    UpliftValidationReport,
    make_synthetic_uplift_data,
    validate_xlearner,
)
from .x_learner import XLearner

__all__ = [
    "XLearner",
    # explanations
    "CateExplainer",
    "explain_cate",
    "ShapExplanation",
    # segments
    "rank_by_quantile",
    "rank_segments",
    "QuantileReport",
    "SegmentReport",
    # validation
    "make_synthetic_uplift_data",
    "validate_xlearner",
    "SyntheticData",
    "UpliftValidationReport",
]
