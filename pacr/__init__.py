"""Prevalence-adaptive conformal rule-out (PACR) for diabetic retinopathy triage."""

from .prevalence import (
    pacc_prevalence,
    pacc_upper_bound,
    em_prevalence,
    em_upper_bound,
    bbse_prevalence,
)
from .thresholds import (
    GRID,
    clopper_pearson_upper,
    clopper_pearson_lower,
    false_omission_rate,
    for_upper_bound,
    pacr_threshold,
    plugin_for_threshold,
    split_conformal_threshold,
    weighted_conformal_threshold,
    class_conditional_threshold,
    sensitivity_threshold,
)

__version__ = "1.0.0"
