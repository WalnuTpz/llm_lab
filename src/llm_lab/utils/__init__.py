"""General utilities."""

from llm_lab.utils.device import dtype_from_str, resolve_device, safe_dtype_for_device
from llm_lab.utils.params import ParameterReport, count_parameters, parameter_report

__all__ = [
    "ParameterReport",
    "count_parameters",
    "dtype_from_str",
    "parameter_report",
    "resolve_device",
    "safe_dtype_for_device",
]
