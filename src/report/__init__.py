"""
Report Module - Minimal Generator

Provides a minimal report generator that aggregates run results and can
optionally persist JSON output.
"""

from .generator import generate_report, save_report_json

__all__ = ["generate_report", "save_report_json"]
