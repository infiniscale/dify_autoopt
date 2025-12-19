"""
Report Generator - Minimal Implementation

Aggregates workflow run results into a simple report dict and provides
an optional JSON persistence helper.
"""

import json
from datetime import datetime
from typing import Any, Dict, List


def generate_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate a list of run result dicts into a summary report.

    Expected item keys (best-effort): workflow_id, label, status, metrics{duration_seconds}
    """
    total = len(results)
    success = sum(1 for r in results if r.get("status") == "success")
    durations = [r.get("metrics", {}).get("duration_seconds", 0.0) for r in results]
    avg_duration = (sum(durations) / len(durations)) if durations else 0.0

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_runs": total,
        "success": success,
        "success_rate": (success / total) if total else 0.0,
        "average_duration_seconds": avg_duration,
        "items": results,
    }


def save_report_json(report: Dict[str, Any], path: str) -> None:
    """Persist report dict into a JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
