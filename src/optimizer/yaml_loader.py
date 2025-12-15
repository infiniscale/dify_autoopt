"""
Workflow YAML loader for optimizer routines.

Given a workflow id, load the exported workflow YAML from
`io_paths.output_dir/{id}/app_{id}.yaml` (or compatible filenames) using
the configured I/O paths from the runtime config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

import yaml

from src.config.bootstrap import get_runtime
from src.utils.logger import get_logger

logger = get_logger("optimizer.yaml_loader")
DEFAULT_OUTPUT_ROOT = Path("./outputs")


def _resolve_output_root(passed: Optional[str | Path]) -> Path:
    """Resolve the root output directory from runtime config or explicit value."""
    if passed:
        return Path(passed).expanduser()
    try:
        rt = get_runtime()
        io_paths = rt.app.io_paths or {}
        configured = io_paths.get("output_dir") or io_paths.get("output")
        if configured:
            return Path(configured).expanduser()
    except Exception:
        # Runtime may not be initialized; fall back to default
        pass
    return DEFAULT_OUTPUT_ROOT.expanduser()


def _dedupe(seq: Iterable[Path]) -> List[Path]:
    seen = set()
    ordered: List[Path] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _safe_dirname_from_id(app_id: str) -> str:
    """Filesystem-safe directory name."""
    try:
        import re

        name = str(app_id)
        name = name.replace("/", "_").replace("\\", "_")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        name = name.strip("._-") or "app"
        return name
    except Exception:
        return "app"


def _candidate_paths(output_root: Path, workflow_id: str) -> List[Path]:
    """Build candidate file paths following workflow export conventions."""
    safe_id = _safe_dirname_from_id(workflow_id)
    dirs: Sequence[Path] = (
        [output_root / safe_id, output_root / workflow_id]
        if safe_id != workflow_id
        else [output_root / safe_id]
    )

    filenames = [
        f"app_{safe_id}.yaml",
        f"app_{safe_id}.yml",
        f"app_{safe_id}_export.yaml",
        f"app_{safe_id}_export.yml",
        "app_export.yaml",
        "app_export.yml",
    ]
    if safe_id != workflow_id:
        filenames.extend(
            [
                f"app_{workflow_id}.yaml",
                f"app_{workflow_id}.yml",
                f"app_{workflow_id}_export.yaml",
                f"app_{workflow_id}_export.yml",
            ]
        )

    candidates = (directory / name for directory in dirs for name in filenames)
    return _dedupe(candidates)


def load_workflow_yaml(workflow_id: str, *, output_dir: Optional[str | Path] = None) -> tuple[Any, Path]:
    """
    Load an exported workflow YAML file for the given workflow id.

    Returns:
        A tuple of (parsed_yaml, resolved_path).

    Raises:
        FileNotFoundError: If no matching YAML file exists.
        yaml.YAMLError: If the YAML content cannot be parsed.
    """
    if not workflow_id:
        raise ValueError("workflow_id is required")

    root = _resolve_output_root(output_dir)
    candidates = _candidate_paths(root, workflow_id)

    for path in candidates:
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            try:
                logger.info(
                    "Loaded workflow YAML",
                    extra={"workflow_id": workflow_id, "path": str(path)},
                )
            except Exception:
                pass
            return data, path

    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Workflow YAML for '{workflow_id}' not found. Tried: {tried}")


class WorkflowYamlLoader:
    """Helper class wrapper for loading workflow YAML exports."""

    def __init__(self, output_dir: Optional[str | Path] = None) -> None:
        self.output_dir = output_dir

    def load(self, workflow_id: str) -> tuple[Any, Path]:
        """Load workflow YAML using an optional predefined output_dir."""
        return load_workflow_yaml(workflow_id, output_dir=self.output_dir)
