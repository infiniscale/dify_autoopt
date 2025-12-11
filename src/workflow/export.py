"""
Workflow Export Utilities

Export a Dify application's DSL via the console API and save it to the
configured output directory (io_paths.output_dir) or a provided path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import os
import re
import requests
import json
import yaml

from src.utils.logger import get_logger, log_performance
from .apps import _resolve_token, _resolve_base_url, _mask_token, _login_token


def _infer_filename(app_id: str, content_disposition: str | None, content_type: str | None) -> str:
    if content_disposition:
        # Try to extract filename="..." or filename=...
        m = re.search(r'filename\*=UTF-8\'\'([^;]+)', content_disposition)
        if m:
            return m.group(1)
        m = re.search(r'filename="?([^";]+)"?', content_disposition)
        if m:
            return m.group(1)
    # Fallback based on content type
    if content_type:
        ct = content_type.lower()
        if 'zip' in ct:
            return f"app_{app_id}_export.zip"
        if 'yaml' in ct or 'yml' in ct or 'text/yaml' in ct or 'application/x-yaml' in ct:
            return f"app_{app_id}_export.yml"
        if 'json' in ct:
            return f"app_{app_id}_export.json"
    # Default to YAML filename
    return f"app_{app_id}_export.yml"


def _resolve_output_dir(passed: Optional[str | Path]) -> Path:
    if passed:
        p = Path(passed)
        p.mkdir(parents=True, exist_ok=True)
        return p
    # Try unified runtime config
    try:
        from src.config.bootstrap import get_runtime
        rt = get_runtime()
        out_dir = (rt.app.io_paths or {}).get("output_dir") or "./outputs"
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    except Exception:
        p = Path("./outputs")
        p.mkdir(parents=True, exist_ok=True)
        return p


def _safe_dirname_from_id(app_id: str) -> str:
    """Create a filesystem-safe directory name from an app/workflow id.

    Replaces path separators and non-alnum characters with underscores,
    keeping letters, digits, dash, underscore and dot.
    """
    try:
        name = str(app_id)
        name = name.replace('/', '_').replace('\\', '_')
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
        name = name.strip('._-') or "app"
        return name
    except Exception:
        return "app"


@log_performance("workflow_export_app_dsl")
def export_app_dsl(
    app_id: str,
    *,
    include_secret: bool = False,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    output_dir: Optional[str | Path] = None,
    timeout: int = 30,
) -> Path:
    """Export DSL for a given app and save to output_dir.

    Returns the path to the saved file.
    """
    logger = get_logger("workflow.export")

    resolved_base = _resolve_base_url(base_url)
    if not resolved_base:
        raise RuntimeError("No base_url provided and runtime not initialized.")

    resolved_token = _resolve_token(token)
    if not resolved_token:
        resolved_token = _login_token(resolved_base)
    if not resolved_token:
        raise RuntimeError("No access token available. Please login (username/password) or set DIFY_API_TOKEN.")

    url = f"{resolved_base}/console/api/apps/{app_id}/export"
    params = {"include_secret": str(include_secret).lower()}
    headers = {"Authorization": f"Bearer {resolved_token}"}

    try:
        logger.info(
            "Exporting app DSL",
            extra={
                "app_id": app_id,
                "url": url,
                "params": params,
                "token": _mask_token(resolved_token),
            },
        )
    except Exception:
        pass

    resp = requests.get(url, headers=headers, params=params, timeout=timeout, stream=True)
    resp.raise_for_status()

    content_disposition = resp.headers.get("Content-Disposition")
    content_type = resp.headers.get("Content-Type")
    # Ensure inferred filename is safe (no path components)
    filename = Path(_infer_filename(app_id, content_disposition, content_type)).name

    out_root = _resolve_output_dir(output_dir)
    subdir = out_root / _safe_dirname_from_id(app_id)
    subdir.mkdir(parents=True, exist_ok=True)

    # Determine whether to convert JSON -> YAML
    is_zip = bool(content_type and "zip" in content_type.lower())
    wrote_yaml = False

    if not is_zip:
        # Try parse as JSON; if ok, dump YAML instead
        body_bytes = resp.content  # consume full content
        try:
            payload = json.loads(body_bytes.decode(resp.encoding or "utf-8"))
            # Only convert the JSON 'data' field content into YAML if present
            to_dump = payload.get("data", payload) if isinstance(payload, dict) else payload
            # If 'data' is a JSON string, parse it to an object first
            if isinstance(to_dump, str):
                try:
                    parsed_inner = json.loads(to_dump)
                    to_dump = parsed_inner
                except Exception:
                    # Optional: try YAML load if not valid JSON, else keep raw string
                    try:
                        to_dump = yaml.safe_load(to_dump)
                    except Exception:
                        # keep as string
                        pass
            # Build .yml filename
            base_name = filename
            # simple extension swap to .yml
            if "." in base_name:
                base_name = re.sub(r"\.[^.]+$", ".yml", base_name)
            else:
                base_name = base_name + ".yml"
            out_path = subdir / base_name
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(to_dump, f, allow_unicode=True, sort_keys=False)
            wrote_yaml = True
            try:
                logger.info(
                    "App DSL exported (JSON->YAML)",
                    extra={"path": str(out_path.resolve()), "bytes": out_path.stat().st_size},
                )
            except Exception:
                pass
            return out_path
        except Exception:
            # Not JSON; fall back to raw write using inferred filename
            pass

    # Raw write (zip or non-JSON content)
    out_path = subdir / filename
    with open(out_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    try:
        logger.info(
            "App DSL exported (raw)",
            extra={"path": str(out_path.resolve()), "bytes": out_path.stat().st_size, "content_type": content_type},
        )
    except Exception:
        pass
    return out_path
