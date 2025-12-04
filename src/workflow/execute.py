"""
Workflow Execution Utilities

Build batched workflow inputs from unified config and execute a workflow
via Dify console API, handling file uploads first and substituting file IDs.

Notes
- Input values support three types: file, string, number.
- Detection heuristic:
  - If a value is a path to an existing file -> file
  - If a value is numeric (int/float) -> number
  - Else -> string
- For list-typed fields, all list lengths must match; scalar values are broadcast.
- Upload endpoint can be customized; defaults to `{base_url}/console/api/files/upload`.
- Run endpoint can be customized; defaults to `{base_url}/console/api/apps/{app_id}/workflows/run`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import json
import os
import requests

from src.utils.logger import get_logger, log_performance
from .apps import _mask_token


def _is_file(value: Any) -> bool:
    try:
        p = Path(str(value))
        return p.exists() and p.is_file()
    except Exception:
        return False


def _is_number(value: Any) -> bool:
    try:
        if isinstance(value, (int, float)):
            return True
        s = str(value).strip()
        if s.lower() in {"nan", "inf", "-inf"}:
            return False
        float(s)
        return True
    except Exception:
        return False


def _coerce_number(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return value
    s = str(value).strip()
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return value


def _broadcast_and_validate(inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build list of per-run inputs by broadcasting scalars and validating list lengths."""
    # Determine run count N
    list_lengths: List[int] = []
    for v in inputs.values():
        if isinstance(v, list):
            list_lengths.append(len(v))
    if list_lengths:
        if not all(n == list_lengths[0] for n in list_lengths):
            raise ValueError(f"List inputs must have the same length, got: {list_lengths}")
        N = list_lengths[0]
    else:
        N = 1

    batches: List[Dict[str, Any]] = []
    for i in range(N):
        run: Dict[str, Any] = {}
        for k, v in inputs.items():
            if isinstance(v, list):
                run[k] = v[i]
            else:
                run[k] = v
        batches.append(run)
    return batches


def _upload_file(
    file_path: Path,
    *,
    base_url: str,
    token: str,
    timeout: int,
    upload_path: Optional[str] = None,
) -> str:
    """Upload a file and return a file_id.

    Defaults to POST {base_url}/console/api/files/upload with form field 'file'.
    """
    logger = get_logger("workflow.execute")
    url = upload_path or f"{base_url}/console/api/files/upload"
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": (file_path.name, open(file_path, "rb"))}
    try:
        logger.debug(
            "Uploading file",
            extra={"url": url, "filename": file_path.name, "token": _mask_token(token)},
        )
    except Exception:
        pass
    try:
        resp = requests.post(url, headers=headers, files=files, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Try common shapes {data: {id: ...}} or {id: ...}
        fid = None
        if isinstance(data, dict):
            if isinstance(data.get("data"), dict):
                fid = data["data"].get("id") or data["data"].get("file_id")
            fid = fid or data.get("id") or data.get("file_id")
        if not fid:
            raise RuntimeError("Upload succeeded but file id not found in response")
        try:
            logger.info("File uploaded", extra={"file_id": fid})
        except Exception:
            pass
        return str(fid)
    finally:
        try:
            files["file"][1].close()  # type: ignore[index]
        except Exception:
            pass


def _resolve_api_base(passed_api_base: Optional[str]) -> Optional[str]:
    if passed_api_base:
        return passed_api_base.rstrip("/")
    try:
        from src.config.bootstrap import get_runtime
        rt = get_runtime()
        api_base = (rt.app.dify or {}).get("api_base")
        return str(api_base).rstrip("/") if api_base else None
    except Exception:
        return None


def _prepare_run_payload_v1(
    run_inputs: Dict[str, Any],
    *,
    api_base: str,
    api_key: str,
    timeout: int,
    upload_path: Optional[str],
    input_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Prepare inputs for v1 API (api_base). Upload files and return inputs mapping.

    Honors declared input types when provided:
    - file: always upload and substitute file_id (supports scalar or list)
    - number: coerce to int/float
    - string: pass through
    Falls back to heuristics when type is not declared.
    """
    prepared: Dict[str, Any] = {}
    up_path = upload_path or f"{api_base}/files/upload"

    def _handle_one(val, declared: Optional[str]):
        # file type: upload regardless of file existence heuristic
        if declared == "file":
            if isinstance(val, list):
                return [
                    _upload_file(Path(str(x)), base_url=api_base, token=api_key, timeout=timeout, upload_path=up_path)
                    for x in val
                ]
            return _upload_file(Path(str(val)), base_url=api_base, token=api_key, timeout=timeout, upload_path=up_path)
        # number type: coerce
        if declared == "number":
            if isinstance(val, list):
                return [_coerce_number(x) for x in val]
            return _coerce_number(val)
        # string or unknown: use heuristics to support legacy configs
        if isinstance(val, list):
            if val and all(_is_file(x) for x in val):
                return [
                    _upload_file(Path(str(x)), base_url=api_base, token=api_key, timeout=timeout, upload_path=up_path)
                    for x in val
                ]
            if val and all(_is_number(x) for x in val):
                return [_coerce_number(x) for x in val]
            return val
        # scalar
        if _is_file(val):
            return _upload_file(Path(str(val)), base_url=api_base, token=api_key, timeout=timeout, upload_path=up_path)
        if _is_number(val):
            return _coerce_number(val)
        return val

    for k, v in run_inputs.items():
        declared_type = (input_types or {}).get(k)
        prepared[k] = _handle_one(v, declared_type.lower() if isinstance(declared_type, str) else None)

    return prepared


@log_performance("workflow_execute_v1")
def execute_workflow_v1(
    app_id: str,
    inputs: Dict[str, Any],
    *,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 60,
    upload_path: Optional[str] = None,
    run_path: Optional[str] = None,
    user: Optional[str] = "autoopt",
    response_mode: str = "blocking",
    input_types: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Execute a workflow via Dify public API (api_base) using per-workflow api_key.

    - Uploads files to `{api_base}/files/upload`
    - Runs workflow at `{api_base}/workflows/run` with body {workflow_id, inputs, user, response_mode}
    """
    logger = get_logger("workflow.execute")
    resolved_api = _resolve_api_base(api_base)
    if not resolved_api:
        raise RuntimeError("No api_base provided and runtime not initialized.")
    if not api_key:
        raise RuntimeError("No api_key provided for workflow execution.")

    batches = _broadcast_and_validate(inputs or {})
    try:
        logger.info("Workflow execution prepared (v1)", extra={"batches": len(batches), "detail": batches, "app_id": app_id})
    except Exception:
        pass

    run_url = (run_path or f"{resolved_api}/workflows/run")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    results: List[Dict[str, Any]] = []
    for idx, run_inputs in enumerate(batches, start=1):
        inputs_prepared = _prepare_run_payload_v1(
            run_inputs,
            api_base=resolved_api,
            api_key=api_key,
            timeout=timeout,
            upload_path=upload_path,
            input_types=input_types,
        )
        body = {
            "workflow_id": app_id,
            "inputs": inputs_prepared,
            "response_mode": response_mode,
        }
        if user:
            body["user"] = user
        try:
            logger.debug(
                "Executing workflow run (v1)",
                extra={"index": idx, "url": run_url, "auth": _mask_token(api_key), "keys": list(body.keys())},
            )
        except Exception:
            pass
        resp = requests.post(run_url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        results.append(resp.json())

    try:
        logger.info("Workflow execution completed (v1)", extra={"runs": len(results)})
    except Exception:
        pass
    return results


def execute_workflow_from_config(
    app_id: str,
    *,
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    timeout: int = 60,
    upload_path: Optional[str] = None,
    run_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load inputs for a workflow from unified config and execute.

    Looks up `runtime.app.workflows` for an entry with matching id and uses its `inputs` mapping.
    """
    from src.config.bootstrap import get_runtime

    rt = get_runtime()
    wf_inputs: Dict[str, Any] = {}
    wf_api_key: Optional[str] = None
    wf_types: Dict[str, str] = {}
    for w in (rt.app.workflows or []):
        wid = getattr(w, "id", None)
        if not wid:
            continue
        if str(wid) == str(app_id):
            raw_inputs = getattr(w, "inputs", {}) or {}
            # Normalize new shape: variable: {type, value} -> extract 'value'
            inputs_norm: Dict[str, Any] = {}
            if isinstance(raw_inputs, dict):
                for k, v in raw_inputs.items():
                    if isinstance(v, dict) and 'value' in v:
                        inputs_norm[k] = v.get('value')
                        if isinstance(v.get('type'), str):
                            wf_types[k] = v.get('type')
                    else:
                        inputs_norm[k] = v
            else:
                inputs_norm = {}
            wf_inputs = inputs_norm
            wf_api_key = getattr(w, "api_key", None)
            break

    if not wf_inputs:
        raise ValueError(f"No inputs found in config for workflow id: {app_id}")

    # Require public API (api_base) with per-workflow api_key
    api_base_resolved = _resolve_api_base(None)
    if not (wf_api_key and api_base_resolved):
        raise RuntimeError("Workflow execution requires dify.api_base and workflow.api_key in config")
    return execute_workflow_v1(
        app_id,
        wf_inputs,
        api_base=api_base_resolved,
        api_key=wf_api_key,
        timeout=timeout,
        upload_path=None,  # default {api_base}/files/upload
        run_path=None,     # default {api_base}/workflows/run
        input_types=wf_types or None,
    )
