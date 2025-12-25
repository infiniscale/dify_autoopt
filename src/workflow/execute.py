"""
Workflow execution helpers focused on Dify public APIs.

This module keeps three responsibilities:
- Upload a file to Dify and return the `upload_file_id`.
- Send a workflow run request (blocking) and wait for the response.
- Expand workflow inputs defined in config into per-run rows and execute them sequentially.
"""

from __future__ import annotations

import mimetypes
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from requests import HTTPError

from src.utils.logger import get_logger, log_performance
from .api_keys import list_api_keys
from .apps import _mask_token, _resolve_token, _login_token
from .export import _safe_dirname_from_id

logger = get_logger("workflow.execute")
RESULT_DIRNAME = "result"
PLACEHOLDER_ID = "{ID}"
PLACEHOLDER_TIME = "{TIME}"
TIME_FORMAT = "%Y%m%d%H%M%S"
SUCCESS_STATUSES = {"success", "partial_success"}
RETRYABLE_STATUSES = {"timeout", "failed", "error", "canceled"}


class WorkflowRetryExhausted(RuntimeError):
    def __init__(
            self,
            message: str,
            *,
            last_result: Optional[Dict[str, Any]] = None,
            last_error: Optional[Exception] = None,
            status: Optional[str] = None,
            attempts: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.last_result = last_result
        self.last_error = last_error
        self.status = status
        self.attempts = attempts


def _is_file_path(value: Any) -> bool:
    try:
        path = Path(str(value)).expanduser()
        return path.is_file()
    except Exception:
        return False


def _broadcast_inputs(raw_inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Expand dict inputs into a list of rows.

    - Scalar values are broadcast.
    - List values must share the same length and will be zipped by index.
    """
    list_lengths = [len(v) for v in raw_inputs.values() if isinstance(v, list)]
    if list_lengths and any(n != list_lengths[0] for n in list_lengths):
        raise ValueError(f"List inputs must have the same length, got: {list_lengths}")

    run_count = list_lengths[0] if list_lengths else 1
    rows: List[Dict[str, Any]] = []
    for idx in range(run_count):
        row: Dict[str, Any] = {}
        for key, value in raw_inputs.items():
            row[key] = value[idx] if isinstance(value, list) else value
        rows.append(row)
    return rows


def _normalize_inputs(
    inputs: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Normalize inputs coming from config.

    - Accepts either a list of rows or a dict of field -> value.
    - Extracts declared types when the value is shaped as {"type": "...", "value": ...}.
    """
    declared_types: Dict[str, str] = {}
    if isinstance(inputs, list):
        return [dict(row) for row in inputs], declared_types

    def _normalize_key(name: str) -> str:
        # Fix common typo: FilelD (lowercase l + uppercase D) -> FileID
        if name == "FilelD":
            try:
                logger.warning("Normalize input key", extra={"from": name, "to": "FileID"})
            except Exception:
                pass
            return "FileID"
        return name

    flattened: Dict[str, Any] = {}
    for key, raw in inputs.items():
        norm_key = _normalize_key(str(key))
        if norm_key in flattened:
            raise ValueError(f"Duplicated input key after normalization: {norm_key}")
        if isinstance(raw, dict) and "value" in raw:
            flattened[norm_key] = raw.get("value")
            if isinstance(raw.get("type"), str):
                declared_types[norm_key] = raw["type"].lower()
        else:
            flattened[norm_key] = raw

    return _broadcast_inputs(flattened), declared_types


def _apply_runtime_placeholders(value: Any, replacements: Dict[str, str]) -> Any:
    if isinstance(value, str):
        rendered = value
        for token, replacement in replacements.items():
            rendered = rendered.replace(token, replacement)
        return rendered
    if isinstance(value, dict):
        return {key: _apply_runtime_placeholders(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_apply_runtime_placeholders(item, replacements) for item in value]
    return value


def _as_document(file_id: str) -> Dict[str, Any]:
    return {
        "type": "document",
        "transfer_method": "local_file",
        "upload_file_id": file_id,
    }


def upload_dify_file(
    file_path: Union[str, Path],
    *,
    base_url: str,
    api_key: str,
    user: Optional[str] = None,
    timeout: int = 600,
    upload_path: Optional[str] = None,
) -> str:
    """Upload a file to Dify and return the upload_file_id."""
    path = Path(file_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"File not found for upload: {path}")

    url = (upload_path or f"{base_url}/files/upload").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}"}
    mime, _ = mimetypes.guess_type(str(path))
    files = {"file": (path.name, path.open("rb"), mime or "application/octet-stream")}
    data = {"user": user} if user else None

    logger.debug("Uploading file to Dify", extra={"url": url, "file": path.name, "token": _mask_token(api_key)})
    try:
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        file_id = (
            payload.get("id")
            or payload.get("file_id")
            or (payload.get("data") or {}).get("id")
            or (payload.get("data") or {}).get("file_id")
        )
        if not file_id:
            raise RuntimeError("Upload succeeded but file id missing in response")
        logger.info("File uploaded", extra={"file_id": file_id})
        return str(file_id)
    finally:
        try:
            files["file"][1].close()  # type: ignore[index]
        except Exception:
            pass


def _persist_result(
    data: Any,
    output_dir: Union[str, Path],
    workflow_id: str,
    index: int,
    inputs: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist workflow run result (and inputs) to markdown file."""
    try:
        base = Path(output_dir).expanduser().resolve()
        target_dir = base / str(workflow_id) / RESULT_DIRNAME
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / f"{index}.md"
        content_lines = [
            "# Workflow Result\n",
            f"- workflow_id: {workflow_id}\n",
            f"- input_index: {index}\n",
        ]
        if inputs:
            try:
                import json as _json

                content_lines.append("\n## Inputs\n")
                content_lines.append("```json\n")
                content_lines.append(_json.dumps(inputs, ensure_ascii=False, indent=2))
                content_lines.append("\n```\n")
            except Exception:
                content_lines.append(f"\n## Inputs\n{inputs}\n")
        if meta:
            try:
                import json as _json

                content_lines.append("\n## Metadata\n")
                content_lines.append("```json\n")
                content_lines.append(_json.dumps(meta, ensure_ascii=False, indent=2))
                content_lines.append("\n```\n")
            except Exception:
                content_lines.append(f"\n## Metadata\n{meta}\n")

        content_lines.append("\n## Response\n")
        try:
            import json as _json

            content_lines.append("```json\n")
            content_lines.append(_json.dumps(data, ensure_ascii=False, indent=2))
            content_lines.append("\n```\n")
        except Exception:
            content_lines.append(str(data))
        target_file.write_text("".join(content_lines), encoding="utf-8")
        logger.debug(
            "Persisted workflow result",
            extra={"file": str(target_file), "workflow_id": workflow_id, "index": index},
        )
    except Exception as exc:  # noqa: BLE001
        try:
            logger.warning("Failed to persist workflow result", extra={"error": str(exc)})
        except Exception:
            pass


def _persist_result_json(
    data: Any,
    output_dir: Union[str, Path],
    workflow_id: str,
    index: int,
    inputs: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Path:
    """Persist workflow run result to structured JSON for downstream optimization."""
    base = Path(output_dir).expanduser().resolve()
    target_dir = base / _safe_dirname_from_id(workflow_id) / "runs"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_file = target_dir / f"run_{index}.json"
    payload = {
        "workflow_id": workflow_id,
        "index": index,
        "inputs": inputs or {},
        "result": data,
        "meta": meta or {},
    }
    try:
        import json as _json

        target_file.write_text(_json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug(
            "Persisted workflow result JSON",
            extra={"file": str(target_file), "workflow_id": workflow_id, "index": index},
        )
    except Exception as exc:  # noqa: BLE001
        try:
            logger.warning("Failed to persist workflow JSON", extra={"error": str(exc)})
        except Exception:
            pass
    return target_file


def _prepare_inputs_with_files(
    run_inputs: Dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    user: Optional[str],
    timeout: int,
    upload_path: Optional[str],
    declared_types: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    prepared: Dict[str, Any] = {}
    declared_types = declared_types or {}
    for key, value in run_inputs.items():
        declared = declared_types.get(key, "").lower()
        if declared in {"file", "document"} or _is_file_path(value):
            logger.debug("Detected file input, preparing upload", extra={"field": key, "path": str(value)})
            file_id = upload_dify_file(
                value, base_url=base_url, api_key=api_key, user=user, timeout=timeout, upload_path=upload_path
            )
            prepared[key] = _as_document(file_id)
        else:
            prepared[key] = value
    return prepared


def _run_workflow_once(
    app_id: str,
    inputs: Dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    user: Optional[str],
    timeout: int,
    response_mode: str,
    run_path: Optional[str],
) -> Dict[str, Any]:
    url = (run_path or f"{base_url}/workflows/run").rstrip("/")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: Dict[str, Any] = {"inputs": inputs, "response_mode": response_mode}
    if user:
        payload["user"] = user

    logger.debug(
        "Triggering workflow run",
        extra={"url": url, "workflow_id": app_id, "token": _mask_token(api_key), "keys": list(payload.keys())},
    )
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _ensure_versioned_base(url: str) -> str:
    """Ensure the API base ends with /v1 (without trailing slash)."""
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def _resolve_base_url(passed_base_url: Optional[str]) -> Optional[str]:
    """Resolve base_url (without /v1) from args or config, then append /v1."""
    if passed_base_url:
        return _ensure_versioned_base(passed_base_url)
    try:
        from src.config.bootstrap import get_runtime

        rt = get_runtime()
        dify_cfg = rt.app.dify or {}
        base_url = dify_cfg.get("base_url")
        if base_url:
            return _ensure_versioned_base(str(base_url))
        return None
    except Exception:
        return None


def _normalize_status(value: Any) -> str:
    lowered = str(value).strip().lower()
    if lowered in {"success", "succeeded", "ok", "completed", "done"}:
        return "success"
    if lowered in {"partial_success", "partial-success", "partial"}:
        return "partial_success"
    if lowered in {"timeout", "timed_out", "time_out"}:
        return "timeout"
    if lowered in {"failed", "failure"}:
        return "failed"
    if lowered in {"error", "errored", "exception"}:
        return "error"
    if lowered in {"canceled", "cancelled", "aborted"}:
        return "canceled"
    return lowered


def _status_from_result(result: Dict[str, Any]) -> str:
    if not isinstance(result, dict):
        return "unknown"
    candidates: List[str] = []

    def _collect(container: Dict[str, Any]) -> None:
        for key in ("status", "state", "result"):
            if key in container and container[key] is not None:
                candidates.append(str(container[key]))

    _collect(result)
    nested_result = result.get("result")
    if isinstance(nested_result, dict):
        _collect(nested_result)
        nested_data = nested_result.get("data")
        if isinstance(nested_data, dict):
            _collect(nested_data)
    data = result.get("data")
    if isinstance(data, dict):
        _collect(data)

    if not candidates and result.get("output") is not None:
        return "success"

    for cand in candidates:
        normalized = _normalize_status(cand)
        if normalized in SUCCESS_STATUSES or normalized in RETRYABLE_STATUSES:
            return normalized
    return _normalize_status(candidates[0]) if candidates else "unknown"


def _resolve_test_user(passed_user: Optional[str]) -> Optional[str]:
    """Resolve test user from args or config."""
    if passed_user:
        return passed_user
    try:
        from src.config.bootstrap import get_runtime

        rt = get_runtime()
        dify_cfg = rt.app.dify or {}
        test_user = dify_cfg.get("test_user")
        if test_user:
            return str(test_user)
        return None
    except Exception:
        return None


def _console_base_from_api(base: str) -> str:
    """Strip trailing /v1 if present to get console base URL."""
    cleaned = base.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[:-3]
    return cleaned


def _refresh_app_api_key(app_id: str, api_base_with_v1: str, logger):
    """Try to refresh app api_key using console token (env/access token or login)."""
    console_base = _console_base_from_api(api_base_with_v1)
    console_token = _resolve_token(None)
    if not console_token:
        console_token = _login_token(console_base)
    if not console_token:
        return None
    keys = list_api_keys(console_base, app_id, console_token, create_when_missing=True, create_name="auto-refresh")
    for item in keys:
        for key_name in ("api_key", "key", "apiKey", "token"):
            if item.get(key_name):
                new_key = str(item[key_name])
                try:
                    logger.info(
                        "Refreshed workflow api_key",
                        extra={
                            "workflow_id": app_id,
                            "base_url": console_base,
                            "token": _mask_token(console_token),
                            "api_key_prefix": _mask_token(new_key),
                        },
                    )
                except Exception:
                    pass
                return new_key
    return None


@log_performance("workflow_execute_v1")
def execute_workflow_v1(
    app_id: str,
    inputs: Union[Dict[str, Any], List[Dict[str, Any]]],
    *,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    timeout: int = 9000,
    batch_timeout: Optional[Union[int, float]] = None,
    upload_path: Optional[str] = None,
    run_path: Optional[str] = None,
        user: Optional[str] = None,
    response_mode: str = "blocking",
    input_types: Optional[Dict[str, str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    persist_results: bool = False,
    concurrency: int = 1,
    retry_count: int = 0,
    persist_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a workflow via the Dify public API (blocking).

    - If any input value points to a real file, upload it first.
    - Supports list inputs by expanding them into per-run rows (zip by index).
    """
    resolved_base = _resolve_base_url(base_url)
    if not resolved_base:
        raise RuntimeError("No base_url provided and runtime not initialized.")
    if not api_key:
        raise RuntimeError("No api_key provided for workflow execution.")
    resolved_user = _resolve_test_user(user) or "autoopt"
    current_api_key: str = api_key
    key_lock = Lock()

    def _get_api_key() -> str:
        with key_lock:
            return current_api_key

    def _set_api_key(new_key: str) -> None:
        nonlocal current_api_key
        with key_lock:
            current_api_key = new_key

    logger.debug(
        "Resolved workflow endpoints",
        extra={
            "base_url": resolved_base,
            "upload_path": (upload_path or f"{resolved_base}/files/upload"),
            "run_path": (run_path or f"{resolved_base}/workflows/run"),
            "api_key": _mask_token(_get_api_key()),
            "per_request_timeout": timeout,
        },
    )

    rows, declared = _normalize_inputs(inputs)
    if input_types:
        declared.update({k: v.lower() for k, v in input_types.items()})

    logger.info(
        "Prepared workflow runs",
        extra={
            "workflow_id": app_id,
            "rows": len(rows),
            "declared_file_fields": [k for k, v in declared.items() if v == "file"],
        },
    )

    results: List[Tuple[int, Dict[str, Any], Dict[str, Any], Optional[Path]]] = []

    effective_batch_timeout: Optional[float]
    try:
        effective_batch_timeout = float(batch_timeout) if batch_timeout is not None else float(timeout) * max(
            len(rows), 1
        )
    except Exception:
        effective_batch_timeout = None
    if effective_batch_timeout is None or effective_batch_timeout <= 0:
        try:
            effective_batch_timeout = float(timeout) * max(len(rows), 1)
        except Exception:
            effective_batch_timeout = float(timeout)

    # Build static metadata for persistence (avoid tokens)
    meta_base: Dict[str, Any] = {
        "execution": {
            "timeout": timeout,
            "retry_count": retry_count,
            "concurrency": concurrency,
            "response_mode": response_mode,
            "user": resolved_user,
        },
        "paths": {
            "upload_path": upload_path or f"{resolved_base}/files/upload",
            "run_path": run_path or f"{resolved_base}/workflows/run",
            "output_dir": str(Path(output_dir).resolve()) if output_dir else None,
        },
        "base_url": resolved_base,
    }
    if persist_metadata:
        try:
            meta_base.update(persist_metadata)
        except Exception:
            pass

    def _one_row(
            idx: int,
            row: Dict[str, Any],
            *,
            allow_fail_return: bool,
    ) -> Tuple[int, Dict[str, Any], Dict[str, Any], Optional[Path]]:
        attempts = max(0, retry_count) + 1
        last_exc: Optional[Exception] = None
        last_result: Optional[Dict[str, Any]] = None
        last_status: Optional[str] = None
        timestamp = time.strftime(TIME_FORMAT)
        replacements = {
            PLACEHOLDER_ID: str(idx),
            PLACEHOLDER_TIME: timestamp,
        }
        rendered_row = _apply_runtime_placeholders(row, replacements)

        def _finalize_result(
                prepared_inputs: Dict[str, Any],
                run_result: Dict[str, Any],
        ) -> Tuple[int, Dict[str, Any], Dict[str, Any], Optional[Path]]:
            persisted_path: Optional[Path] = None
            if persist_results and output_dir:
                _persist_result(run_result, output_dir, app_id, idx, prepared_inputs, row_meta)
                persisted_path = _persist_result_json(run_result, output_dir, app_id, idx, prepared_inputs, row_meta)
            try:
                logger.info(
                    "工作流单条完成",
                    extra={
                        "workflow_id": app_id,
                        "index": idx,
                        "raw_inputs": row,
                        "prepared_keys": list(prepared_inputs.keys()),
                    },
                )
            except Exception:
                pass
            return idx, prepared_inputs, run_result, persisted_path

        def _retry_exhausted_result(
                *,
                result: Optional[Dict[str, Any]] = None,
                error: Optional[Exception] = None,
                status: Optional[str] = None,
        ) -> Dict[str, Any]:
            payload: Dict[str, Any] = {
                "status": "failed",
                "error": "retry_exhausted",
                "attempts": attempts,
            }
            if status:
                payload["last_status"] = status
            if error:
                payload["exception"] = str(error)
            if result is not None:
                payload["result"] = result
            return payload

        for attempt in range(1, attempts + 1):
            api_key_local = _get_api_key()
            row_meta = dict(meta_base)
            try:
                row_meta["raw_inputs"] = row
            except Exception:
                pass
            try:
                prepared_inputs = _prepare_inputs_with_files(
                    rendered_row,
                    base_url=resolved_base,
                    api_key=api_key_local,
                    user=resolved_user,
                    timeout=timeout,
                    upload_path=upload_path,
                    declared_types=declared,
                )
                meta_fields: Dict[str, Any] = {}
                if output_dir and "__output_dir" not in prepared_inputs:
                    meta_fields["__output_dir"] = str(Path(output_dir).resolve())
                if "__workflow_id" not in prepared_inputs:
                    meta_fields["__workflow_id"] = app_id
                if "__input_index" not in prepared_inputs:
                    meta_fields["__input_index"] = idx
                prepared_inputs.update(meta_fields)
                logger.debug("Executing workflow row", extra={"index": idx, "keys": list(prepared_inputs.keys())})
                run_result = _run_workflow_once(
                    app_id,
                    prepared_inputs,
                    base_url=resolved_base,
                    api_key=api_key_local,
                    user=resolved_user,
                    timeout=timeout,
                    response_mode=response_mode,
                    run_path=run_path,
                )
                # If run succeeds after potential refresh, align local key for subsequent runs
                if api_key_local != _get_api_key():
                    _set_api_key(api_key_local)
                last_result = run_result
                last_status = _status_from_result(run_result)
                if last_status in SUCCESS_STATUSES:
                    return _finalize_result(prepared_inputs, run_result)
                if last_status in RETRYABLE_STATUSES:
                    if attempt < attempts:
                        backoff = min(2 ** (attempt - 1), 5)
                        try:
                            logger.warning(
                                "业务失败触发重试",
                                extra={
                                    "workflow_id": app_id,
                                    "index": idx,
                                    "status": last_status,
                                    "attempt": attempt,
                                    "attempts": attempts,
                                    "sleep_seconds": backoff,
                                },
                            )
                        except Exception:
                            pass
                        time.sleep(backoff)
                        continue
                    if allow_fail_return:
                        try:
                            logger.warning(
                                "重试耗尽（业务失败）",
                                extra={
                                    "workflow_id": app_id,
                                    "index": idx,
                                    "status": last_status,
                                    "attempts": attempts,
                                },
                            )
                        except Exception:
                            pass
                        failure_payload = _retry_exhausted_result(result=run_result, status=last_status)
                        return _finalize_result(prepared_inputs, failure_payload)
                    raise WorkflowRetryExhausted(
                        f"workflow failed after retries, status={last_status}",
                        last_result=run_result,
                        status=last_status,
                        attempts=attempts,
                    )
                return _finalize_result(prepared_inputs, run_result)
            except HTTPError as exc:
                status = getattr(exc.response, "status_code", None)
                # Prioritize token refresh for 401
                if status == 401:
                    new_key = _refresh_app_api_key(app_id, resolved_base, logger)
                    if new_key:
                        _set_api_key(new_key)
                        last_exc = exc
                        # retry in next loop iteration with new key
                    else:
                        raise
                else:
                    last_exc = exc
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

            if attempt < attempts:
                try:
                    logger.warning(
                        "单条运行失败，准备重试",
                        extra={"index": idx, "attempt": attempt, "max_attempts": attempts, "error": str(last_exc)},
                    )
                except Exception:
                    pass
                time.sleep(min(2 ** (attempt - 1), 5))
                continue
            if last_exc:
                if allow_fail_return:
                    try:
                        logger.warning(
                            "重试耗尽（异常）",
                            extra={
                                "workflow_id": app_id,
                                "index": idx,
                                "attempts": attempts,
                                "error": str(last_exc),
                            },
                        )
                    except Exception:
                        pass
                    failure_payload = _retry_exhausted_result(error=last_exc, status=last_status)
                    return _finalize_result(prepared_inputs, failure_payload)
                raise last_exc
        # Should not reach here
        raise RuntimeError("Unexpected retry loop exit")

    if concurrency and concurrency > 1 and len(rows) > 1:
        logger.info(
            "并发执行工作流",
            extra={
                "workflow_id": app_id,
                "rows": len(rows),
                "concurrency": concurrency,
                "batch_timeout": effective_batch_timeout,
                "per_request_timeout": timeout,
            },
        )
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {
                pool.submit(_one_row, idx, row, allow_fail_return=True): idx
                for idx, row in enumerate(rows, start=1)
            }
            try:
                for fut in as_completed(futures, timeout=effective_batch_timeout):
                    idx = futures[fut]
                    try:
                        results.append(fut.result())
                        logger.debug("工作流单条完成", extra={"index": idx})
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("单条运行失败", extra={"index": idx, "error": str(exc)})
            except FuturesTimeout:
                pending = [i for f, i in futures.items() if not f.done()]
                for f in futures:
                    if not f.done():
                        f.cancel()
                raise RuntimeError(f"并发执行超时，未完成条目: {pending}")
    else:
        for idx, row in enumerate(rows, start=1):
            results.append(_one_row(idx, row, allow_fail_return=False))

    results.sort(key=lambda x: x[0])
    persisted_files: List[Path] = []
    final_results: List[Dict[str, Any]] = []
    for idx, prepared_inputs, run_result, persisted_path in results:
        final_results.append(run_result)
        if persisted_path:
            persisted_files.append(persisted_path)

    logger.info("Workflow execution finished", extra={"runs": len(final_results)})

    if persist_results and output_dir:
        try:
            import json as _json

            base = Path(output_dir).expanduser().resolve()
            summary_dir = base / _safe_dirname_from_id(app_id) / "runs"
            summary_dir.mkdir(parents=True, exist_ok=True)
            summary_file = summary_dir / "runs_summary.json"
            failure_count = len(
                [
                    r
                    for r in final_results
                    if str(r.get("result", "")).lower() != "success" and r.get("status") not in {"success", "succeeded"}
                ]
            )
            summary_payload = {
                "workflow_id": app_id,
                "total_runs": len(final_results),
                "failures": failure_count,
                "files": [str(p) for p in persisted_files],
            }
            summary_file.write_text(_json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.debug("Persisted workflow run summary", extra={"file": str(summary_file), "workflow_id": app_id})
        except Exception as exc:  # noqa: BLE001
            try:
                logger.warning("Failed to write run summary", extra={"error": str(exc)})
            except Exception:
                pass
    return results


def execute_workflow_from_config(
    app_id: str,
    *,
    base_url: Optional[str] = None,
    timeout: int = 9000,
    batch_timeout: Optional[Union[int, float]] = None,
    upload_path: Optional[str] = None,
    run_path: Optional[str] = None,
    persist_results: bool = True,
    retry_count: Optional[int] = None,
    persist_metadata: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Load inputs for a workflow from unified config and execute each row sequentially.

    The config may define list-valued fields; each index across the lists is treated as one run.
    """
    from src.config.bootstrap import get_runtime

    rt = get_runtime()
    workflow_entry = None
    for w in (rt.app.workflows or []):
        wid = getattr(w, "id", None)
        if wid and str(wid) == str(app_id):
            workflow_entry = w
            break

    if not workflow_entry:
        raise ValueError(f"No workflow config found for id: {app_id}")

    raw_inputs = getattr(workflow_entry, "inputs", {}) or {}
    rows, declared_types = _normalize_inputs(raw_inputs)
    api_key = getattr(workflow_entry, "api_key", None)
    output_dir = (rt.app.io_paths or {}).get("output_dir")
    exec_cfg = rt.app.execution or {}
    exec_retry = retry_count if retry_count is not None else exec_cfg.get("retry_count", 0)
    effective_batch_timeout: Optional[Union[int, float]] = batch_timeout
    if effective_batch_timeout is None:
        try:
            effective_batch_timeout = float(timeout) * max(len(rows), 1)
        except Exception:
            effective_batch_timeout = None
    exec_meta = {
        "execution_config": {
            "timeout": timeout,
            "batch_timeout": effective_batch_timeout,
            "retry_count": exec_retry,
            "concurrency": exec_cfg.get("concurrency"),
            "response_mode": "blocking",
        },
        "source": "config",
    }
    if persist_metadata:
        try:
            exec_meta.update(persist_metadata)
        except Exception:
            pass
    resolved_base = _resolve_base_url(base_url)
    if not api_key or not resolved_base:
        raise RuntimeError("Workflow execution requires dify.base_url and workflow.api_key in config")

    dify_cfg = rt.app.dify or {}
    test_user = dify_cfg.get("test_user")
    return execute_workflow_v1(
        app_id,
        rows,
        base_url=resolved_base,
        api_key=api_key,
        timeout=timeout,
        batch_timeout=effective_batch_timeout,
        upload_path=upload_path,
        run_path=run_path,
        input_types=declared_types,
        output_dir=output_dir,
        persist_results=persist_results and bool(output_dir),
        concurrency=rt.app.execution.get("concurrency") if rt.app.execution else 1,
        retry_count=int(exec_retry) if isinstance(exec_retry, int) else 0,
        persist_metadata=exec_meta,
        user=str(test_user) if test_user else None,
    )


def _get_concurrency_manager(name: str):
    """Return concurrency manager by name ('workflow'|'optimizer') if runtime is initialized."""
    try:
        from src.config.bootstrap import get_runtime

        rt = get_runtime()
        cm = getattr(rt, f"{name}_concurrency", None)
        if cm:
            return cm
        conc_map = getattr(rt, "concurrency", None) or {}
        return conc_map.get(name)
    except Exception:
        return None
