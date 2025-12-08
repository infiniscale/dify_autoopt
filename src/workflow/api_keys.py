"""
Workflow API key helper.

Provides a minimal function to fetch API keys for a given app id from the Dify console API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests

from src.utils.logger import get_logger

logger = get_logger("workflow.api_keys")


def list_api_keys(base_url: str, app_id: str, token: str, timeout: float = 10.0) -> List[Dict[str, Any]]:
    """
    Fetch API keys for a given app id from the Dify console API.

    Args:
        base_url: Dify base URL, e.g. http://xy.dnset.com:1280
        app_id: application id
        token: console access token (Bearer)
        timeout: request timeout in seconds

    Returns:
        A list of API key dicts (best-effort parsing). Returns [] on errors.
    """
    url = f"{base_url.rstrip('/')}/console/api/apps/{app_id}/api-keys"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        logger.info("Fetching API keys", extra={"url": url, "app_id": app_id})
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        keys = data.get("data") or data.get("keys") or data
        if isinstance(keys, list):
            return keys
    except Exception as exc:  # noqa: BLE001
        try:
            logger.warning("Failed to fetch API keys", extra={"url": url, "app_id": app_id, "error": str(exc)})
        except Exception:
            pass
    return []
