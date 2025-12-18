"""
Workflow Apps API helpers

Provide utilities to fetch the current list of Dify apps using the console API.

Usage example (fetch all pages):
    from src.workflow.apps import list_all_apps
    apps = list_all_apps(url="http://xy.dnset.com:1280/console/api/apps?page=1&limit=30&name=&is_created_by_me=false")
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List

import os
import requests
from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode

from src.utils.logger import get_logger, log_performance


def _mask_token(token: Optional[str]) -> str:
    try:
        if not token:
            return "<empty>"
        if len(token) < 8:
            return "<masked>"
        return f"{token[:4]}****{token[-4:]}"
    except Exception:
        return "<masked>"


def _resolve_token(passed_token: Optional[str]) -> Optional[str]:
    if passed_token:
        return passed_token
    # Env first
    env_token = os.getenv("DIFY_API_TOKEN") or os.getenv("ACCESS_TOKEN")
    if env_token:
        return env_token.strip()
    # Runtime auth api_key/access_token fallback (登录后可直接放在 config.auth.api_key/access_token)
    try:
        from src.config.bootstrap import get_runtime
        rt = get_runtime()
        auth_cfg = getattr(rt, "app", None)
        auth_cfg = auth_cfg.auth if auth_cfg else None
        if auth_cfg:
            for key in ("api_key", "access_token"):
                val = auth_cfg.get(key) if isinstance(auth_cfg, dict) else getattr(auth_cfg, key, None)
                if val:
                    return str(val).strip()
    except Exception:
        pass
    # Token store fallback
    try:
        from pathlib import Path
        cfg_path = Path("config/env_config.yaml")
        if not cfg_path.exists():
            return None
        from src.auth.token_opt import Token
        token = Token(config_path=str(cfg_path)).get_access_token()
        return token.strip() if token else None
    except Exception:
        return None


def _login_token(base_url: str) -> Optional[str]:
    """Attempt to login using runtime auth.username/password to obtain a console token."""
    try:
        from src.config.bootstrap import get_runtime
        rt = get_runtime()
        auth_cfg = rt.app.auth if getattr(rt, "app", None) else {}
        username = auth_cfg.get("username") if isinstance(auth_cfg, dict) else getattr(auth_cfg, "username", None)
        password = auth_cfg.get("password") if isinstance(auth_cfg, dict) else getattr(auth_cfg, "password", None)
        if not username or not password:
            return None
        from src.auth.login import DifyAuthClient
        client = DifyAuthClient(base_url=base_url, email=username, password=password)
        resp = client.login()
        if isinstance(resp, dict):
            return resp.get("access_token")
    except Exception:
        return None
    return None


def _resolve_base_url(passed_base_url: Optional[str]) -> Optional[str]:
    if passed_base_url:
        return passed_base_url.rstrip("/")
    try:
        from src.config.bootstrap import get_runtime
        rt = get_runtime()
        if rt.dify_base_url:
            return str(rt.dify_base_url).rstrip("/")
    except Exception:
        pass
    return None


@log_performance("workflow_list_all_apps")
def list_all_apps(
    *,
    url: Optional[str] = None,
    base_url: Optional[str] = None,
    limit: int = 50,
    name: str = "",
    is_created_by_me: bool = False,
    token: Optional[str] = None,
    timeout: int = 10,
    max_pages: int = 1000,
) -> List[Dict[str, Any]]:
    """Fetch all apps by paginating until completion.

    Returns a flat list of app items.
    """
    logger = get_logger("workflow.apps")

    resolved_token = _resolve_token(token)
    if not resolved_token:
        raise RuntimeError("No access token available. Provide `token` or set DIFY_API_TOKEN or configure token store.")
    headers = {"Authorization": f"Bearer {resolved_token}"}

    # Build base target and a helper to create page-specific request
    if url:
        # Normalize URL and allow overriding/ensuring expected params
        def build_url(page: int) -> str:
            parts = list(urlsplit(url))
            q = parse_qs(parts[3], keep_blank_values=True)
            q["page"] = [str(page)]
            q["limit"] = [str(limit)]
            if name is not None:
                q["name"] = [name]
            q["is_created_by_me"] = [str(is_created_by_me).lower()]
            parts[3] = urlencode(q, doseq=True)
            return urlunsplit(parts)
        params = None
    else:
        resolved_base = _resolve_base_url(base_url)
        if not resolved_base:
            raise RuntimeError("No base_url provided and runtime not initialized.")
        target_base = f"{resolved_base}/console/api/apps"
        def build_url(page: int) -> str:
            return target_base
        def build_params(page: int) -> Dict[str, Any]:
            return {
                "page": page,
                "limit": limit,
                "name": name,
                "is_created_by_me": str(is_created_by_me).lower(),
            }
        params = build_params

    all_items: List[Dict[str, Any]] = []
    total_reported: Optional[int] = None
    page = 1
    while page <= max_pages:
        target = build_url(page)
        page_params = None if (params is None) else params(page)
        try:
            logger.debug(
                "Fetching apps page",
                extra={"page": page, "limit": limit, "url": target, "params": page_params or {}, "token": _mask_token(resolved_token)},
            )
        except Exception:
            pass

        resp = requests.get(target, headers=headers, params=page_params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

        # Extract page items and total if available
        items: List[Dict[str, Any]] = []
        page_total: Optional[int] = None
        if isinstance(data, dict):
            d = data.get("data")
            if isinstance(d, dict) and isinstance(d.get("items"), list):
                items = d.get("items", [])  # type: ignore[assignment]
                page_total = d.get("total") if isinstance(d.get("total"), int) else None
            elif isinstance(d, list):
                items = d  # type: ignore[assignment]
            elif isinstance(data.get("items"), list):
                items = data.get("items", [])  # type: ignore[assignment]

        all_items.extend(items)
        if total_reported is None and page_total is not None:
            total_reported = page_total

        # Stopping conditions
        if not items:
            break
        if total_reported is not None and len(all_items) >= total_reported:
            break
        if len(items) < limit:
            break

        page += 1

    try:
        logger.info("Apps fetched (all)", extra={"count": len(all_items), "pages": page})
    except Exception:
        pass

    return all_items
