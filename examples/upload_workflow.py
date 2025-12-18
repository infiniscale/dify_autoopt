"""
示例脚本：登录控制台并上传本地 DSL 文件。

用法：
  python examples/upload_workflow.py --yaml ./outputs/.../app_..._export.yml
（默认从 config/config.yaml 读取 base_url 与 auth.username/password）
如已有控制台 Token，可设置环境变量 DIFY_API_TOKEN 跳过登录。
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.bootstrap import bootstrap_from_unified, get_runtime
from src.utils.logger import setup_logging, get_logger

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # optional dependency
    load_dotenv = None


async def main() -> None:
    await setup_logging(None)
    logger = get_logger("examples.upload_workflow")
    if load_dotenv:
        try:
            load_dotenv(dotenv_path=Path(".env"), override=False)
        except Exception:
            pass

    # 延后导入依赖 get_logger 的模块，避免未初始化日志
    from src.auth.login import DifyAuthClient
    from src.workflow.imports import import_app_yaml
    from src.workflow.publish import publish_workflow
    from src.workflow.api_keys import list_api_keys

    parser = argparse.ArgumentParser(description="Login and upload a workflow YAML")
    parser.add_argument("--config", default="config/config.yaml", help="统一配置文件路径（默认 config/config.yaml）")
    parser.add_argument("--yaml", required=True, help="要上传的本地 YAML 路径")
    args = parser.parse_args()

    # 加载统一配置，提取 base_url / auth
    rt = None
    cfg_path = Path(args.config)
    if cfg_path.exists():
        try:
            rt = bootstrap_from_unified(cfg_path)
        except Exception:
            rt = None
    if rt is None:
        try:
            rt = get_runtime()
        except Exception:
            rt = None

    base_url = None
    username = None
    password = None
    if rt and getattr(rt, "app", None):
        base_url = (rt.app.dify or {}).get("base_url") or getattr(rt, "dify_base_url", None)
        auth_cfg = rt.app.auth or {}
        username = auth_cfg.get("username")
        password = auth_cfg.get("password")

    # 支持从环境变量填充占位符
    def _resolve_env_placeholder(val: str | None) -> str | None:
        if not val:
            return val
        if val.startswith("${") and val.endswith("}"):
            key = val.strip("${}")
            return os.getenv(key) or None
        return val

    username = _resolve_env_placeholder(username)
    password = _resolve_env_placeholder(password)

    if not base_url:
        raise RuntimeError("配置中缺少 dify.base_url，请确认 config/config.yaml 设置正确")

    token = None
    if username and password:
        client = DifyAuthClient(base_url=base_url, email=username, password=password)
        try:
            login_resp = client.login()
            token = login_resp.get("access_token") if isinstance(login_resp, dict) else None
        except Exception as ex:  # noqa: BLE001
            logger.error(
                "登录失败",
                extra={
                    "error": str(ex),
                    "base_url": base_url,
                    "username": f"{username[:2]}****" if isinstance(username, str) else None,
                },
            )
            raise
        if not token:
            raise RuntimeError("登录失败，未获取到 access_token；可设置 DIFY_API_TOKEN 跳过登录")

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 文件不存在: {yaml_path}")

    import_resp = import_app_yaml(yaml_path=yaml_path, base_url=base_url, token=token)
    def _extract_app_id(resp: dict) -> str | None:
        if not isinstance(resp, dict):
            return None
        if resp.get("app_id"):
            return str(resp["app_id"])
        if resp.get("id") and not resp.get("data"):
            return str(resp["id"])
        data = resp.get("data")
        if isinstance(data, dict):
            if data.get("app_id"):
                return str(data["app_id"])
            if data.get("id"):
                return str(data["id"])
            if isinstance(data.get("app"), dict) and data["app"].get("id"):
                return str(data["app"]["id"])
        return None

    app_id = _extract_app_id(import_resp) if isinstance(import_resp, dict) else None
    if not app_id:
        logger.error("导入成功但未解析到 app_id", extra={"response_keys": list(import_resp.keys()) if isinstance(import_resp, dict) else None})
        print("导入响应：", import_resp)
        return

    try:
        publish_workflow(app_id, base_url=base_url, token=token)
    except Exception as ex:  # noqa: BLE001
        logger.error("发布工作流失败", extra={"app_id": app_id, "error": str(ex)})
        raise

    keys = list_api_keys(
        base_url,
        app_id,
        token or "",
        create_when_missing=True,
        create_name=f"upload-{yaml_path.stem}",
    )
    chosen_key = None
    for item in keys:
        for key_name in ("api_key", "key", "apiKey"):
            if key_name in item and item[key_name]:
                chosen_key = str(item[key_name])
                break
        if chosen_key:
            break

    logger.info(
        "上传+发布完成",
        extra={
            "yaml": str(yaml_path),
            "base_url": base_url,
            "app_id": app_id,
            "api_key_prefix": chosen_key[:4] + "****" if chosen_key and len(chosen_key) >= 8 else chosen_key,
            "api_keys_count": len(keys),
            "raw_api_keys": keys,
        },
    )
    print("导入响应：", import_resp)
    print("API Keys：", keys)


if __name__ == "__main__":
    asyncio.run(main())
