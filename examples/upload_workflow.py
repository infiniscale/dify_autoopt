"""
示例脚本：登录控制台并上传本地 DSL 文件。

用法：
  python examples/upload_workflow.py --base-url http://xy.dnset.com:1280 \
    --username your_email --password your_password \
    --yaml ./outputs/3f6422ab-3ed6-4a6d-af0e-b5df715c0080/app_3f6422ab-3ed6-4a6d-af0e-b5df715c0080_export.yml

或使用已配置的 DIFY_API_TOKEN 直接上传：
  DIFY_API_TOKEN=... python examples/upload_workflow.py --base-url http://xy.dnset.com:1280 --yaml <path>
"""

import argparse
from pathlib import Path

from src.auth.login import DifyAuthClient
from src.workflow.imports import import_app_yaml
from src.config.bootstrap import bootstrap_from_unified, get_runtime


def main() -> None:
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

    if not base_url:
        raise RuntimeError("配置中缺少 dify.base_url，请确认 config/config.yaml 设置正确")

    token = None
    if username and password:
        client = DifyAuthClient(base_url=base_url, email=username, password=password)
        login_resp = client.login()
        token = login_resp.get("access_token") if isinstance(login_resp, dict) else None
        if not token:
            raise RuntimeError("登录失败，未获取到 access_token")

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 文件不存在: {yaml_path}")

    resp = import_app_yaml(
        yaml_path=yaml_path,
        base_url=base_url,
        token=token,
    )
    print("上传完成，响应：", resp)


if __name__ == "__main__":
    main()
