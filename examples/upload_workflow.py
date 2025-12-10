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


def main() -> None:
    parser = argparse.ArgumentParser(description="Login and upload a workflow YAML")
    parser.add_argument("--base-url", required=True, help="Dify 控制台 base_url，例如 http://xy.dnset.com:1280")
    parser.add_argument("--username", help="控制台用户名（邮箱）")
    parser.add_argument("--password", help="控制台密码")
    parser.add_argument("--yaml", required=True, help="要上传的本地 YAML 路径")
    args = parser.parse_args()

    token = None
    if args.username and args.password:
        client = DifyAuthClient(base_url=args.base_url, email=args.username, password=args.password)
        login_resp = client.login()
        token = login_resp.get("access_token") if isinstance(login_resp, dict) else None
        if not token:
            raise RuntimeError("登录失败，未获取到 access_token")

    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML 文件不存在: {yaml_path}")

    resp = import_app_yaml(
        yaml_path=yaml_path,
        base_url=args.base_url,
        token=token,
    )
    print("上传完成，响应：", resp)


if __name__ == "__main__":
    main()
