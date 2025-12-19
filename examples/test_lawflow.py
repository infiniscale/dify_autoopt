#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import mimetypes
import requests

# ========================
# 配置区域
# ========================

DIFY_BASE_URL = "http://xy.dnset.com:1280/v1"
DIFY_API_KEY = "app-mnZUnTtmjjHjyAAcS9KxYEfA"
ROOT_DIR = "./files"

# 固定字符串字段（你可以按需修改）
DEFAULT_RULESET_API_URL = "http://your-ruleset-service/api"
DEFAULT_REVIEW_BG = "None"
DEFAULT_CONTRACT_ID_PREFIX = "CID-111"
DEFAULT_FILE_ID_PREFIX = "FID-2222"

END_USER_ID = "test-auto"

ALLOWED_EXT = {"pdf", "docx", "doc", "md"}

STOP_ON_FAILURE = False


# ========================
# 文件工具函数
# ========================

def is_allowed(path: str):
    ext = os.path.splitext(path)[1].lower().replace(".", "")
    return ext in ALLOWED_EXT


def upload_file(path: str) -> str:
    url = f"{DIFY_BASE_URL}/files/upload"
    headers = {"Authorization": f"Bearer {DIFY_API_KEY}"}

    mime, _ = mimetypes.guess_type(path)
    mime = mime or "application/octet-stream"

    with open(path, "rb") as f:
        files = {
            "file": (os.path.basename(path), f, mime)
        }
        data = {"user": END_USER_ID}
        resp = requests.post(url, headers=headers, files=files, data=data, timeout=300)

    if resp.status_code not in (200, 201):
        raise RuntimeError(f"文件上传失败 {path} → {resp.text}")

    file_id = resp.json().get("id")
    print(f"[UPLOAD] {path} → file_id={file_id}")
    return file_id


def run_workflow(file_id: str, filename: str):
    url = f"{DIFY_BASE_URL}/workflows/run"
    headers = {
        "Authorization": f"Bearer {DIFY_API_KEY}",
        "Content-Type": "application/json",
    }

    # ========================
    # 构造输入字段（重点）
    # ========================

    payload = {
        "inputs": {
            # 1) ContractFile 是文件类型输入
            "ContractFile": {
                "type": "document",
                "transfer_method": "local_file",
                "upload_file_id": file_id,
            },

            # 2) 其他字段全部是字符串
            "RulesetApiUrl": DEFAULT_RULESET_API_URL,
            "ReviewBG": DEFAULT_REVIEW_BG,

            "ContractID": f"{DEFAULT_CONTRACT_ID_PREFIX}{os.path.splitext(filename)[0]}",
            "FileID": f"{DEFAULT_FILE_ID_PREFIX}{file_id[:10]}",
        },
        "response_mode": "blocking",
        "user": END_USER_ID,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=9000)

    if resp.status_code != 200:
        raise RuntimeError(f"工作流调用失败: {resp.text}")

    return resp.json()


def process_response(resp, filename):
    data = resp.get("data", {})
    status = data.get("status")
    outputs = data.get("outputs")
    error = data.get("error")
    workflow_id = data.get("workflow_id")

    print(f"[WORKFLOW] {filename} → status={status}")
    print(f"[WORKFLOW] workflow_id = {workflow_id}")

    if status == "succeeded":
        print(f"[OUTPUT] {outputs}")
        return True
    else:
        print(f"[ERROR] {error}")
        return False


# ========================
# 主流程
# ========================

def main():
    print(f"开始处理文件夹：{ROOT_DIR}")

    for root, _, files in os.walk(ROOT_DIR):
        for name in files:
            full = os.path.join(root, name)

            if not is_allowed(full):
                print(f"[SKIP] 不支持的类型：{full}")
                continue

            try:
                # 1 上传
                file_id = upload_file(full)

                # 2 执行工作流
                resp = run_workflow(file_id, name)

                # 3 检查结果
                ok = process_response(resp, name)

                if not ok and STOP_ON_FAILURE:
                    print("[STOP] 工作流失败，停止处理。")
                    return
                else:
                    print("[NEXT] 下一个文件...\n")

                time.sleep(0.4)

            except Exception as e:
                print(f"[EXCEPTION] {name} 错误: {e}")
                if STOP_ON_FAILURE:
                    return

    print("全部处理完成！")


if __name__ == "__main__":
    main()
