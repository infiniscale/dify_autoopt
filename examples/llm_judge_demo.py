"""
Minimal script to test LLM judge payload against a reference/output pair.

Usage:
  python examples/llm_judge_demo.py --url http://localhost:8000/v1/chat/completions --model my-model --api-key sk-xxx
"""

from __future__ import annotations

import argparse
import json
import requests


def build_payload(model: str, reference: str, output: str) -> dict:
    prompt_lines = [
        "You are a strict evaluator. Treat the reference as gold output.",
        "Task: Extract must-have content units from reference, then check if ALL are covered in the workflow output.",
        "Return strict JSON with keys:",
        'verdict: "pass" or "fail" (fail if ANY must-have is missing or incorrect)',
        'missing_items: list of missing must-have units',
        'incorrect_items: list of incorrect/misinterpreted units',
        'format_ok: true/false for structural/format issues',
        "Reference (gold):",
        reference,
        "Workflow output:",
        output,
    ]
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an evaluator comparing outputs to references."},
            {"role": "user", "content": "\n".join(prompt_lines)},
        ],
        "stream": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM judge demo")
    parser.add_argument("--url", required=True, help="LLM endpoint, e.g. http://host:port/v1/chat/completions")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--api-key", default=None, help="Optional API key")
    parser.add_argument("--reference", required=False, default="示例参考文本：所有权与交付、不可抗力、发票条款需覆盖。")
    parser.add_argument("--output", required=False, default="示例输出：目前只提到了付款条款，没有提不可抗力。")
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args()

    payload = build_payload(args.model, args.reference, args.output)
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"

    print("== Payload ==")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    resp = requests.post(args.url, json=payload, headers=headers, timeout=args.timeout)
    print(f"\n== Response status: {resp.status_code} ==")
    print(resp.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
