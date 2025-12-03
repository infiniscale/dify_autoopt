"""
登录模块日志输出测试

目的：验证在未初始化集中日志时，登录模块回退到标准 logging，
并且会输出关键的 INFO/DEBUG/ERROR 日志，logger 名称为 "auth.login"。
"""

import logging
from unittest.mock import Mock, patch

import pytest

from src.auth.login import DifyAuthClient, AuthenticationError


@pytest.fixture
def client() -> DifyAuthClient:
    return DifyAuthClient(
        base_url="https://test.dify.com",
        email="tester@example.com",
        password="secret",
        timeout=5,
    )


def _mock_response_ok(payload: dict):
    m = Mock()
    m.status_code = 200
    m.json.return_value = payload
    m.raise_for_status.return_value = None
    return m


def test_login_emits_info_and_debug_logs(client, caplog):
    """登录成功应输出DEBUG开始、DEBUG完成与INFO成功日志，logger名为auth.login。"""
    caplog.set_level(logging.DEBUG, logger="auth.login")

    payload = {
        "result": "success",
        "data": {"access_token": "abcd1234efgh5678", "refresh_token": "r-xyz"},
    }
    with patch("requests.post", return_value=_mock_response_ok(payload)) as _:
        result = client.login()

    assert result["access_token"] == "abcd1234efgh5678"

    messages = [r.getMessage() for r in caplog.records if r.name == "auth.login"]
    # 关键日志片段
    assert any("开始登录请求" in msg for msg in messages)
    assert any("登录请求完成" in msg for msg in messages)
    assert any("登录成功" in msg for msg in messages)


def test_login_failure_emits_error_log(client, caplog):
    """登录失败时应输出ERROR日志，并抛出AuthenticationError。"""
    caplog.set_level(logging.ERROR, logger="auth.login")

    payload = {"result": "fail", "message": "Invalid credentials"}
    with patch("requests.post", return_value=_mock_response_ok(payload)):
        with pytest.raises(AuthenticationError):
            client.login()

    err_messages = [r.getMessage() for r in caplog.records if r.name == "auth.login"]
    assert any("Authentication Failed" in msg or "认证失败" in msg for msg in err_messages)

