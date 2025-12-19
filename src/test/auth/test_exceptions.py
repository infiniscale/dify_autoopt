"""
异常处理测试用例，测试各种异常场景
"""
import pytest
import responses
from unittest.mock import Mock, patch, mock_open
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, JSONDecodeError

from src.auth.login import (
    DifyAuthClient,
    AuthenticationError,
    SessionExpiredError,
    PermissionDeniedError,
    NetworkConnectionError,
    ConfigurationError
)


class TestDifyAuthClientExceptions:
    """Dify认证客户端异常处理测试类"""

    @pytest.fixture
    def client(self):
        """创建客户端实例"""
        return DifyAuthClient(
            base_url="https://test.dify.com",
            email="test@example.com",
            password="test_password",
            timeout=10
        )

    def test_login_timeout_exception(self, client):
        """测试登录超时异常"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Timeout("Request timed out")

            with pytest.raises(NetworkConnectionError, match="登录请求超时"):
                client.login()

    def test_login_connection_error(self, client):
        """测试登录连接错误异常"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = ConnectionError("Connection failed")

            with pytest.raises(NetworkConnectionError, match="无法连接到服务器"):
                client.login()

    def test_login_general_request_exception(self, client):
        """测试登录通用请求异常"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = RequestException("General request error")

            with pytest.raises(NetworkConnectionError, match="网络请求失败"):
                client.login()

    def test_login_json_decode_error(self, client):
        """测试登录JSON解码错误"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            with pytest.raises(AuthenticationError, match="服务器响应格式错误"):
                client.login()

    def test_login_unexpected_exception(self, client):
        """测试登录未预期异常"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(AuthenticationError, match="登录失败"):
                client.login()

    def test_logout_timeout_exception(self, client):
        """测试登出超时异常"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Timeout("Request timed out")

            with pytest.raises(NetworkConnectionError, match="登出请求超时"):
                client.logout("test_token")

    def test_logout_connection_error(self, client):
        """测试登出连接错误"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = ConnectionError("Connection failed")

            with pytest.raises(NetworkConnectionError, match="无法连接到服务器"):
                client.logout("test_token")

    def test_logout_json_decode_error(self, client):
        """测试登出JSON解码错误"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            with pytest.raises(AuthenticationError, match="登出响应格式错误"):
                client.logout("test_token")

    def test_logout_unexpected_exception(self, client):
        """测试登出未预期异常"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(AuthenticationError, match="登出失败"):
                client.logout("test_token")

    @responses.activate
    def test_authentication_error_with_custom_message(self, client):
        """测试带自定义消息的认证错误"""
        responses.add(
            responses.POST,
            "https://test.dify.com/console/api/login",
            json={"result": "fail", "message": "账户已被锁定"},
            status=200
        )

        with pytest.raises(AuthenticationError) as exc_info:
            client.login()

        assert str(exc_info.value) == "账户已被锁定"

    @responses.activate
    def test_various_http_status_codes(self, client):
        """测试各种HTTP状态码的异常处理"""
        test_cases = [
            (400, "认证失败: 400"),
            (404, "认证失败: HTTP 404"),
            (500, "认证失败: HTTP 500"),
            (502, "认证失败: HTTP 502"),
            (503, "认证失败: HTTP 503")
        ]

        for status_code, expected_msg in test_cases:
            responses.reset()
            responses.add(
                responses.POST,
                "https://test.dify.com/console/api/login",
                status=status_code
            )

            with pytest.raises(AuthenticationError, match=str(status_code)):
                client.login()

    def test_exception_inheritance(self):
        """测试自定义异常的继承关系"""
        assert issubclass(AuthenticationError, Exception)
        assert issubclass(SessionExpiredError, Exception)
        assert issubclass(PermissionDeniedError, Exception)
        assert issubclass(NetworkConnectionError, Exception)
        assert issubclass(ConfigurationError, Exception)

    def test_exception_can_be_caught(self, client):
        """测试异常可以被正确捕获"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Timeout("Timeout")

            # 测试可以捕获特定异常
            with pytest.raises(NetworkConnectionError):
                client.login()

            # 测试可以捕获父类异常
            with pytest.raises(Exception):
                client.login()

    def test_login_job_exception_handling(self, client):
        """测试登录任务中的异常处理"""
        with patch.object(client, 'login') as mock_login:
            mock_login.side_effect = AuthenticationError("认证失败")

            with patch('builtins.open', mock_open()):
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger

                    # 应该抛出异常而不是静默处理
                    with pytest.raises(Exception, match="认证失败: 认证失败"):
                        client.login_job()

    def test_exception_preserves_original_message(self, client):
        """测试异常保留原始错误信息"""
        test_messages = [
            "网络不可达",
            "请求超时",
            "认证服务不可用"
        ]

        for message in test_messages:
            with patch('requests.post') as mock_post:
                mock_post.side_effect = ConnectionError(message)

                try:
                    client.login()
                    assert False, "应该抛出异常"
                except NetworkConnectionError as e:
                    assert message in str(e)


class TestExceptionTypes:
    """异常类型专门测试"""

    def test_authentication_error_creation(self):
        """测试认证错误创建"""
        error = AuthenticationError("测试错误")
        assert str(error) == "测试错误"

    def test_session_expired_error_creation(self):
        """测试会话过期错误创建"""
        error = SessionExpiredError("会话过期")
        assert str(error) == "会话过期"

    def test_permission_denied_error_creation(self):
        """测试权限拒绝错误创建"""
        error = PermissionDeniedError("权限不足")
        assert str(error) == "权限不足"

    def test_network_connection_error_creation(self):
        """测试网络连接错误创建"""
        error = NetworkConnectionError("网络故障")
        assert str(error) == "网络故障"

    def test_configuration_error_creation(self):
        """测试配置错误创建"""
        error = ConfigurationError("配置缺失")
        assert str(error) == "配置缺失"
