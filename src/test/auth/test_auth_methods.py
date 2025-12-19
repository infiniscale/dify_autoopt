"""
主要认证方法的测试用例，包括登录、登出功能
"""
import pytest
import responses
from unittest.mock import Mock, patch, mock_open
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

from src.auth.login import DifyAuthClient, AuthenticationError, SessionExpiredError


class TestDifyAuthClient:
    """Dify认证客户端测试类"""

    @pytest.fixture
    def client(self):
        """创建客户端实例"""
        return DifyAuthClient(
            base_url="https://test.dify.com",
            email="test@example.com",
            password="test_password",
            timeout=10
        )

    @pytest.fixture
    def mock_login_success(self):
        """模拟登录成功响应"""
        return {
            "result": "success",
            "data": {
                "access_token": "test_access_token_123",
                "refresh_token": "test_refresh_token_456"
            }
        }

    @pytest.fixture
    def mock_logout_success(self):
        """模拟登出成功响应"""
        return {"result": "success"}

    def test_client_initialization(self):
        """测试客户端初始化"""
        client = DifyAuthClient(
            base_url="http://localhost:3000/",
            timeout=30,
            email="admin@test.com",
            password="secret123"
        )

        assert client.base_url == "http://localhost:3000"
        assert client.timeout == 30
        assert client.email == "admin@test.com"
        assert client.password == "secret123"

    @responses.activate
    def test_login_success(self, client, mock_login_success):
        """测试登录成功"""
        responses.add(
            responses.POST,
            "https://test.dify.com/console/api/login",
            json=mock_login_success,
            status=200
        )

        result = client.login()

        assert result is not None
        assert result["access_token"] == "test_access_token_123"
        assert result["refresh_token"] == "test_refresh_token_456"

    @responses.activate
    def test_login_api_failure(self, client):
        """测试登录API返回失败"""
        responses.add(
            responses.POST,
            "https://test.dify.com/console/api/login",
            json={"result": "fail", "message": "Invalid credentials"},
            status=200
        )

        with pytest.raises(AuthenticationError, match="Invalid credentials"):
            client.login()

    @responses.activate
    def test_login_401_error(self, client):
        """测试登录返回401错误"""
        responses.add(
            responses.POST,
            "https://test.dify.com/console/api/login",
            status=401
        )

        with pytest.raises(AuthenticationError, match="用户名或密码错误"):
            client.login()

    @responses.activate
    def test_login_403_error(self, client):
        """测试登录返回403错误"""
        responses.add(
            responses.POST,
            "https://test.dify.com/console/api/login",
            status=403
        )

        # 403 在实现中映射为 PermissionDeniedError
        from src.auth.login import PermissionDeniedError
        with pytest.raises(PermissionDeniedError, match="权限"):
            client.login()

    @responses.activate
    def test_login_429_error(self, client):
        """测试登录返回429错误"""
        responses.add(
            responses.POST,
            "https://test.dify.com/console/api/login",
            status=429
        )

        with pytest.raises(AuthenticationError, match="请求过于频繁"):
            client.login()

    @responses.activate
    def test_login_timeout(self, client):
        """测试登录超时"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Timeout("Request timed out")

            with pytest.raises(Exception, match="登录请求超时"):
                client.login()

    @responses.activate
    def test_logout_success(self, client, mock_logout_success):
        """测试登出成功"""
        responses.add(
            responses.GET,
            "https://test.dify.com/console/api/logout",
            json=mock_logout_success,
            status=200
        )

        result = client.logout("test_token")

        assert result is True

    @responses.activate
    def test_logout_401_error(self, client):
        """测试登出返回401错误（会话过期）"""
        responses.add(
            responses.GET,
            "https://test.dify.com/console/api/logout",
            status=401
        )

        with pytest.raises(SessionExpiredError, match="会话已过期，请重新登录"):
            client.logout("expired_token")

    @responses.activate
    def test_logout_failure_response(self, client):
        """测试登出返回失败响应"""
        responses.add(
            responses.GET,
            "https://test.dify.com/console/api/logout",
            json={"result": "fail", "message": "Invalid token"},
            status=200
        )

        with pytest.raises(AuthenticationError, match="Invalid token"):
            client.logout("invalid_token")

    @responses.activate
    def test_logout_connection_error(self, client):
        """测试登出连接错误"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = ConnectionError("Connection failed")

            with pytest.raises(Exception, match="无法连接到服务器"):
                client.logout("test_token")

    def test_login_payload_format(self, client):
        """测试登录请求负载格式"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.json.return_value = {
                "result": "success",
                "data": {
                    "access_token": "test_token",
                    "refresh_token": "test_refresh"
                }
            }
            mock_response.raise_for_status.return_value = None
            mock_post.return_value = mock_response

            client.login()

            # 验证请求参数
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['json']['email'] == "test@example.com"
            assert call_args[1]['json']['password'] == "test_password"
            assert call_args[1]['json']['language'] == "zh-Hans"
            assert call_args[1]['json']['remember_me'] is True

    def test_logout_headers(self, client):
        """测试登出请求头"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"result": "success"}
            mock_response.status_code = 200
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response

            client.logout("test_access_token")

            # 验证请求头
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert 'Authorization' in call_args[1]['headers']
            assert call_args[1]['headers']['Authorization'] == "Bearer test_access_token"
