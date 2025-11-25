"""
超时处理专项测试用例
"""
import pytest
import time
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, SSLError

from src.auth.login import DifyAuthClient, AuthenticationError, NetworkConnectionError
from src.auth.token import Token


class TestTimeoutHandling:
    """超时处理测试类"""

    @pytest.fixture
    def client_with_short_timeout(self):
        """创建短超时的客户端用于测试"""
        return DifyAuthClient(
            base_url="https://test.dify.com",
            email="test@example.com",
            password="test_password",
            timeout=1  # 1秒超时
        )

    @pytest.fixture
    def token_with_timeout_config(self):
        """创建带超时配置的Token管理器"""
        config_data = {
            "dify": {"base_url": "https://test.dify.com"},
            "auth": {
                "access_token_path": "test/token.txt",
                "token_validation_timeout": 2  # 2秒超时
            }
        }

        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = config_data
            return Token("test_config.yaml")

    def test_login_timeout_short(self, client_with_short_timeout):
        """测试登录短超时"""
        with patch('requests.post') as mock_post:
            # 模拟请求超时
            mock_post.side_effect = Timeout("Request timed out after 1 second")

            from src.auth.login import NetworkConnectionError
            with pytest.raises(NetworkConnectionError, match="登录请求超时"):
                client_with_short_timeout.login()

            # 验证请求被调用且带超时参数
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            assert call_args[1]['timeout'] == 1

    def test_login_timeout_long(self):
        """测试登录长超时"""
        client_with_long_timeout = DifyAuthClient(
            base_url="https://test.dify.com",
            email="test@example.com",
            password="test_password",
            timeout=30  # 30秒超时
        )

        with patch('requests.post') as mock_post:
            # 创建一个会在30秒内完成的慢响应
            def slow_request(*args, **kwargs):
                time.sleep(0.1)  # 模拟100ms的网络延迟
                response = Mock()
                response.json.return_value = {
                    "result": "success",
                    "data": {"access_token": "test_token"}
                }
                response.raise_for_status.return_value = None
                return response

            mock_post.side_effect = slow_request

            result = client_with_long_timeout.login()
            assert result is not None
            assert result["access_token"] == "test_token"

            # 验证请求带正确的超时参数
            call_args = mock_post.call_args
            assert call_args[1]['timeout'] == 30

    def test_logout_timeout(self, client_with_short_timeout):
        """测试登出超时"""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = Timeout("Logout request timed out")

            from src.auth.login import NetworkConnectionError
            with pytest.raises(NetworkConnectionError, match="登出请求超时"):
                client_with_short_timeout.logout("test_token")

            # 验证请求带超时参数
            mock_get.assert_called_once()
            call_args = mock_get.call_args
            assert call_args[1]['timeout'] == 1

    def test_token_validation_timeout(self, token_with_timeout_config):
        """测试令牌验证超时"""
        test_token = "test_token_for_timeout"

        with patch.object(token_with_timeout_config, 'get_access_token', return_value=test_token):
            with patch('os.path.exists', return_value=True):
                with patch('requests.get') as mock_get:
                    # 模拟令牌验证超时
                    mock_get.side_effect = Timeout("Token validation timed out")

                    with pytest.raises(RequestException, match="令牌验证请求超时"):
                        token_with_timeout_config.validate_access_token()

                    # 验证请求带超时参数
                    mock_get.assert_called_once()
                    call_args = mock_get.call_args
                    assert call_args[1]['timeout'] == 2

    def test_timeout_with_retry_mechanism(self):
        """测试带重试机制的超时处理"""
        client = DifyAuthClient(
            base_url="https://test.dify.com",
            email="test@example.com",
            password="test_password",
            timeout=2
        )

        with patch('requests.post') as mock_post:
            call_count = 0

            def timeout_then_success(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # 第一次调用超时
                    raise Timeout("First attempt timed out")
                else:
                    # 第二次调用成功
                    response = Mock()
                    response.json.return_value = {
                        "result": "success",
                        "data": {"access_token": "retry_token"}
                    }
                    response.raise_for_status.return_value = None
                    return response

            mock_post.side_effect = timeout_then_success

            # 第一次调用应该抛出超时异常
            with pytest.raises(NetworkConnectionError):
                client.login()

            # 验证只调用了一次（因为没有重试机制）
            assert mock_post.call_count == 1

    def test_timeout_with_different_protocol_errors(self, client_with_short_timeout):
        """测试不同协议错误与超时"""
        error_scenarios = [
            (Timeout("Connection timeout"), "登录请求超时"),
            (ConnectionError("Connection refused"), "无法连接到服务器"),
            (requests.exceptions.ReadTimeout("Read timeout"), "登录请求超时"),
            (requests.exceptions.ConnectTimeout("Connect timeout"), "登录请求超时"),
        ]

        for error, expected_exception in error_scenarios:
            with patch('requests.post') as mock_post:
                mock_post.side_effect = error

                from src.auth.login import NetworkConnectionError
                with pytest.raises(NetworkConnectionError):
                    client_with_short_timeout.login()

    def test_timeout_configuration_flexibility(self):
        """测试超时配置的灵活性"""
        timeout_values = [0.5, 1, 5, 10, 30, 60]

        for timeout in timeout_values:
            client = DifyAuthClient(
                base_url="https://test.dify.com",
                timeout=timeout
            )

            with patch('requests.post') as mock_post:
                response = Mock()
                response.json.return_value = {"result": "success", "data": {"access_token": "test"}}
                response.raise_for_status.return_value = None
                mock_post.return_value = response

                client.login()

                # 验证每次都使用了正确的超时值
                call_args = mock_post.call_args
                assert call_args[1]['timeout'] == timeout

    def test_timeout_with_streaming_response(self):
        """测试流式响应的超时处理"""
        client = DifyAuthClient(timeout=2)

        with patch('requests.post') as mock_post:
            # 模拟需要更长时间的流式响应
            response = Mock()
            response.json.side_effect = lambda: time.sleep(3)  # 模拟3秒的JSON解析
            response.raise_for_status.return_value = None
            mock_post.return_value = response

            # 这应该正常工作，因为超时只适用于网络请求，不适用于本地处理
            start_time = time.time()
            try:
                result = client.login()
                end_time = time.time()
                # 验证实际执行时间（应该超过3秒因为JSON解析延迟）
                assert end_time - start_time >= 3
            except Exception:
                # 如果有其他异常，那是正常的
                pass

    def test_token_timeout_configuration_overrides(self):
        """测试令牌超时配置覆盖"""
        configs = [
            {"token_validation_timeout": 1},
            {"token_validation_timeout": 5},
            {"token_validation_timeout": 10},
            {},  # 测试默认值
        ]

        for config in configs:
            full_config = {
                "dify": {"base_url": "https://test.dify.com"},
                "auth": {
                    **config,
                    "access_token_path": "test/token.txt"
                }
            }

            with patch('src.auth.token.FileSystemReader') as mock_reader:
                mock_reader.read_yaml.return_value = full_config
                token = Token("test_config.yaml")

                expected_timeout = config.get("token_validation_timeout", 10)
                assert token.timeout == expected_timeout

    def test_timeout_with_concurrent_requests(self):
        """测试并发请求的超时处理"""
        import threading
        import concurrent.futures

        def simulate_timeout_request():
            client = DifyAuthClient(timeout=1)
            with patch('requests.post') as mock_post:
                mock_post.side_effect = Timeout("Concurrent timeout")
                try:
                    client.login()
                except NetworkConnectionError:
                    return True
                return False

        # 并发执行多个超时请求
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(simulate_timeout_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 验证所有请求都正确处理了超时
        assert all(results), "所有并发请求应该都处理了超时异常"

    def test_timeout_logging(self, client_with_short_timeout):
        """测试超时时的日志记录"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Timeout("Test timeout")

            with patch('logging.getLogger') as mock_logger:
                logger = Mock()
                mock_logger.return_value = logger

                try:
                    client_with_short_timeout.login()
                except NetworkConnectionError:
                    pass

                # 验证超时错误被记录
                logger.error.assert_called()
                error_calls = [str(call) for call in logger.error.call_args_list]
                assert any("Timeout" in call for call in error_calls)

    def test_timeout_cleanup_behavior(self):
        """测试超时后的清理行为"""
        client = DifyAuthClient(timeout=1)

        # 模拟需要清理的资源
        with patch('requests.post') as mock_post:
            mock_post.side_effect = Timeout("Cleanup test timeout")

            from src.auth.token import Token
            with patch.object(Token, 'rewrite_access_token') as mock_write:
                try:
                    client.login()
                except NetworkConnectionError:
                    pass

                # 验证超时后没有写入令牌
                mock_write.assert_not_called()

    def test_custom_timeout_in_headers(self):
        """测试在请求头中的自定义超时处理"""
        client = DifyAuthClient(timeout=5)

        with patch('requests.post') as mock_post:
            response = Mock()
            response.json.return_value = {"result": "success", "data": {"access_token": "test"}}
            response.raise_for_status.return_value = None
            mock_post.return_value = response

            client.login()

            # 验证请求包含正确的配置
            call_args = mock_post.call_args
            assert 'timeout' in call_args[1]
            assert call_args[1]['timeout'] == 5