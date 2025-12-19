"""
异常处理覆盖测试用例
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, SSLError
from pathlib import Path

from src.auth.login import (
    DifyAuthClient, AuthenticationError, SessionExpiredError,
    PermissionDeniedError, NetworkConnectionError, ConfigurationError
)
from src.auth.token_opt import Token


class TestExceptionHandlingCoverage:
    """异常处理覆盖测试类"""

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
    def token_config(self):
        """令牌配置"""
        return {
            "dify": {"base_url": "https://test.dify.com"},
            "auth": {
                "access_token_path": "test/token.txt",
                "token_validation_timeout": 10
            }
        }

    def test_all_custom_exception_types_inheritance(self):
        """测试所有自定义异常的继承关系"""
        exceptions = [
            AuthenticationError,
            SessionExpiredError,
            PermissionDeniedError,
            NetworkConnectionError,
            ConfigurationError
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, Exception), f"{exc_class.__name__} 应该继承 Exception"

    def test_all_custom_exception_instantiation(self):
        """测试所有自定义异常的实例化"""
        test_message = "测试异常消息"

        exceptions = [
            AuthenticationError(test_message),
            SessionExpiredError(test_message),
            PermissionDeniedError(test_message),
            NetworkConnectionError(test_message),
            ConfigurationError(test_message)
        ]

        for exc in exceptions:
            assert str(exc) == test_message

    def test_authentication_error_variation_scenarios(self, client):
        """测试认证错误的各种场景"""
        error_scenarios = [
            ("无效的用户名或密码", "Invalid credentials"),
            ("账户已被锁定", "Account locked"),
            ("密码已过期", "Password expired"),
            ("账户需要验证", "Account requires verification"),
        ]

        for error_message in scenarios:
            with patch('requests.post') as mock_post:
                response = Mock()
                response.json.return_value = {"result": "fail", "message": error_message}
                response.raise_for_status.return_value = None
                response.status_code = 200
                mock_post.return_value = response

                with pytest.raises(AuthenticationError) as exc_info:
                    client.login()

                assert str(exc_info.value) == error_message

    def test_network_connection_error_detailed_scenarios(self, client):
        """测试网络连接错误的详细场景"""
        network_errors = [
            ConnectionError("Connection refused"),
            ConnectionError("Host unreachable"),
            ConnectionError("Network is down"),
            ConnectionError("No route to host"),
        ]

        for error in network_errors:
            with patch('requests.post') as mock_post:
                mock_post.side_effect = error

                with pytest.raises(NetworkConnectionError) as exc_info:
                    client.login()

                assert "无法连接到服务器" in str(exc_info.value)

    def test_timeout_exception_variations(self, client):
        """测试各种超时异常"""
        timeout_errors = [
            requests.exceptions.Timeout("Connection timeout"),
            requests.exceptions.ReadTimeout("Read timeout"),
            requests.exceptions.ConnectTimeout("Connect timeout"),
            Timeout("Generic timeout"),
        ]

        for error in timeout_errors:
            with patch('requests.post') as mock_post:
                mock_post.side_effect = error

                with pytest.raises(NetworkConnectionError) as exc_info:
                    client.login()

                assert "登录请求超时" in str(exc_info.value)

    def test_permission_denied_error_scenarios(self, client):
        """测试权限拒绝的各种场景"""
        with patch('requests.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 403
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("403 Forbidden")
            mock_post.return_value = mock_response

            with pytest.raises(PermissionDeniedError) as exc_info:
                client.login()

            assert "访问被拒绝，权限不足" in str(exc_info.value)

    def test_session_expired_error_scenarios(self, client):
        """测试会话过期的各种场景"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 401
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("401 Unauthorized")
            mock_get.return_value = mock_response

            with pytest.raises(SessionExpiredError) as exc_info:
                client.logout("expired_token")

            assert "会话已过期，请重新登录" in str(exc_info.value)

    def test_file_operation_exceptions_token_rewrite(self, token_config):
        """测试令牌重写时的文件操作异常"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = token_config

            token = Token("test_config.yaml")

            # 测试写入权限错误
            with patch('builtins.open') as mock_open:
                mock_open.side_effect = IOError("Permission denied")

                with pytest.raises(IOError, match="无法写入访问令牌文件"):
                    token.rewrite_access_token("test_token")

            # 测试目录创建失败
            with patch('src.auth.token.Path') as mock_path:
                mock_parent = Mock()
                mock_parent.mkdir.side_effect = OSError("Directory creation failed")
                mock_path.return_value.parent = mock_parent

                with patch('builtins.open'):
                    with pytest.raises(OSError, match="Directory creation failed"):
                        token.rewrite_access_token("test_token")

    def test_file_operation_exceptions_token_read(self, token_config):
        """测试令牌读取时的文件操作异常"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = token_config

            token = Token("test_config.yaml")

            # 测试读取权限错误
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open') as mock_open:
                    mock_open.side_effect = IOError("Permission denied")

                    with pytest.raises(IOError, match="无法读取访问令牌文件"):
                        token.get_access_token()

    def test_configuration_file_scenarios(self):
        """测试配置文件的各种异常场景"""
        # 测试文件不存在
        with patch('src.auth.login.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.side_effect = FileNotFoundError("Config not found")

            from src.auth.login import run
            with pytest.raises(FileNotFoundError, match="无法找到配置文件"):
                run("nonexistent.yaml")

        # 测试YAML格式错误
        with patch('src.auth.login.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.side_effect = ValueError("Invalid YAML format")

            from src.auth.login import run
            with pytest.raises(ValueError, match="Invalid YAML format"):
                run("invalid.yaml")

        # 测试权限错误
        with patch('src.auth.login.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.side_effect = PermissionError("Permission denied")

            from src.auth.login import run
            with pytest.raises(PermissionError, match="Permission denied"):
                run("protected.yaml")

    def test_exception_chaining_preservation(self, client):
        """测试异常链保存"""
        original_error = ValueError("Original message")

        with patch('requests.post') as mock_post:
            mock_post.side_effect = original_error

            try:
                client.login()
            except AuthenticationError as e:
                # 验证异常信息包含原始错误
                assert "登录失败" in str(e)

    def test_exception_in_token_validation_various_modes(self, token_config):
        """测试令牌验证的各种异常模式"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = token_config

            token = Token("test_config.yaml")
            test_token = "test_token"

            # 测试SSL错误
            with patch('requests.get') as mock_get:
                mock_get.side_effect = requests.exceptions.SSLError("SSL verification failed")

                with pytest.raises(RequestException) as exc_info:
                    token.validate_access_token()
                assert "令牌验证失败" in str(exc_info.value)

    def test_exception_handling_in_login_job(self, client):
        """测试登录任务中的异常处理"""
        exception_types = [
            AuthenticationError("认证失败"),
            NetworkConnectionError("网络错误"),
            SessionExpiredError("会话过期"),
            PermissionDeniedError("权限不足"),
            Exception("未知错误")
        ]

        for exception in exception_types:
            with patch.object(client, 'login') as mock_login:
                mock_login.side_effect = exception

                with patch('src.auth.login.Token') as mock_token_class:
                    mock_token_class.return_value = Mock()

                    # 每种异常都应该被正确重新抛出
                    with pytest.raises(type(exception)) as exc_info:
                        client.login_job()

                    # 验证异常消息格式
                    expected_message = str(exception)
                    if isinstance(exception, (AuthenticationError, NetworkConnectionError, SessionExpiredError,
                                              PermissionDeniedError)):
                        assert expected_message in str(exc_info.value)
                    else:
                        assert "登录任务执行出错" in str(exc_info.value)

    def test_config_validation_detailed_errors(self):
        """测试配置验证的详细错误信息"""
        # 测试缺少不同配置项的组合
        config_scenarios = [
            ({}, "缺少dify配置块,缺少auth配置块"),
            ({"dify": {}}, "缺少auth配置块"),
            ({"auth": {}}, "缺少dify配置块"),
            ({"dify": {}, "auth": {}}, "dify.base_url,auth.username,auth.password"),
            ({"dify": {"base_url": "https://test.com"}, "auth": {}}, "auth.username,auth.password"),
            ({"dify": {}, "auth": {"username": "test"}}, "dify.base_url,auth.password"),
        ]

        for config, expected_errors in config_scenarios:
            with patch('src.auth.login.FileSystemReader') as mock_reader:
                mock_reader.read_yaml.return_value = config

                from src.auth.login import run
                with pytest.raises((KeyError, ValueError)) as exc_info:
                    run("test_config.yaml")

                error_message = str(exc_info.value)
                for expected_error in expected_errors.split(','):
                    assert expected_error in error_message

    def test_exception_context_preservation(self, client):
        """测试异常上下文保存"""
        # 测试网络错误的上下文信息
        error_message = "Network unreachable: No route to host"
        with patch('requests.post') as mock_post:
            mock_post.side_effect = ConnectionError(error_message)

            try:
                client.login()
            except NetworkConnectionError as e:
                # 验证原始错误信息被保留
                assert "无法连接到服务器: " in str(e)
                assert "Network unreachable" in str(e)

    def test_edge_case_exception_scenarios(self, token_config):
        """测试边界情况的异常场景"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = token_config

            token = Token("test_config.yaml")

            # 测试同时发生的多个异常
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open', side_effect=MemoryError("Out of memory")):
                    with pytest.raises(RuntimeError, match="读取访问令牌失败"):
                        token.get_access_token()

    def test_exception_logging_and_monitoring(self, client):
        """测试异常日志记录和监控"""
        with patch('requests.post') as mock_post:
            mock_post.side_effect = ConnectionError("Test connection error")

            with patch('logging.getLogger') as mock_logger:
                logger = Mock()
                mock_logger.return_value = logger

                try:
                    client.login()
                except NetworkConnectionError:
                    pass

                # 验证错误日志被记录
                logger.error.assert_called()
                error_calls = [str(call) for call in logger.error.call_args_list]
                assert any("Connection Error" in call for call in error_calls)

    def test_exception_recovery_scenarios(self, client):
        """测试异常恢复场景"""
        # 模拟间歇性网络错误
        call_count = 0

        def intermittent_error(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("First attempt failed")
            else:
                response = Mock()
                response.json.return_value = {"result": "success", "data": {"access_token": "recovery_token"}}
                response.raise_for_status.return_value = None
                return response

        with patch('requests.post') as mock_post:
            mock_post.side_effect = intermittent_error

            # 第一次调用失败
            with pytest.raises(NetworkConnectionError):
                client.login()

            # 调用计数验证只尝试了一次（没有自动重试）
            assert mock_post.call_count == 1
