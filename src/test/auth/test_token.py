"""
Token管理器的完整测试用例
"""
import pytest
import responses
from unittest.mock import Mock, patch, mock_open
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError
from pathlib import Path

from src.auth.token_opt import Token


class TestToken:
    """Token管理器测试类"""

    @pytest.fixture
    def mock_config(self):
        """模拟配置数据"""
        return {
            "dify": {
                "base_url": "https://test.dify.com"
            },
            "auth": {
                "access_token_path": "test/tokens/access_token.txt",
                "token_validation_timeout": 5
            }
        }

    @pytest.fixture
    def token_manager(self, mock_config):
        """创建Token管理器实例"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = mock_config
            return Token("test_config.yaml")

    @pytest.fixture
    def mock_config_missing_auth(self):
        """缺少auth配置"""
        return {"dify": {"base_url": "https://test.dify.com"}}

    @pytest.fixture
    def mock_config_missing_dify(self):
        """缺少dify配置"""
        return {"auth": {"access_token_path": "test/tokens/access_token.txt"}}

    @pytest.fixture
    def mock_config_missing_token_path(self):
        """缺少access_token_path配置"""
        return {
            "dify": {"base_url": "https://test.dify.com"},
            "auth": {"username": "test@test.com"}
        }

    def test_token_initialization_success(self, mock_config):
        """测试Token管理器成功初始化"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = mock_config
            token = Token("test_config.yaml")

            assert token.access_token_path == "test/tokens/access_token.txt"
            assert token.base_url == "https://test.dify.com"
            assert token.timeout == 5

    def test_token_initialization_default_config(self, mock_config):
        """测试Token管理器使用默认配置初始化"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = mock_config
            token = Token()  # 使用默认配置路径

            assert token.access_token_path == "test/tokens/access_token.txt"
            mock_reader.read_yaml.assert_called_once_with(Path("config/env_config.yaml"))

    def test_token_initialization_missing_auth_config(self, mock_config_missing_auth):
        """测试缺少auth配置时的初始化"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = mock_config_missing_auth

            with pytest.raises(KeyError, match="配置文件中缺少auth配置块"):
                Token("test_config.yaml")

    def test_token_initialization_missing_dify_config(self, mock_config_missing_dify):
        """测试缺少dify配置时的初始化"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = mock_config_missing_dify

            with pytest.raises(KeyError, match="配置文件中缺少dify配置块"):
                Token("test_config.yaml")

    def test_token_initialization_missing_token_path(self, mock_config_missing_token_path):
        """测试缺少access_token_path配置时的初始化"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = mock_config_missing_token_path

            with pytest.raises(ValueError, match="配置文件中缺少access_token_path配置项"):
                Token("test_config.yaml")

    def test_token_initialization_file_not_found(self):
        """测试配置文件不存在时的初始化"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.side_effect = FileNotFoundError("Config file not found")

            with pytest.raises(FileNotFoundError, match="无法找到配置文件"):
                Token("nonexistent_config.yaml")

    def test_rewrite_access_token_success(self, token_manager):
        """测试成功写入访问令牌"""
        test_token = "access_test_token_123456"

        with patch('builtins.open', mock_open()) as mock_file:
            with patch('src.auth.token.Path') as mock_path:
                mock_path.return_value.parent.mkdir = Mock()

                result = token_manager.rewrite_access_token(test_token)

                assert result is True
                mock_file.assert_called_once_with("test/tokens/access_token.txt", "w", encoding="utf-8")
                mock_file().write.assert_called_once_with(test_token)

    def test_rewrite_access_token_directory_creation(self, token_manager):
        """测试写入令牌时创建目录"""
        test_token = "access_test_token_123456"

        with patch('builtins.open', mock_open()):
            with patch('src.auth.token.Path') as mock_path:
                mock_parent = Mock()
                mock_path.return_value.parent = mock_parent

                token_manager.rewrite_access_token(test_token)

                mock_parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)

    def test_rewrite_access_token_io_error(self, token_manager):
        """测试写入令牌时的IO错误"""
        test_token = "access_test_token_123456"

        with patch('builtins.open', mock_open()) as mock_file:
            mock_file.side_effect = IOError("Permission denied")

            with pytest.raises(IOError, match="无法写入访问令牌文件"):
                token_manager.rewrite_access_token(test_token)

    def test_get_access_token_success(self, token_manager):
        """测试成功读取访问令牌"""
        test_token = "access_test_token_123456"

        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data=test_token)) as mock_file:
                result = token_manager.get_access_token()

                assert result == test_token
                mock_file.assert_called_once_with("test/tokens/access_token.txt", "r", encoding="utf-8")

    def test_get_access_token_file_not_exists(self, token_manager):
        """测试访问令牌文件不存在"""
        with patch('os.path.exists', return_value=False):
            with patch('logging.getLogger') as mock_logger:
                logger = Mock()
                mock_logger.return_value = logger

                result = token_manager.get_access_token()

                assert result is None
                logger.warning.assert_called()

    def test_get_access_token_empty_file(self, token_manager):
        """测试访问令牌文件为空"""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open(read_data="   \n  ")) as mock_file:
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger

                    result = token_manager.get_access_token()

                    assert result is None
                    logger.warning.assert_called_with("访问令牌文件为空")

    def test_get_access_token_io_error(self, token_manager):
        """测试读取令牌时的IO错误"""
        with patch('os.path.exists', return_value=True):
            with patch('builtins.open', mock_open()) as mock_file:
                mock_file.side_effect = IOError("Permission denied")

                with pytest.raises(IOError, match="无法读取访问令牌文件"):
                    token_manager.get_access_token()

    @responses.activate
    def test_validate_access_token_success(self, token_manager):
        """测试令牌验证成功"""
        test_token = "valid_access_token_123"

        # 模拟Dify API响应
        responses.add(
            responses.GET,
            "https://test.dify.com/console/api/apps",
            json={"data": []},
            status=200
        )

        with patch.object(token_manager, 'get_access_token', return_value=test_token):
            with patch('os.path.exists', return_value=True):
                result = token_manager.validate_access_token()

                assert result is True

    @responses.activate
    def test_validate_access_token_invalid(self, token_manager):
        """测试令牌验证失败"""
        test_token = "invalid_access_token_456"

        # 模拟Dify API返回401错误
        responses.add(
            responses.GET,
            "https://test.dify.com/console/api/apps",
            status=401
        )

        with patch.object(token_manager, 'get_access_token', return_value=test_token):
            with patch('os.path.exists', return_value=True):
                result = token_manager.validate_access_token()

                assert result is False

    def test_validate_access_token_file_not_exists(self, token_manager):
        """测试令牌文件不存在的情况"""
        with patch('os.path.exists', return_value=False):
            with patch('logging.getLogger') as mock_logger:
                logger = Mock()
                mock_logger.return_value = logger

                result = token_manager.validate_access_token()

                assert result is False
                logger.warning.assert_called_with("访问令牌文件不存在")

    def test_validate_access_token_token_none(self, token_manager):
        """测试无法获取令牌的情况"""
        with patch('os.path.exists', return_value=True):
            with patch.object(token_manager, 'get_access_token', return_value=None):
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger

                    result = token_manager.validate_access_token()

                    assert result is False
                    logger.warning.assert_called_with("无法读取访问令牌")

    def test_validate_access_token_timeout(self, token_manager):
        """测试令牌验证超时"""
        test_token = "timeout_token_123"

        with patch.object(token_manager, 'get_access_token', return_value=test_token):
            with patch('os.path.exists', return_value=True):
                with patch('requests.get') as mock_get:
                    mock_get.side_effect = Timeout("Request timed out")

                    with pytest.raises(RequestException, match="令牌验证请求超时"):
                        token_manager.validate_access_token()

    def test_validate_access_token_request_error(self, token_manager):
        """测试令牌验证网络错误"""
        test_token = "request_error_token_123"

        with patch.object(token_manager, 'get_access_token', return_value=test_token):
            with patch('os.path.exists', return_value=True):
                with patch('requests.get') as mock_get:
                    mock_get.side_effect = ConnectionError("Connection failed")

                    with pytest.raises(RequestException, match="令牌验证网络请求失败"):
                        token_manager.validate_access_token()

    def test_clear_access_token_file_exists(self, token_manager):
        """测试清除存在的令牌文件"""
        with patch('os.path.exists', return_value=True):
            with patch('os.remove') as mock_remove:
                result = token_manager.clear_access_token()

                assert result is True
                mock_remove.assert_called_once_with("test/tokens/access_token.txt")

    def test_clear_access_token_file_not_exists(self, token_manager):
        """测试清除不存在的令牌文件"""
        with patch('os.path.exists', return_value=False):
            with patch('os.remove') as mock_remove:
                result = token_manager.clear_access_token()

                assert result is True
                mock_remove.assert_not_called()

    def test_clear_access_token_os_error(self, token_manager):
        """测试清除令牌文件时的系统错误"""
        with patch('os.path.exists', return_value=True):
            with patch('os.remove') as mock_remove:
                mock_remove.side_effect = OSError("Permission denied")

                result = token_manager.clear_access_token()

                assert result is False

    def test_token_validation_with_different_status_codes(self, token_manager):
        """测试不同状态码的令牌验证"""
        test_token = "test_token"

        for status_code in [200, 401, 403, 500, 503]:
            with patch.object(token_manager, 'get_access_token', return_value=test_token):
                with patch('os.path.exists', return_value=True):
                    with patch('requests.get') as mock_get:
                        mock_response = Mock()
                        mock_response.status_code = status_code
                        mock_get.return_value = mock_response

                        result = token_manager.validate_access_token()

                        if status_code == 200:
                            assert result is True
                        else:
                            assert result is False

    @patch('src.auth.token.FileSystemReader')
    def test_timeout_config_default_value(self, mock_reader):
        """测试超时配置默认值"""
        config_without_timeout = {
            "dify": {"base_url": "https://test.dify.com"},
            "auth": {"access_token_path": "test/tokens/access_token.txt"}
        }
        mock_reader.read_yaml.return_value = config_without_timeout

        token = Token("test_config.yaml")
        assert token.timeout == 10  # 默认值

    def test_token_operations_with_logging(self, token_manager):
        """测试令牌操作的日志记录"""
        test_token = "log_test_token_123456"

        with patch('builtins.open', mock_open()):
            with patch('src.auth.token.Path') as mock_path:
                mock_parent = Mock()
                mock_path.return_value.parent = mock_parent

                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger

                    token_manager.rewrite_access_token(test_token)

                    # 验证成功日志
                    logger.info.assert_called()
                    # 验证调试日志（令牌掩码）
                    logger.debug.assert_called()
                    # 验证日志中包含掩码令牌
                    debug_calls = [str(call) for call in logger.debug.call_args_list]
                    assert any("123456" in call for call in debug_calls)