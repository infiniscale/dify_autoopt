"""
配置文件路径参数化测试用例
"""
import pytest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

from src.auth.login import DifyAuthClient, run, AuthenticationError
from src.auth.token import Token


class TestConfigurationParameterization:
    """配置参数化测试类"""

    @pytest.fixture
    def mock_config_data(self):
        """标准的模拟配置数据"""
        return {
            "dify": {"base_url": "https://api.dify.com"},
            "auth": {
                "username": "test@example.com",
                "password": "test_password",
                "access_token_path": "tokens/access_token.txt",
                "token_validation_timeout": 15
            }
        }

    @pytest.fixture
    def custom_config_paths(self):
        """多种配置文件路径测试案例"""
        return [
            "config/env_config.yaml",      # 默认路径
            "config/dev_config.yaml",       # 开发环境
            "config/prod_config.yaml",      # 生产环境
            "/etc/app/config.yaml",         # 系统配置
            "../config/config.yaml",        # 相对路径
            "./custom.yaml",                # 当前目录
        ]

    @patch('src.auth.login.FileSystemReader')
    @patch('src.auth.login.BackgroundScheduler')
    def test_run_with_default_config_path(self, mock_scheduler, mock_config_reader, mock_config_data):
        """测试使用默认配置路径运行"""
        mock_config_reader.read_yaml.return_value = mock_config_data

        mock_scheduler_instance = Mock()
        mock_scheduler.return_value = mock_scheduler_instance

        with patch('src.auth.login.time.sleep') as mock_sleep:
            # 模拟程序运行一次后停止
            mock_sleep.side_effect = [None, KeyboardInterrupt()]

            try:
                run()  # 使用默认配置路径
            except (KeyboardInterrupt, SystemExit):
                pass

            # 验证使用默认路径
            mock_config_reader.read_yaml.assert_called_once_with(Path("config/env_config.yaml"))

    @patch('src.auth.login.FileSystemReader')
    @patch('src.auth.login.BackgroundScheduler')
    def test_run_with_custom_config_path(self, mock_scheduler, mock_config_reader, mock_config_data):
        """测试使用自定义配置路径运行"""
        custom_path = ".env_test.yaml"
        mock_config_reader.read_yaml.return_value = mock_config_data

        mock_scheduler_instance = Mock()
        mock_scheduler.return_value = mock_scheduler_instance

        with patch('src.auth.login.time.sleep') as mock_sleep:
            mock_sleep.side_effect = [None, KeyboardInterrupt()]

            try:
                run(custom_path)  # 使用自定义配置路径
            except (KeyboardInterrupt, SystemExit):
                pass

            # 验证使用自定义路径
            mock_config_reader.read_yaml.assert_called_once_with(Path(custom_path))

    @patch('src.auth.login.FileSystemReader')
    def test_run_with_various_config_formats(self, mock_config_reader, mock_config_data):
        """测试各种配置文件格式和路径"""
        custom_path = "config/test_config.yaml"
        mock_config_reader.read_yaml.return_value = mock_config_data

        with patch('src.auth.login.BackgroundScheduler') as mock_scheduler:
            mock_scheduler_instance = Mock()
            mock_scheduler.return_value = mock_scheduler_instance

            with patch('src.auth.login.time.sleep') as mock_sleep:
                mock_sleep.side_effect = [None, KeyboardInterrupt()]

                try:
                    run(custom_path)
                except (KeyboardInterrupt, SystemExit):
                    pass

                # 验证配置读取路径被正确传递
                mock_config_reader.read_yaml.assert_called_once_with(Path(custom_path))

    @patch('src.auth.login.FileSystemReader')
    def test_run_config_with_unicode_path(self, mock_config_reader, mock_config_data):
        """测试包含Unicode字符的配置文件路径"""
        unicode_path = "配置文件/中文_路径.yaml"
        mock_config_reader.read_yaml.return_value = mock_config_data

        with patch('src.auth.login.BackgroundScheduler') as mock_scheduler:
            mock_scheduler_instance = Mock()
            mock_scheduler.return_value = mock_scheduler_instance

            with patch('src.auth.login.time.sleep') as mock_sleep:
                mock_sleep.side_effect = [None, KeyboardInterrupt()]

                try:
                    run(unicode_path)
                except (KeyboardInterrupt, SystemExit):
                    pass

                # 验证Unicode路径被正确处理
                mock_config_reader.read_yaml.assert_called_once_with(Path(unicode_path))

    def test_token_with_various_config_paths(self, mock_config_data):
        """测试Token管理器使用各种配置路径"""
        test_paths = [
            "config/token_test.yaml",
            "dev.yaml",
            "/absolute/path/config.yaml",
            "relative/path/config.yaml"
        ]

        for config_path in test_paths:
            with patch('src.auth.token.FileSystemReader') as mock_reader:
                mock_reader.read_yaml.return_value = mock_config_data

                token = Token(config_path)

                # 验证使用指定的配置路径
                mock_reader.read_yaml.assert_called_once_with(Path(config_path))

    def test_token_with_custom_config_data(self):
        """测试Token使用自定义配置数据"""
        custom_config = {
            "dify": {"base_url": "https://custom.dify.com"},
            "auth": {
                "access_token_path": "custom/path/token.txt",
                "token_validation_timeout": 30
            }
        }

        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = custom_config

            token = Token("custom_config.yaml")

            assert token.base_url == "https://custom.dify.com"
            assert token.access_token_path == "custom/path/token.txt"
            assert token.timeout == 30

    @patch('src.auth.login.FileSystemReader')
    def test_run_with_missing_dify_block(self, mock_config_reader, mock_config_data):
        """测试缺少dify配置块时的错误处理"""
        config_without_dify = {"auth": mock_config_data["auth"]}
        mock_config_reader.read_yaml.return_value = config_without_dify

        with pytest.raises(KeyError, match="dify配置块"):
            run("test_config.yaml")

    @patch('src.auth.login.FileSystemReader')
    def test_run_with_missing_auth_block(self, mock_config_reader, mock_config_data):
        """测试缺少auth配置块时的错误处理"""
        config_without_auth = {"dify": mock_config_data["dify"]}
        mock_config_reader.read_yaml.return_value = config_without_auth

        with pytest.raises(KeyError, match="auth配置块"):
            run("test_config.yaml")

    @patch('src.auth.login.FileSystemReader')
    def test_run_with_missing_config_items(self, mock_config_reader):
        """测试缺少具体配置项时的详细错误信息"""
        partial_config = {
            "dify": {},  # 缺少base_url
            "auth": {
                "username": "test@example.com"
                # 缺少password
            }
        }
        mock_config_reader.read_yaml.return_value = partial_config

        with pytest.raises(ValueError, match="缺少必要的认证信息.*dify.base_url.*auth.password"):
            run("test_config.yaml")

    def test_relative_and_absolute_path_handling(self):
        """测试相对路径和绝对路径的处理"""
        relative_config = {
            "dify": {"base_url": "https://relative.dify.com"},
            "auth": {"access_token_path": "tokens/relative.txt", "token_validation_timeout": 5}
        }

        absolute_config = {
            "dify": {"base_url": "https://absolute.dify.com"},
            "auth": {"access_token_path": "/tmp/tokens/absolute.txt", "token_validation_timeout": 20}
        }

        # 测试相对路径
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = relative_config

            token = Token("relative_config.yaml")
            assert token.access_token_path == "tokens/relative.txt"

        # 测试绝对路径
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = absolute_config

            token = Token("absolute_config.yaml")
            assert token.access_token_path == "/tmp/tokens/absolute.txt"

    @patch('src.auth.login.FileSystemReader')
    def test_environment_specific_configs(self, mock_config_reader):
        """测试特定环境的配置文件"""
        env_configs = {
            "dev_config.yaml": {
                "dify": {"base_url": "https://dev.dify.com"},
                "auth": {
                    "username": "dev@example.com",
                    "password": "dev_password",
                    "access_token_path": "dev_tokens/token.txt",
                    "token_validation_timeout": 5
                }
            },
            "staging_config.yaml": {
                "dify": {"base_url": "https://staging.dify.com"},
                "auth": {
                    "username": "staging@example.com",
                    "password": "staging_password",
                    "access_token_path": "staging_tokens/token.txt",
                    "token_validation_timeout": 10
                }
            },
            "prod_config.yaml": {
                "dify": {"base_url": "https://prod.dify.com"},
                "auth": {
                    "username": "prod@example.com",
                    "password": "prod_password",
                    "access_token_path": "/var/lib/dify/tokens/token.txt",
                    "token_validation_timeout": 30
                }
            }
        }

        for config_file, config_data in env_configs.items():
            with patch('src.auth.token.FileSystemReader') as mock_reader:
                mock_reader.read_yaml.return_value = config_data

                token = Token(f"env_configs/{config_file}")

                # 验证每个环境配置都正确加载
                assert token.base_url == config_data["dify"]["base_url"]
                assert token.access_token_path == config_data["auth"]["access_token_path"]
                assert token.timeout == config_data["auth"]["token_validation_timeout"]

    def test_config_path_with_special_characters(self):
        """测试包含特殊字符的配置路径"""
        special_paths = [
            "config/test config.yaml",          # 空格
            "config/test-config.yaml",         # 连字符
            "config/test_config.yaml",          # 下划线
            "config/test123.yaml",              # 数字
            "config/test.yaml.ext",             # 多个扩展名
        ]

        config_data = {
            "dify": {"base_url": "https://test.dify.com"},
            "auth": {"access_token_path": "token.txt", "token_validation_timeout": 10}
        }

        for path in special_paths:
            with patch('src.auth.token.FileSystemReader') as mock_reader:
                mock_reader.read_yaml.return_value = config_data

                # 验证特殊字符路径被正确处理
                token = Token(path)
                mock_reader.read_yaml.assert_called_once_with(Path(path))

    @patch('src.auth.login.FileSystemReader')
    def test_configuration_template_validation(self, mock_config_reader):
        """测试配置模板验证"""
        # 完整的预期配置结构
        expected_structure = {
            "dify": {
                "base_url": "https://api.dify.com"
            },
            "auth": {
                "username": "required",
                "password": "required",
                "access_token_path": "required",
                "token_validation_timeout": "optional"
            }
        }

        # 测试完整配置
        mock_config_reader.read_yaml.return_value = expected_structure

        with patch('src.auth.login.BackgroundScheduler') as mock_scheduler:
            mock_scheduler_instance = Mock()
            mock_scheduler.return_value = mock_scheduler_instance

            with patch('src.auth.login.time.sleep') as mock_sleep:
                mock_sleep.side_effect = [None, KeyboardInterrupt()]

                try:
                    run("complete_config.yaml")
                except (KeyboardInterrupt, SystemExit):
                    pass

                # 验证完整配置能正常工作
                mock_config_reader.read_yaml.assert_called_once_with(Path("complete_config.yaml"))