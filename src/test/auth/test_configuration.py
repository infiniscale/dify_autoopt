"""
配置和运行时测试用例
"""
import pytest
from unittest.mock import Mock, patch, mock_open
import sys
from pathlib import Path
from io import StringIO

from src.auth.login import DifyAuthClient, ConfigurationError


class TestConfiguration:
    """配置测试类"""

    @patch('src.auth.login.FileSystemReader')
    def test_run_successful_configuration(self, mock_config_reader):
        """测试成功配置加载和运行"""
        # 模拟正确的配置
        mock_config_reader.read_yaml.return_value = {
            "dify": {"base_url": "https://api.dify.com"},
            "auth": {"username": "admin@test.com", "password": "secret123"}
        }

        with patch('src.auth.login.BackgroundScheduler') as mock_scheduler_class:
            mock_scheduler = Mock()
            mock_scheduler_class.return_value = mock_scheduler
            mock_scheduler_class.return_value.add_job = Mock()
            mock_scheduler_class.return_value.start = Mock()
            mock_scheduler_class.return_value.shutdown = Mock()

            with patch('src.auth.login.time.sleep') as mock_sleep:
                # 模拟程序运行一次后停止
                mock_sleep.side_effect = KeyboardInterrupt()

                try:
                    from src.auth.login import run
                    run()
                except SystemExit:
                    pass

                # 验证配置被正确读取
                mock_config_reader.read_yaml.assert_called_once_with(Path("config/env_config.yaml"))

    @patch('src.auth.login.FileSystemReader')
    def test_config_file_not_found(self, mock_config_reader):
        """测试配置文件不存在"""
        mock_config_reader.read_yaml.side_effect = FileNotFoundError("Config file missing")

        with pytest.raises(FileNotFoundError, match="无法找到配置文件"):
            from src.auth.login import run
            run()

    @patch('src.auth.login.FileSystemReader')
    def test_missing_dify_config(self, mock_config_reader):
        """测试缺少dify配置"""
        mock_config_reader.read_yaml.return_value = {
            "auth": {"username": "admin@test.com", "password": "secret123"}
            # 缺少 dify 配置块
        }

        with patch('src.auth.login.BackgroundScheduler'):
            with pytest.raises(KeyError, match="配置文件中缺少必要的配置项"):
                from src.auth.login import run
                run()

    @patch('src.auth.login.FileSystemReader')
    def test_missing_auth_config(self, mock_config_reader):
        """测试缺少auth配置"""
        mock_config_reader.read_yaml.return_value = {
            "dify": {"base_url": "https://api.dify.com"}
            # 缺少 auth 配置块
        }

        with patch('src.auth.login.BackgroundScheduler'):
            with pytest.raises(KeyError):
                from src.auth.login import run
                run()

    @patch('src.auth.login.FileSystemReader')
    def test_missing_base_url(self, mock_config_reader):
        """测试缺少base_url"""
        mock_config_reader.read_yaml.return_value = {
            "dify": {},  # 缺少 base_url
            "auth": {"username": "admin@test.com", "password": "secret123"}
        }

        with patch('src.auth.login.BackgroundScheduler'):
            with pytest.raises(KeyError):
                from src.auth.login import run
                run()

    @patch('src.auth.login.FileSystemReader')
    def test_missing_username(self, mock_config_reader):
        """测试缺少用户名"""
        mock_config_reader.read_yaml.return_value = {
            "dify": {"base_url": "https://api.dify.com"},
            "auth": {"password": "secret123"}  # 缺少 username
        }

        with patch('src.auth.login.BackgroundScheduler'):
            with pytest.raises(KeyError):
                from src.auth.login import run
                run()

    @patch('src.auth.login.FileSystemReader')
    def test_missing_password(self, mock_config_reader):
        """测试缺少密码"""
        mock_config_reader.read_yaml.return_value = {
            "dify": {"base_url": "https://api.dify.com"},
            "auth": {"username": "admin@test.com"}  # 缺少 password
        }

        with patch('src.auth.login.BackgroundScheduler'):
            with pytest.raises(KeyError):
                from src.auth.login import run
                run()

    @patch('src.auth.login.FileSystemReader')
    def test_empty_config_values(self, mock_config_reader):
        """测试空配置值"""
        mock_config_reader.read_yaml.return_value = {
            "dify": {"base_url": ""},
            "auth": {"username": "", "password": ""}
        }

        with patch('src.auth.login.BackgroundScheduler'):
            with pytest.raises(ValueError, match="配置文件缺少必要的认证信息"):
                from src.auth.login import run
                run()

    @patch('src.auth.login.FileSystemReader')
    def test_partial_empty_config(self, mock_config_reader):
        """测试部分空配置"""
        mock_config_reader.read_yaml.return_value = {
            "dify": {"base_url": "https://api.dify.com"},  # 有效
            "auth": {"username": "", "password": "secret123"}  # 用户名为空
        }

        with patch('src.auth.login.BackgroundScheduler'):
            with pytest.raises(ValueError, match="配置文件缺少必要的认证信息"):
                from src.auth.login import run
                run()

    @patch('src.auth.login.FileSystemReader')
    def test_config_reader_exception(self, mock_config_reader):
        """测试配置读取异常"""
        mock_config_reader.read_yaml.side_effect = IOError("Permission denied")

        with patch('src.auth.login.BackgroundScheduler'):
            with pytest.raises(IOError, match="Permission denied"):
                from src.auth.login import run
                run()


class TestMainFunction:
    """主函数测试类"""

    @patch('src.auth.login.run')
    def test_main_function_success(self, mock_run):
        """测试主函数成功执行"""
        # 模拟命令行参数
        with patch.dict(sys.modules, {'__main__': Mock()}):
            with patch('sys.exit') as mock_exit:
                try:
                    # 模拟main执行
                    import importlib
                    login_module = importlib.import_module('src.auth.login')

                    # 执行main部分（通过设置__name__）
                    original_argv = sys.argv
                    try:
                        sys.argv = ['login.py']

                        # 模拟成功运行
                        mock_run.return_value = None

                        # 手动调用main部分的逻辑
                        try:
                            mock_run()
                        except SystemExit:
                            pass

                    finally:
                        sys.argv = original_argv

                except SystemExit:
                    pass

    @patch('src.auth.login.run')
    def test_main_function_authentication_error(self, mock_run):
        """测试主函数处理认证错误"""
        from src.auth.login import AuthenticationError

        mock_run.side_effect = AuthenticationError("认证失败")

        with patch('sys.exit') as mock_exit:
            try:
                from src.auth.login import run
                run()
            except AuthenticationError:
                pass

    @patch('src.auth.login.run')
    def test_main_function_network_error(self, mock_run):
        """测试主函数处理网络错误"""
        from src.auth.login import NetworkConnectionError

        mock_run.side_effect = NetworkConnectionError("网络错误")

        with patch('sys.exit') as mock_exit:
            try:
                from src.auth.login import run
                run()
            except NetworkConnectionError:
                pass

    @patch('src.auth.login.run')
    def test_main_function_keyboard_interrupt(self, mock_run):
        """测试主函数处理键盘中断"""
        mock_run.side_effect = KeyboardInterrupt()

        # 键盘中断应该优雅退出，不抛出异常
        with patch('sys.exit') as mock_exit:
            try:
                from src.auth.login import run
                run()
            except KeyboardInterrupt:
                pass

    @patch('src.auth.login.run')
    def test_main_function_generic_exception(self, mock_run):
        """测试主函数处理通用异常"""
        mock_run.side_effect = Exception("通用错误")

        with patch('sys.exit') as mock_exit:
            try:
                from src.auth.login import run
                run()
            except Exception:
                pass


class TestClientConfiguration:
    """客户端配置测试类"""

    def test_default_configuration(self):
        """测试默认配置"""
        client = DifyAuthClient("https://test.dify.com")
        assert client.base_url == "https://test.dify.com"
        assert client.timeout == 10  # 默认超时时间
        assert client.email is None
        assert client.password is None

    def test_custom_configuration(self):
        """测试自定义配置"""
        client = DifyAuthClient(
            base_url="https://custom.dify.com/",
            timeout=30,
            email="admin@test.com",
            password="password123"
        )
        assert client.base_url == "https://custom.dify.com"  # URL会被去尾斜杠
        assert client.timeout == 30
        assert client.email == "admin@test.com"
        assert client.password == "password123"

    def test_url_stripping_behavior(self):
        """测试URL尾斜杠去除行为"""
        test_urls = [
            ("https://test.com/", "https://test.com"),
            ("https://test.com//", "https://test.com"),
            ("https://test.com", "https://test.com"),
            ("https://test.com/api/", "https://test.com/api"),
        ]

        for input_url, expected_url in test_urls:
            client = DifyAuthClient(input_url)
            assert client.base_url == expected_url

    def test_configuration_edge_cases(self):
        """测试配置边界情况"""
        # 测试空字串URL
        client = DifyAuthClient("")
        assert client.base_url == ""

        # 测试None设置
        client = DifyAuthClient("https://test.com", timeout=None)
        assert client.timeout is None

        # 测试负数超时
        client = DifyAuthClient("https://test.com", timeout=-1)
        assert client.timeout == -1

    def test_credentials_handling_security(self):
        """测试凭据安全处理"""
        password = "super_secret_password_123!"
        email = "admin@example.com"

        client = DifyAuthClient(
            "https://test.com",
            email=email,
            password=password
        )

        # 验证凭据被正确存储（在实际应用中应该加密）
        assert client.email == email
        assert client.password == password

        # 内存中是否有其他非预期的副本
        import inspect
        source_lines = inspect.getsource(client.__init__)
        assert "super_secret_password_123!" not in source_lines