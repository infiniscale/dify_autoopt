"""
定时任务和调度器的测试用例
"""
import pytest
from unittest.mock import Mock, patch, mock_open
import time
from pathlib import Path

from apscheduler.schedulers.background import BackgroundScheduler
from src.auth.login import DifyAuthClient, AuthenticationError, NetworkConnectionError


class TestLoginJob:
    """登录任务测试类"""

    @pytest.fixture
    def client(self):
        """创建客户端实例"""
        return DifyAuthClient(
            base_url="https://test.dify.com",
            email="test@example.com",
            password="test_password"
        )

    @pytest.fixture
    def mock_login_result(self):
        """模拟登录结果"""
        return {
            "access_token": "test_access_token_123456",
            "refresh_token": "test_refresh_token_789012"
        }

    def test_login_job_success(self, client, mock_login_result):
        """测试登录任务成功执行"""
        with patch('src.auth.token_opt.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = {
                "dify": {"base_url": "https://test.dify.com"},
                "auth": {"access_token_path": "access_token.txt"},
            }
            with patch.object(client, 'login') as mock_login_method:
                mock_login_method.return_value = mock_login_result
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger
                    # 令牌写入通过 Token 管理器
                    with patch('src.auth.login.Token') as MockToken:
                        token_instance = Mock()
                        token_instance.rewrite_access_token.return_value = True
                        token_instance.validate_access_token.return_value = True
                        MockToken.return_value = token_instance

                        client.login_job()

                        # 验证登录方法被调用
                        mock_login_method.assert_called_once()
                        # 验证令牌通过Token管理器被写入
                        token_instance.rewrite_access_token.assert_called_once_with("test_access_token_123456")
                        token_instance.validate_access_token.assert_called_once()
                        # 验证日志记录
                        assert logger.info.call_count >= 2  # 开始任务 + 登录成功

    def test_login_job_failure(self, client):
        """测试登录任务失败"""
        with patch('src.auth.token_opt.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = {
                "dify": {"base_url": "https://test.dify.com"},
                "auth": {"access_token_path": "access_token.txt"},
            }
            with patch.object(client, 'login') as mock_login_method:
                mock_login_method.return_value = None
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger

                    client.login_job()

                    # 验证错误日志被记录
                    logger.error.assert_called_with("登录失败")

    def test_login_job_authentication_error(self, client):
        """测试登录任务处理认证错误"""
        with patch('src.auth.token_opt.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = {
                "dify": {"base_url": "https://test.dify.com"},
                "auth": {"access_token_path": "access_token.txt"},
            }
            with patch.object(client, 'login') as mock_login_method:
                mock_login_method.side_effect = AuthenticationError("用户名或密码错误")
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger

                    # 应该抛出异常
                    with pytest.raises(Exception, match="认证失败: 用户名或密码错误"):
                        client.login_job()

                    # 验证错误日志被记录
                    logger.error.assert_called()

    def test_login_job_network_error(self, client):
        """测试登录任务处理网络错误"""
        with patch('src.auth.token_opt.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = {
                "dify": {"base_url": "https://test.dify.com"},
                "auth": {"access_token_path": "access_token.txt"},
            }
            with patch.object(client, 'login') as mock_login_method:
                mock_login_method.side_effect = NetworkConnectionError("网络不可达")
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger

                    # 应该抛出异常
                    with pytest.raises(Exception, match="网络连接异常: 网络不可达"):
                        client.login_job()

                    # 验证错误日志被记录
                    logger.error.assert_called()

    def test_login_job_session_expired(self, client):
        """测试登录任务处理会话过期"""
        from src.auth.login import SessionExpiredError
        with patch('src.auth.token_opt.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = {
                "dify": {"base_url": "https://test.dify.com"},
                "auth": {"access_token_path": "access_token.txt"},
            }
            with patch.object(client, 'login') as mock_login_method:
                mock_login_method.side_effect = SessionExpiredError("会话已过期")
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger

                    # 应该抛出异常
                    with pytest.raises(Exception, match="会话过期: 会话已过期"):
                        client.login_job()

                    # 验证错误日志被记录
                    logger.error.assert_called()

    def test_login_job_permission_denied(self, client):
        """测试登录任务处理权限拒绝"""
        from src.auth.login import PermissionDeniedError
        with patch('src.auth.token_opt.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = {
                "dify": {"base_url": "https://test.dify.com"},
                "auth": {"access_token_path": "access_token.txt"},
            }
            with patch.object(client, 'login') as mock_login_method:
                mock_login_method.side_effect = PermissionDeniedError("权限不足")
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger

                    # 应该抛出异常
                    with pytest.raises(Exception, match="权限不足: 权限不足"):
                        client.login_job()

                    # 验证错误日志被记录
                    logger.error.assert_called()

    def test_login_job_token_masking(self, client, mock_login_result):
        """测试访问令牌掩码显示"""
        mask_token = "test_access_token_123456"
        mock_login_result["access_token"] = mask_token

        with patch('src.auth.token_opt.FileSystemReader') as mock_reader:
            mock_reader.read_yaml.return_value = {
                "dify": {"base_url": "https://test.dify.com"},
                "auth": {"access_token_path": "access_token.txt"},
            }
            with patch.object(client, 'login') as mock_login_method:
                mock_login_method.return_value = mock_login_result
                with patch('logging.getLogger') as mock_logger:
                    logger = Mock()
                    mock_logger.return_value = logger
                    with patch('src.auth.login.Token') as MockToken:
                        token_instance = Mock()
                        token_instance.rewrite_access_token.return_value = True
                        token_instance.validate_access_token.return_value = True
                        MockToken.return_value = token_instance

                        client.login_job()

                        # 验证调试日志中的令牌被掩码
                        debug_calls = [str(call) for call in logger.debug.call_args_list]
                        assert any("test****3456" in call for call in debug_calls)


class TestScheduler:
    """调度器测试类"""

    @patch('src.auth.login.FileSystemReader')
    def test_run_function_scheduler_setup(self, mock_config_reader):
        """测试run函数中调度器设置"""
        # 模拟配置
        mock_config_reader.read_yaml.return_value = {
            "dify": {"base_url": "https://test.dify.com"},
            "auth": {"username": "test@test.com", "password": "password123"}
        }

        # 模拟BackgroundScheduler
        with patch('src.auth.login.BackgroundScheduler') as mock_scheduler_class:
            mock_scheduler = Mock()
            mock_scheduler_class.return_value = mock_scheduler
            mock_scheduler_class.return_value.add_job = Mock()
            mock_scheduler_class.return_value.start = Mock()
            mock_scheduler_class.return_value.shutdown = Mock()

            # 模拟time.sleep避免无限循环
            with patch('src.auth.login.time.sleep') as mock_sleep:
                mock_sleep.side_effect = [None, KeyboardInterrupt()]

                # 导入并测试run函数
                try:
                    from src.auth.login import run
                    run()
                except SystemExit:
                    pass  # 忽略键盘中断

                # 验证调度器配置
                mock_scheduler.add_job.assert_called_once()
                mock_scheduler.start.assert_called_once()
                mock_scheduler.assert_called_once()

    @patch('src.auth.login.FileSystemReader')
    def test_run_function_config_error(self, mock_config_reader):
        """测试run函数配置错误处理"""
        # 模拟配置文件不存在
        mock_config_reader.read_yaml.side_effect = FileNotFoundError("Config file not found")

        with patch('src.auth.login.BackgroundScheduler') as mock_scheduler_class:
            with pytest.raises(FileNotFoundError, match="无法找到配置文件"):
                from src.auth.login import run
                run()

    @patch('src.auth.login.FileSystemReader')
    def test_run_function_missing_config_keys(self, mock_config_reader):
        """测试run函数缺少配置键的错误处理"""
        # 模拟配置缺少必要字段
        mock_config_reader.read_yaml.return_value = {
            "dify": {"base_url": "https://test.dify.com"},
            # 缺少 auth 配置
        }

        with patch('src.auth.login.BackgroundScheduler'):
            with pytest.raises(ValueError, match="配置文件缺少必要的认证信息"):
                from src.auth.login import run
                run()

    def test_background_scheduler_integration(self):
        """测试与BackgroundScheduler的集成"""
        scheduler = BackgroundScheduler()

        # 创建客户端
        client = DifyAuthClient("https://test.dify.com", email="test@test.com", password="123")

        # 检查是否可以添加任务
        try:
            scheduler.add_job(
                client.login_job,
                'interval',
                hours=1,
                id='test_job'
            )
            job_added = True
        except Exception:
            job_added = False

        finally:
            # 清理
            if scheduler.running:
                scheduler.shutdown()

        assert job_added is True

    def test_multiple_jobs_prevention(self):
        """测试防止重复任务添加"""
        scheduler = BackgroundScheduler()
        client = DifyAuthClient("https://test.dify.com")

        try:
            # 添加第一个任务
            scheduler.add_job(
                client.login_job,
                'interval',
                hours=1,
                id='duplicate_job'
            )

            # 尝试添加重复ID的任务，应该抛出异常
            with pytest.raises(Exception):
                scheduler.add_job(
                    client.login_job,
                    'interval',
                    hours=1,
                    id='duplicate_job'
                )

        finally:
            if scheduler.running:
                scheduler.shutdown()
