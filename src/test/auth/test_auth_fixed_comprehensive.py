"""
修复后的认证模块综合测试 - 运行所有修复后验证测试
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

# 导入主要模块
from src.auth.login import DifyAuthClient, AuthenticationError
from src.auth.token_opt import Token

# 标记这个模块为综合测试
pytestmark = pytest.mark.auth_fixed_tests

class TestAuthenticationModuleFixed:
    """修复后的认证模块综合测试"""

    def test_module_imports_successfully(self):
        """测试模块可以成功导入"""
        from src.auth.login import DifyAuthClient, AuthenticationError
        from src.auth.token_opt import Token

        # 验证类可以实例化
        client = DifyAuthClient("https://test.com")
        token = Token("test_config.yaml")

        assert client is not None
        assert token is not None

    def test_syntax_errors_fixed(self):
        """验证语法错误已修复"""
        from src.auth.login import DifyAuthClient
        import inspect

        # 获取login_job方法的源代码
        source = inspect.getsource(DifyAuthClient.login_job)

        # 确保没有修复前的错误语法
        lines = source.split('\n')
        for line in lines:
            if 'raise f"' in line:
                assert False, f"发现未修复的异常语法: {line}"

    def test_config_validation_improved(self):
        """验证配置验证逻辑已改进"""
        from src.auth.login import run
        from src.config.loaders.config_loader import FileSystemReader

        # 测试安全的配置获取
        with pytest.raises(KeyError, match="配置文件中缺少dify配置块"):
            run("nonexistent_config.yaml")

    def test_token_improvements(self):
        """验证token模块的改进"""
        from src.auth.token_opt import Token
        import inspect

        # 验证方法名已修复
        methods = [name for name, method in inspect.getmembers(Token(), predicate=inspect.ismethod)]
        assert 'validate_access_token' in methods
        assert 'vaildate_access_token' not in methods, "仍然存在拼写错误的方法名"

        # 验证新功能存在
        assert 'clear_access_token' in methods

    def test_fixed_login_job_data_structure(self):
        """测试修复后的login_job数据处理"""
        with patch('src.auth.token.FileSystemReader') as mock_reader:
            # 模拟配置
            mock_reader.read_yaml.return_value = {
                "dify": {"base_url": "https://test.dify.com"},
                "auth": {"access_token_path": "test/token.txt"}
            }

            client = DifyAuthClient("https://test.dify.com", "test@test.com", "password")

            # 模拟登录成功的标准API响应
            with patch.object(client, 'login') as mock_login:
                mock_login.return_value = {
                    "result": "success",
                    "data": {"access_token": "test_token_123", "refresh_token": "refresh_token"}
                }

                with patch.object(Token, 'rewrite_access_token') as mock_rewrite:
                    with patch.object(Token, 'validate_access_token') as mock_validate:
                        mock_validate.return_value = True
                        mock_rewrite.return_value = True

                        # 应该成功执行，不抛出异常
                        client.login_job()

                        # 验证令牌被正确保存
                        mock_rewrite.assert_called_once_with("test_token_123")