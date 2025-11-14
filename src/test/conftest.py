"""
pytest 配置和共享 fixtures

Date: 2025-11-13
Author: backend-developer
"""

import pytest


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """配置测试日志输出"""
    # 使用 loguru 的默认配置即可
    # ExcelExporter 已经使用了 _get_safe_logger() 来处理日志未初始化的情况
    pass
