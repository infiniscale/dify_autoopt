"""
pytest 配置和共享 fixtures

Date: 2025-11-13
Author: backend-developer
"""

import sys
from pathlib import Path
import pytest


# 将项目根目录加入 sys.path，确保 `import src...` 可用
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session", autouse=True)
def setup_logging():
    """配置测试日志输出（保持最小化）"""
    # 使用 loguru 的默认配置即可
    pass
