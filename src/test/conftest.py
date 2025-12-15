"""
pytest 配置和共享 fixtures

Date: 2025-11-13
Author: backend-developer
"""

import sys
from pathlib import Path
import pytest


"""Path setup for tests.

Ensures both project root (so `import src...` works) and the tests root
directory (so pytest can import package-style conftest like
`executor.conftest`) are available on sys.path during collection.
"""

# 项目根目录: /.../<repo>
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 测试根目录: /.../<repo>/src/test
TEST_ROOT = Path(__file__).resolve().parent
if str(TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(TEST_ROOT))


@pytest.fixture(scope="session", autouse=True)
def init_logging():
    """Initialize project logging for tests (minimal)."""
    import asyncio
    from src.utils.logger import setup_logging as _setup

    loop = asyncio.get_event_loop()
    loop.run_until_complete(_setup())
    yield
