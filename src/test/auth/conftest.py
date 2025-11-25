# 测试配置文件
import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = str(Path(__file__).parent.parent.parent.parent)
sys.path.insert(0, project_root)

# 配置日志
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# 测试常量
TEST_BASE_URL = "https://test.dify.com"
TEST_EMAIL = "test@example.com"
TEST_PASSWORD = "test_password"
TEST_TIMEOUT = 10

# 测试令牌
TEST_ACCESS_TOKEN = "test_access_token_123456"
TEST_REFRESH_TOKEN = "test_refresh_token_789012"

# 模拟配置文件路径
MOCK_CONFIG_PATH = "config/env_config.yaml"