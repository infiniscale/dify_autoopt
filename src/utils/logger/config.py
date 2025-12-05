"""
日期: 2025-01-12
作者: rrong
描述: 日志模块配置文件示例和配置说明
"""

# ===== 日志配置文件模板 =====

"""
# config/logging_config.yaml

# 全局日志配置
logging:
  # 基础配置
  global:
    # 日志级别: TRACE, DEBUG, INFO, WARNING, ERROR, CRITICAL
    level: "INFO"

    # 日志格式: structured（结构化JSON）, simple（简单文本）
    format: "structured"

    # 日期时间格式
    date_format: "%Y-%m-%d %H:%M:%S"

    # 是否启用结构化日志
    structured_format: true

    # 是否自动捕获未处理异常
    catch_exceptions: true

    # 是否包含异常堆栈信息
    include_traceback: true

    # 异常信息最大长度
    max_exception_length: 1000

  # 输出配置
  outputs:
    # 控制台输出
    console:
      enabled: true
      level: "INFO"
      include_colors: true

    # 文件输出
    file:
      enabled: true
      level: "DEBUG"
      path: "logs"
      rotation: "daily"           # daily, weekly, monthly, 或者具体大小如 "100 MB"
      retention: "30 days"         # 保留时间
      compression: "zip"          # 压缩格式: zip, gzip, tar

    # 错误日志单独文件
    error_file:
      enabled: true
      level: "ERROR"
      path: "logs"
      filename: "error.log"
      rotation: "daily"
      retention: "90 days"
      compression: "zip"

  # 性能配置
  performance:
    # 是否启用异步模式
    async_mode: true

    # 缓冲区大小（字节）
    buffer_size: 8192

    # 刷新间隔（秒）
    flush_interval: 5

    # 异步队列最大大小
    max_queue_size: 10000

    # 是否启用批量写入
    batch_writes: true

    # 批量写入大小
    batch_size: 100

  # 上下文配置
  context:
    # 自动添加的上下文字段
    auto_fields:
      - "module"
      - "version"
      - "request_id"
      - "session_id"

    # 自定义上下文字段
    custom_fields:
      service: "dify_autoopt"
      team: "optimization"
      environment: "production"

  # 健康检查和监控
  health:
    # 错误通知webhook（可选）
    error_notification_webhook: null
      # url: "https://hooks.slack.com/..."
      # headers:
      #   Authorization: "Bearer token"

    # 异常时自动重试
    auto_retry_on_exception: true

    # 最大重试次数
    max_auto_retry_attempts: 3

    # 重试延迟（秒）
    retry_delay: 1.0

    # 性能监控阈值
    performance_thresholds:
      max_log_latency_ms: 100       # 最大日志延迟
      max_queue_size: 8000          # 最大队列大小
      max_memory_usage_mb: 500      # 最大内存使用量

  # 高级配置
  advanced:
    # 日志文件编码
    encoding: "utf-8"

    # 是否启用日志染色（不同级别不同颜色）
    colorize: true

    # 是否启用进程ID
    process_id: true

    # 是否启用线程ID
    thread_id: true

    # 自定义格式化器（如使用自定义格式）
    # custom_formatter: "my_module.format.CustomFormatter"

    # 日志过滤规则
    filters:
      # 排除某些模块的日志
      exclude:
        - "urllib3.connectionpool"
        - "requests.packages.urllib3"

      # 只包含特定级别的日志
      include_only: []

      # 自定义过滤规则
      custom_rules:
        - condition: "module == 'noisy_module'"
          max_level: "WARNING"

  # 开发和调试配置
  development:
    # 是否启用详细调试信息
    verbose: false

    # 是否启用性能分析日志
    profile_performance: false

    # 是否记录所有HTTP请求/响应
    log_http_traffic: false

    # 是否记录SQL查询
    log_sql_queries: false

  # 生产环境特定配置
  production:
    # 是否启用日志采样（只记录部分日志以减少量）
    enable_sampling: false

    # 采样率（0.0-1.0）
    sampling_rate: 0.1

    # 关键日志不采样
    critical_logs_always: true

    # 是否启用日志聚合和发送到外部服务
    external_logging:
      enabled: false

      # Elasticsearch配置
      elasticsearch:
        hosts: ["localhost:9200"]
        index_prefix: "dify_autoopt"
        bulk_size: 100

      # Prometheus配置
      prometheus:
        enabled: false
        port: 9090
        metrics_prefix: "dify_autoopt"

      # 其他服务配置...
"""


# ===== 配置验证模式 =====
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union


class ConsoleConfig(BaseModel):
    enabled: bool = True
    level: str = "INFO"
    include_colors: bool = True

    @validator('level')
    def validate_level(cls, v):
        valid_levels = ["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level: {v}. Must be one of {valid_levels}')
        return v.upper()


class FileConfig(BaseModel):
    enabled: bool = True
    level: str = "DEBUG"
    path: str = "logs"
    rotation: str = "daily"
    retention: str = "30 days"
    compression: str = "zip"
    filename: Optional[str] = None


class HealthConfig(BaseModel):
    error_notification_webhook: Optional[Dict[str, Any]] = None
    auto_retry_on_exception: bool = True
    max_auto_retry_attempts: int = 3
    retry_delay: float = 1.0
    performance_thresholds: Dict[str, Union[int, float]] = Field(default_factory=lambda: {
        "max_log_latency_ms": 100,
        "max_queue_size": 8000,
        "max_memory_usage_mb": 500
    })


class ContextConfig(BaseModel):
    auto_fields: List[str] = Field(default_factory=lambda: [
        "module", "version", "request_id", "session_id"
    ])
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class PerformanceConfig(BaseModel):
    async_mode: bool = True
    buffer_size: int = 8192
    flush_interval: int = 5
    max_queue_size: int = 10000
    batch_writes: bool = True
    batch_size: int = 100


class OutputsConfig(BaseModel):
    console: ConsoleConfig = Field(default_factory=ConsoleConfig)
    file: FileConfig = Field(default_factory=FileConfig)
    error_file: Optional[FileConfig] = None


class GlobalConfig(BaseModel):
    level: str = "INFO"
    format: str = "structured"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    structured_format: bool = True
    catch_exceptions: bool = True
    include_traceback: bool = True
    max_exception_length: int = 1000

    @validator('format')
    def validate_format(cls, v):
        if v not in ["structured", "simple"]:
            raise ValueError('Format must be either "structured" or "simple"')
        return v


class LoggingConfig(BaseModel):
    global_config: GlobalConfig = Field(default_factory=GlobalConfig, alias="global")
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    health: HealthConfig = Field(default_factory=HealthConfig)
    advanced: Dict[str, Any] = Field(default_factory=dict)
    development: Dict[str, Any] = Field(default_factory=dict)
    production: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True
"""


# ===== 环境变量配置覆盖 =====
"""
import os
from typing import Any, Dict

def load_config_with_env_override(config_path: str) -> Dict[str, Any]:
    # 加载配置并应用环境变量覆盖

    # 加载基础配置
    base_config = load_config(config_path)

    # 环境变量映射
    env_mappings = {
        'LOG_LEVEL': ['logging', 'global', 'level'],
        'LOG_FORMAT': ['logging', 'global', 'format'],
        'LOG_FILE_PATH': ['logging', 'outputs', 'file', 'path'],
        'LOG_CONSOLE_LEVEL': ['logging', 'outputs', 'console', 'level'],
        'LOG_ASYNC_MODE': ['logging', 'performance', 'async_mode'],
        'LOG_BUFFER_SIZE': ['logging', 'performance', 'buffer_size'],
    }

    # 应用环境变量覆盖
    for env_var, config_path in env_mappings.items():
        env_value = os.getenv(env_var)
        if env_value:
            # 设置嵌套配置值
            current = base_config
            for key in config_path[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # 类型转换
            final_key = config_path[-1]
            if final_key.endswith('_mode') or final_key.endswith('enabled'):
                current[final_key] = env_value.lower() in ('true', '1', 'yes', 'on')
            elif final_key.endswith('_size') or final_key.endswith('_interval'):
                current[final_key] = int(env_value)
            else:
                current[final_key] = env_value

        return base_config
"""


# ===== 配置验证和错误处理 =====
"""
def validate_logging_config(config: Dict[str, Any]) -> None:
    # 验证日志配置
    errors = []

    try:
        logging_config = LoggingConfig(**config.get('logging', {}))
    except Exception as e:
        errors.append(f"Configuration validation failed: {e}")

    # 检查路径权限
    file_path = config.get('logging', {}).get('outputs', {}).get('file', {}).get('path', 'logs')
    try:
        Path(file_path).mkdir(parents=True, exist_ok=True)
        # 测试写入权限
        test_file = Path(file_path) / 'permission_test.log'
        test_file.touch()
        test_file.unlink()
    except PermissionError:
        errors.append(f"No write permission for log directory: {file_path}")
    except Exception as e:
        errors.append(f"Log directory check failed: {e}")

    # 检查异步模式配置
    async_mode = config.get('logging', {}).get('performance', {}).get('async_mode', False)
    if async_mode:
        max_queue_size = config.get('logging', {}).get('performance', {}).get('max_queue_size', 10000)
        if max_queue_size < 1000:
            errors.append("max_queue_size should be at least 1000 for async mode")

    if errors:
        raise ConfigurationException(
            f"Logging configuration validation failed: {'; '.join(errors)}"
        )
"""


# ===== 默认配置生成器 =====
"""
def generate_default_config() -> str:
    # 生成默认配置文件内容
    return '''
logging:
  global:
    level: "INFO"
    format: "structured"
    date_format: "%Y-%m-%d %H:%M:%S"

  outputs:
    console:
      enabled: true
      level: "INFO"

    file:
      enabled: true
      level: "DEBUG"
      path: "logs"

  performance:
    async_mode: true
    buffer_size: 8192

  context:
    auto_fields:
      - "module"
      - "version"
'''
"""


# ===== 配置迁移工具 =====
"""
def migrate_config_v1_to_v2(old_config_path: str, new_config_path: str):
    # 从v1配置迁移到v2配置
    old_config = load_config(old_config_path)

    # 转换新的配置结构
    new_config = {
        'logging': {
            'global': {
                'level': old_config.get('log_level', 'INFO'),
                'format': old_config.get('log_format', 'structured')
            },
            'outputs': {
                'console': {
                    'enabled': old_config.get('console_enabled', True),
                    'level': old_config.get('console_level', 'INFO')
                },
                'file': {
                    'enabled': old_config.get('file_enabled', True),
                    'level': old_config.get('file_level', 'DEBUG'),
                    'path': old_config.get('log_path', 'logs')
                }
            }
        }
    }

    # 保存新配置
    import yaml
    with open(new_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(new_config, f, default_flow_style=False, allow_unicode=True)
"""