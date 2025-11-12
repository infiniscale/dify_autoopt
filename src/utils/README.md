# 通用工具模块（utils）

本目录包含项目的通用工具，当前重点为“日志管理”。

## 日志调用逻辑（强烈建议遵循）

1) 在程序最开始初始化日志（一次即可）
```python
from src.utils.logger import setup_logging
import asyncio

asyncio.run(setup_logging("config/logging_config.yaml"))  # 未传参也可使用默认配置
```

2) 通过工厂方法获取模块级 logger
```python
from src.utils.logger import get_logger

logger = get_logger("main")
logger.info("应用启动")
```

3) 可选：在代码块中绑定上下文或记录性能
```python
from src.utils.logger import log_context, log_performance

with log_context(request_id="req_123", user_id="u_01"):
    logger.info("带上下文的日志")

@log_performance("数据处理")
def process():
    ...
```

4) 文件输出位置与格式
- 默认输出到 `logs/`，文件名形如 `dify_autoopt_YYYY-MM-DD.log`
- 控制台与文件的级别/格式由 `config/logging_config.yaml` 中的 `logging.outputs.*` 与 `logging.global` 控制

## 配置要点（config/logging_config.yaml）
- `logging.global.level`: 日志级别（INFO/DEBUG/...）
- `logging.global.format`: `simple` 或 `structured`
- `logging.outputs.console.enabled`: 是否启用控制台输出
- `logging.outputs.file.enabled`: 是否写入文件；`path` 指定目录

示例片段
```yaml
logging:
  global:
    level: INFO
    format: simple
  outputs:
    console:
      enabled: true
    file:
      enabled: true
      path: logs
      level: DEBUG
```

## 常见问题
- 未先调用 `setup_logging()` 就使用 `get_logger()` 会抛出 `LoggingException`。
- 测试场景参考：`src/test/test_logger_basic.py` 展示了初始化与文件写入的最小用例。

## 目录说明
- `logger/` 日志实现与对外 API（`setup_logging`、`get_logger` 等）
- `exceptions.py` 通用异常定义（如有需要可扩展）

        if self.auth:
            kwargs['auth'] = self.auth

        for attempt in range(self.max_retries + 1):
            try:
                async with ClientSession(timeout=self.timeout) as session:
                    async with session.request(method, url, **kwargs) as response:
                        response.raise_for_status()
                        return await self._process_response(response)
            except Exception as e:
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                    continue
                else:
                    raise
```

### 请求拦截器
```python
class RequestInterceptor:
    def __init__(self):
        self.before_request_hooks = []
        self.after_request_hooks = []

    def add_before_request_hook(self, hook):
        self.before_request_hooks.append(hook)

    def add_after_request_hook(self, hook):
        self.after_request_hooks.append(hook)

    async def before_request(self, method: str, url: str, **kwargs):
        # 执行前置钩子
        for hook in self.before_request_hooks:
            await hook(method, url, **kwargs)

        # 添加通用头信息
        headers = kwargs.get('headers', {})
        headers.update({
            'User-Agent': 'Dify-AutoOpt/1.0',
            'X-Request-ID': generate_request_id(),
        })
        kwargs['headers'] = headers

    async def after_request(self, response, **kwargs):
        # 执行后置钩子
        for hook in self.after_request_hooks:
            await hook(response, **kwargs)
```

### 连接池管理
```python
class ConnectionPoolManager:
    def __init__(self, pool_size: int = 100, max_size: int = 200):
        self.pool_size = pool_size
        self.max_size = max_size
        self._session = None

    async def get_session(self) -> ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.pool_size,
                limit_per_host=self.max_size,
                keepalive_timeout=30,
                enable_cleanup_closed=True
            )
            self._session = ClientSession(connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
```

## 异常定义

### 基础异常类
```python
class DifyAutoOptException(Exception):
    """项目基础异常类"""
    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.cause = cause
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "cause": str(self.cause) if self.cause else None
        }

    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {super().__str__()}"
        return super().__str__()
```

### 具体异常类型
```python
# 认证相关异常
class DifyAuthException(DifyAutoOptException):
    """Dify认证异常"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="AUTH_ERROR", **kwargs)

class SessionExpiredException(DifyAuthException):
    """会话过期异常"""
    def __init__(self, **kwargs):
        super().__init__("会话已过期，请重新登录", **kwargs)

class InvalidCredentialsException(DifyAuthException):
    """无效凭据异常"""
    def __init__(self, **kwargs):
        super().__init__("用户名或密码错误", **kwargs)

# 工作流相关异常
class WorkflowException(DifyAutoOptException):
    """工作流异常"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="WORKFLOW_ERROR", **kwargs)

class WorkflowNotFoundException(WorkflowException):
    """工作流未找到异常"""
    def __init__(self, workflow_id: str, **kwargs):
        super().__init__(f"工作流未找到: {workflow_id}", **kwargs)

class WorkflowExecutionException(WorkflowException):
    """工作流执行异常"""
    def __init__(self, workflow_id: str, error_details: str, **kwargs):
        super().__init__(
            f"工作流执行失败: {workflow_id}",
            details={"workflow_id": workflow_id, "error": error_details},
            **kwargs
        )

# 配置相关异常
class ConfigurationException(DifyAutoOptException):
    """配置异常"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)

class MissingConfigurationException(ConfigurationException):
    """配置缺失异常"""
    def __init__(self, config_key: str, **kwargs):
        super().__init__(f"配置缺失: {config_key}", **kwargs)

class InvalidConfigurationException(ConfigurationException):
    """无效配置异常"""
    def __init__(self, config_key: str, value: Any, **kwargs):
        super().__init__(
            f"无效配置: {config_key} = {value}",
            details={"config_key": config_key, "value": value},
            **kwargs
        )

# 网络相关异常
class NetworkException(DifyAutoOptException):
    """网络异常"""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)

class ConnectionTimeoutException(NetworkException):
    """连接超时异常"""
    def __init__(self, url: str, timeout: int, **kwargs):
        super().__init__(
            f"连接超时: {url} (timeout={timeout}s)",
            details={"url": url, "timeout": timeout},
            **kwargs
        )
```

## 工具函数

### 加密解密工具
```python
import hashlib
import base64
from cryptography.fernet import Fernet

class CryptoUtils:
    @staticmethod
    def generate_key() -> str:
        return Fernet.generate_key().decode()

    @staticmethod
    def encrypt_data(data: str, key: str) -> str:
        f = Fernet(key.encode())
        encrypted = f.encrypt(data.encode())
        return base64.b64encode(encrypted).decode()

    @staticmethod
    def decrypt_data(encrypted_data: str, key: str) -> str:
        f = Fernet(key.encode())
        decoded = base64.b64decode(encrypted_data.encode())
        decrypted = f.decrypt(decoded)
        return decrypted.decode()

    @staticmethod
    def hash_password(password: str, salt: str = None) -> tuple:
        if salt is None:
            salt = os.urandom(32)

        pwd_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000
        )
        return pwd_hash, salt
```

### 日期时间工具
```python
from datetime import datetime, timedelta
import pytz

class DateTimeUtils:
    @staticmethod
    def now_utc() -> datetime:
        """获取当前UTC时间"""
        return datetime.now(pytz.UTC)

    @staticmethod
    def format_datetime(dt: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """格式化日期时间"""
        return dt.strftime(format_str)

    @staticmethod
    def parse_datetime(date_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
        """解析日期时间字符串"""
        return datetime.strptime(date_str, format_str)

    @staticmethod
    def time_range(start_time: datetime, end_time: datetime, step: timedelta = timedelta(hours=1)):
        """生成时间范围生成器"""
        current = start_time
        while current < end_time:
            yield current
            current += step
```

### 字符串工具
```python
import re
from typing import List, Optional

class StringUtils:
    @staticmethod
    def camel_to_snake(name: str) -> str:
        """驼峰转蛇形"""
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    @staticmethod
    def snake_to_camel(name: str) -> str:
        """蛇形转驼峰"""
        components = name.split('_')
        return ''.join(word.capitalize() for word in components)

    @staticmethod
    def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
        """截断文本"""
        if len(text) <= max_length:
            return text
        return text[:max_length - len(suffix)] + suffix

    @staticmethod
    def extract_variables(text: str, pattern: str = r'\{\{(\w+)\}\}') -> List[str]:
        """提取文本中的变量"""
        return re.findall(pattern, text)

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """清理文件名"""
        # 移除或替换非法字符
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # 移除多余的点和空格
        sanitized = re.sub(r'\.+', '.', sanitized).strip()
        return sanitized
```

### 验证工具
```python
import re
from typing import Any, List

class ValidationUtils:
    @staticmethod
    def is_email(email: str) -> bool:
        """验证邮箱格式"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None

    @staticmethod
    def is_url(url: str) -> bool:
        """验证URL格式"""
        pattern = r'^https?://(?:[-\w.])+(?:[:0-9]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
        return re.match(pattern, url) is not None

    @staticmethod
    def is_json_schema_valid(data: Any, schema: dict) -> bool:
        """验证JSON Schema"""
        from jsonschema import validate, ValidationError
        try:
            validate(instance=data, schema=schema)
            return True
        except ValidationError:
            return False

    @staticmethod
    def validate_file_path(path: str, allow_symlinks: bool = False) -> bool:
        """验证文件路径"""
        try:
            # 安全性检查
            if '..' in path:
                return False

            abs_path = os.path.abspath(path)
            if not allow_symlinks and os.path.islink(abs_path):
                return False

            return True
        except (OSError, ValueError):
            return False
```

## 配置参数

```yaml
utils:
  # 日志配置
  logging:
    level: "INFO"
    format: "structured"
    file_path: "logs/app.log"
    max_file_size: "10MB"
    backup_count: 5
    console_output: True
    structured_fields: ["module", "version", "workflow_id", "execution_time"]

  # HTTP客户端配置
  http_client:
    default_timeout: 30
    max_retries: 3
    retry_delay: 1.0
    pool_size: 100
    max_pool_size: 200
    keepalive_timeout: 30
    ssl_verify: True
    compression: True

  # 异常处理配置
  exceptions:
    include_traceback: True
    max_error_details_length: 1000
    error_notification_webhook: null
    auto_retry_on_exception: True
    max_auto_retry_attempts: 3

  # 工具配置
  crypto:
    default_key_rotation_interval: 86400  # 24 hours
    encryption_algorithm: "AES-256-GCM"
    hash_algorithm: "SHA-256"
