"""
日期: 2025-01-12
作者: rrong
描述: 自定义异常类
"""

class ConfigurationException(Exception):
    """配置文件相关异常"""
    pass


class LoggingException(Exception):
    """日志系统相关异常"""
    pass


class WorkflowException(Exception):
    """工作流相关异常"""
    pass


class DataProcessingException(Exception):
    """数据处理相关异常"""
    pass


class CollectorException(Exception):
    """结果采集模块基础异常"""
    pass


class DataValidationException(CollectorException):
    """数据验证异常 - 当接收到的测试结果数据格式不正确时抛出"""
    pass


class ExportException(CollectorException):
    """数据导出异常 - 当Excel导出失败时抛出"""
    pass


class ClassificationException(CollectorException):
    """结果分类异常 - 当性能分类逻辑执行失败时抛出"""
    pass


# ============================================================================
# Executor Module Exceptions
# ============================================================================


class ExecutorException(Exception):
    """执行器模块基础异常 - 所有执行器相关异常的基类"""
    pass


class TaskExecutionException(ExecutorException):
    """任务执行异常 - 当单个任务执行失败时抛出（业务逻辑错误）

    Attributes:
        message: 错误消息
        task_id: 任务ID
        attempt: 尝试次数
    """

    def __init__(self, message: str, task_id: str = None, attempt: int = None):
        super().__init__(message)
        self.message = message
        self.task_id = task_id
        self.attempt = attempt


class TaskTimeoutException(ExecutorException):
    """任务超时异常 - 当任务执行超过指定超时时间时抛出

    Attributes:
        message: 错误消息
        task_id: 任务ID
        timeout_seconds: 超时限制（秒）
        elapsed_seconds: 实际耗时（秒）
    """

    def __init__(self, message: str, task_id: str = None, timeout_seconds: float = None, elapsed_seconds: float = None):
        super().__init__(message)
        self.message = message
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class SchedulerException(ExecutorException):
    """调度器异常 - 当任务调度或队列管理出现问题时抛出"""
    pass


class RateLimitException(ExecutorException):
    """速率限制异常 - 当请求超出速率限制配置时抛出"""
    pass
