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