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