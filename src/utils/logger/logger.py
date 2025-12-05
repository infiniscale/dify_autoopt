"""
基于loguru的高性能日志模块 - 修复版本
日期：2025-01-12
作者：rrong
描述：简化版本，专注于基本功能实现
"""

import os
import sys
import json
import threading
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager, asynccontextmanager

import loguru
from loguru import logger as loguru_logger


class LoggingException(Exception):
    """日志模块异常"""
    pass


class SimpleLogManager:
    """简化的日志管理器"""

    def __init__(self):
        self._configured = False
        self._config = {}
        self._lock = threading.Lock()

    async def initialize(self, config_path: Optional[str] = None) -> None:
        """初始化日志系统"""
        with self._lock:
            if self._configured:
                return

            try:
                # 加载配置
                self._config = self._load_config(config_path)

                # 配置loguru
                await self._setup_loguru()

                self._configured = True

            except Exception as e:
                raise LoggingException(f"Failed to initialize logging: {e}")

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = {
            "level": "INFO",
            "format": "simple",  # simple or structured
            "console_enabled": True,
            "console_level": "INFO",
            "file_enabled": True,
            "file_level": "DEBUG",
            "file_path": "logs",
            "rotation": "1 day",
            "retention": "30 days",
            "compression": "zip",
            "include_colors": True,
            "date_format": "%Y-%m-%d %H:%M:%S"
        }

        if config_path and os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                if 'logging' in config:
                    logging_config = config['logging']
                    global_config = logging_config.get('global', {})
                    outputs_config = logging_config.get('outputs', {})

                    # 更新配置
                    default_config.update({
                        "level": global_config.get('level', "INFO"),
                        "format": global_config.get('format', 'simple'),
                        "include_colors": outputs_config.get('console', {}).get('include_colors', True),
                        "console_enabled": outputs_config.get('console', {}).get('enabled', True),
                        "console_level": outputs_config.get('console', {}).get('level', "INFO"),
                        "file_enabled": outputs_config.get('file', {}).get('enabled', True),
                        "file_level": outputs_config.get('file', {}).get('level', "DEBUG"),
                        "file_path": outputs_config.get('file', {}).get('path', "logs"),
                        "rotation": outputs_config.get('file', {}).get('rotation', "1 day"),
                        "retention": outputs_config.get('file', {}).get('retention', "30 days"),
                        "compression": outputs_config.get('file', {}).get('compression', "zip")
                    })

            except Exception as e:
                print(f"Warning: Failed to load config from {config_path}: {e}")

        return default_config

    async def _setup_loguru(self) -> None:
        """配置loguru logger"""
        # 移除所有默认处理器
        loguru_logger.remove()

        # 创建日志目录
        if self._config["file_enabled"]:
            log_path = Path(self._config["file_path"])
            log_path.mkdir(parents=True, exist_ok=True)

        # 定义格式化器
        def simple_formatter(record: dict) -> str:
            """简单格式化器"""
            return (
                f"{record['time'].strftime(self._config['date_format'])} | "
                f"{record['level'].name} | "
                f"{record['name']} | "
                f"{record['message']}"
            )

        def structured_formatter(record: dict) -> str:
            """结构化格式化器"""
            log_data = {
                "timestamp": record["time"].strftime(self._config["date_format"]),
                "level": record["level"].name,
                "logger": record["name"],
                "message": record["message"],
                "module": record.get("module", ""),
                "function": record.get("function", ""),
                "line": record.get("line", 0),
                "thread": threading.current_thread().name
            }

            # 添加额外信息
            extra = record.get("extra", {})
            if extra:
                log_data["extra"] = extra

            # 添加异常信息
            if "exception" in record and record["exception"] is not None:
                log_data["exception"] = {
                    "type": record["exception"].type,
                    "value": str(record["exception"].value),
                    "traceback": record["exception"].traceback
                }

            return json.dumps(log_data, ensure_ascii=False, default=str)

        # 选择格式化器
        formatter = simple_formatter if self._config["format"] == "simple" else structured_formatter

        # 控制台输出
        if self._config["console_enabled"]:
            loguru_logger.add(
                sys.stdout,
                level=self._config["console_level"],
                format=formatter,
                colorize=self._config["include_colors"],
                catch=True
            )

        # 文件输出
        if self._config["file_enabled"]:
            log_date = datetime.now().strftime("%Y-%m-%d")
            log_file = Path(self._config["file_path"]) / f"dify_autoopt_{log_date}.log"

            loguru_logger.add(
                str(log_file),
                level=self._config["file_level"],
                format=formatter,
                rotation=self._config["rotation"],
                retention=self._config["retention"],
                compression=self._config["compression"],
                catch=True
            )

    def is_configured(self) -> bool:
        """检查是否已配置"""
        return self._configured

    async def shutdown(self) -> None:
        """关闭日志系统"""
        with self._lock:
            if self._configured:
                loguru_logger.remove()
                self._configured = False


# 全局日志管理器实例
_log_manager = SimpleLogManager()


async def setup_logging(config_path: Optional[str] = None) -> SimpleLogManager:
    """初始化日志系统"""
    await _log_manager.initialize(config_path)
    return _log_manager


def get_logger(name: str) -> loguru_logger:
    """获取指定名称的logger"""
    if not _log_manager.is_configured():
        raise LoggingException("Logging system not initialized. Call setup_logging() first.")

    return loguru_logger.bind(name=name)


@contextmanager
def log_context(**context):
    """日志上下文管理器"""
    logger = loguru_logger.bind(context=context)
    try:
        yield logger
    finally:
        pass


def log_performance(operation_name: str):
    """性能监控装饰器"""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start_time = datetime.now()
                try:
                    result = await func(*args, **kwargs)
                    duration = (datetime.now() - start_time).total_seconds()
                    loguru_logger.info(f"Operation '{operation_name}' completed in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = (datetime.now() - start_time).total_seconds()
                    loguru_logger.error(f"Operation '{operation_name}' failed after {duration:.3f}s: {e}")
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                import time
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    loguru_logger.info(f"Operation '{operation_name}' completed in {duration:.3f}s")
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    loguru_logger.error(f"Operation '{operation_name}' failed after {duration:.3f}s: {e}")
                    raise
            return sync_wrapper
    return decorator


# 导出的模块接口
__all__ = [
    'setup_logging',
    'get_logger',
    'log_context',
    'log_performance',
    'LoggingException',
    '_log_manager'
]