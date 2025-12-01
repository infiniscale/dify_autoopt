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
        self._global_context: Dict[str, Any] = {}

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
        """加载配置文件

        支持两种配置风格：
        1) 高级配置（config/logging_config.yaml）：logging.global/outputs/...
        2) 简化配置（config/config.yaml 下的 logging 简化字段）
        """
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

        # 自动探测配置路径：优先 logging_config.yaml，其次 config.yaml
        candidate_paths = []
        if config_path:
            candidate_paths.append(config_path)
        else:
            candidate_paths.extend([
                os.path.join("config", "logging_config.yaml"),
                os.path.join("config", "config.yaml"),
            ])

        selected_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                selected_path = p
                break

        if selected_path:
            try:
                import yaml
                with open(selected_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)

                if isinstance(config, dict) and 'logging' in config:
                    logging_config = config['logging'] or {}

                    # 风格 1：高级配置（有 global/outputs）
                    if isinstance(logging_config, dict) and (
                        'global' in logging_config or 'outputs' in logging_config
                    ):
                        global_config = logging_config.get('global', {})
                        outputs_config = logging_config.get('outputs', {})
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

                    # 风格 2：简化配置（config.yaml 下的扁平字段）
                    else:
                        default_config.update({
                            "level": logging_config.get('level', default_config['level']),
                            "format": logging_config.get('format', default_config['format']),
                            "console_enabled": logging_config.get('console_enabled', default_config['console_enabled']),
                            "file_enabled": logging_config.get('file_enabled', default_config['file_enabled']),
                        })

                        # 可选字段
                        if 'file_path' in logging_config:
                            default_config['file_path'] = logging_config['file_path']
                        if 'console_level' in logging_config:
                            default_config['console_level'] = logging_config['console_level']
                        if 'file_level' in logging_config:
                            default_config['file_level'] = logging_config['file_level']
                        if 'rotation' in logging_config:
                            default_config['rotation'] = logging_config['rotation']
                        if 'retention' in logging_config:
                            default_config['retention'] = logging_config['retention']
                        if 'compression' in logging_config:
                            default_config['compression'] = logging_config['compression']

                # 记录生效的配置源（延后在 _setup_loguru 后输出）
                self._config_source = selected_path

            except Exception as e:
                print(f"Warning: Failed to load logging config from {selected_path}: {e}")

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
            logger_name = record.get('extra', {}).get('name', record['name'])
            return (
                f"{record['time'].strftime(self._config['date_format'])} | "
                f"{record['level'].name} | "
                f"{logger_name} | "
                f"{record['message']}\n"
            )

        def structured_formatter(record: dict) -> str:
            """结构化格式化器"""
            logger_name = record.get('extra', {}).get('name', record['name'])
            log_data = {
                "timestamp": record["time"].strftime(self._config["date_format"]),
                "level": record["level"].name,
                "logger": logger_name,
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
            return json.dumps(log_data, ensure_ascii=False, default=str) + "\n"

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

        # 输出初始化信息
        try:
            src = getattr(self, "_config_source", None)
            loguru_logger.info(
                "日志系统初始化完成",
                extra={
                    "config_source": src or "<defaults>",
                    "level": self._config.get("level"),
                    "format": self._config.get("format"),
                    "console_enabled": self._config.get("console_enabled"),
                    "file_enabled": self._config.get("file_enabled"),
                    "file_path": str(Path(self._config.get("file_path", "logs")).resolve()),
                },
            )
        except Exception:
            pass

    def is_configured(self) -> bool:
        """检查是否已配置"""
        return self._configured

    async def shutdown(self) -> None:
        """关闭日志系统"""
        with self._lock:
            if self._configured:
                loguru_logger.remove()
                self._configured = False

    # ---- 额外的全局上下文与统计接口（最小实现） ----
    def set_global_context(self, **context: Any) -> None:
        """设置全局上下文（最小实现，用于附加到结构化日志extra中）"""
        with self._lock:
            self._global_context.update(context)

    def get_global_context(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._global_context)

    def get_stats(self) -> Dict[str, Any]:
        """返回日志系统的基本统计信息（占位实现）。"""
        return {
            "configured": self._configured,
            "format": self._config.get("format"),
            "console_enabled": self._config.get("console_enabled"),
            "file_enabled": self._config.get("file_enabled"),
        }


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

    # 绑定logger名称与全局上下文（如已设置）
    bound = loguru_logger.bind(name=name)
    global_ctx = _log_manager.get_global_context()
    if global_ctx:
        bound = bound.bind(**global_ctx)
    return bound


@contextmanager
def log_context(**context):
    """日志上下文管理器：在上下文内对日志注入结构化字段（基于loguru.contextualize）。"""
    try:
        with loguru_logger.contextualize(**context):
            yield
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


def log_exception(level: str = "ERROR", reraise: bool = True):
    """异常捕获装饰器（最小实现）。

    Args:
        level: 记录级别（DEBUG/INFO/WARNING/ERROR/CRITICAL）。
        reraise: 是否在记录后重新抛出异常。
    """

    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    loguru_logger.opt(exception=True).log(level, f"Exception in {func.__name__}: {e}")
                    if reraise:
                        raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    loguru_logger.opt(exception=True).log(level, f"Exception in {func.__name__}: {e}")
                    if reraise:
                        raise
            return sync_wrapper

    return decorator


@contextmanager
def log_workflow_trace(workflow_id: str, operation: str, logger: Optional[Any] = None):
    """工作流阶段跟踪上下文管理器（最小实现）。

    在进入/退出时记录开始与结束，并包含耗时信息。
    """
    _logger = logger or loguru_logger
    from time import perf_counter

    start = perf_counter()
    _logger.info(
        "工作流阶段开始",
        extra={
            "workflow_id": workflow_id,
            "operation": operation,
            "status": "start",
        },
    )
    try:
        yield
        duration = perf_counter() - start
        _logger.info(
            "工作流阶段完成",
            extra={
                "workflow_id": workflow_id,
                "operation": operation,
                "status": "completed",
                "duration_seconds": round(duration, 6),
            },
        )
    except Exception:
        duration = perf_counter() - start
        _logger.error(
            "工作流阶段失败",
            extra={
                "workflow_id": workflow_id,
                "operation": operation,
                "status": "error",
                "duration_seconds": round(duration, 6),
            },
            exc_info=True,
        )
        raise


# 导出的模块接口
__all__ = [
    'setup_logging',
    'get_logger',
    'log_context',
    'log_performance',
    'log_exception',
    'log_workflow_trace',
    'LoggingException',
    '_log_manager'
]
