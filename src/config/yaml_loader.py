"""
日期: 2025-01-12
作者: rrong
描述: YAML配置文件加载器
"""

import sys
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        配置内容字典

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML格式错误
    """
    try:
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f) or {}

        return config_data

    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML format in {config_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")


def save_config(config_data: Dict[str, Any], config_path: str) -> None:
    """保存配置到YAML文件

    Args:
        config_data: 配置数据
        config_path: 保存路径
    """
    try:
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True, indent=2)

    except Exception as e:
        raise RuntimeError(f"Failed to save configuration to {config_path}: {e}")


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """获取嵌套配置值

    Args:
        config: 配置字典
        key_path: 配置路径，用点号分隔，如 'logging.global.level'
        default: 默认值

    Returns:
        配置值
    """
    keys = key_path.split('.')
    current = config

    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """设置嵌套配置值

    Args:
        config: 配置字典
        key_path: 配置路径，用点号分隔
        value: 配置值

    Returns:
        更新后的配置字典
    """
    keys = key_path.split('.')
    current = config

    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]

    current[keys[-1]] = value
    return config