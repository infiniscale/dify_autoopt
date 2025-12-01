"""
YAML Configuration Module - Configuration Loader

Date: 2025-11-13
Author: Rebirthli
Description: Loads and parses YAML configuration files with environment variable expansion
"""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from ..models import EnvConfig, WorkflowCatalog, TestPlan
from ..utils.exceptions import ConfigurationError


class FileSystemReader:
    """File system reader for YAML files"""

    @staticmethod
    def read_yaml(path: Path) -> Dict[str, Any]:
        """Read YAML file and return parsed dict"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    raise ConfigurationError(f"YAML file must contain a dictionary, got {type(data).__name__}")
                return data
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML format in {path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read {path}: {e}")


class ConfigLoader:
    """Configuration file loader with validation and environment variable expansion"""

    def __init__(self, fs_reader: Optional[FileSystemReader] = None):
        """
        Initialize ConfigLoader

        Args:
            fs_reader: File system reader (default: FileSystemReader)
        """
        self.fs_reader = fs_reader or FileSystemReader()

    def load_env(self, path: Path) -> EnvConfig:
        """
        Load environment configuration

        Args:
            path: Path to env_config.yaml

        Returns:
            EnvConfig instance

        Raises:
            ConfigurationError: If loading or validation fails
        """
        raw_data = self.fs_reader.read_yaml(path)
        expanded_data = self._expand_env_vars(raw_data)

        try:
            cfg = EnvConfig(**expanded_data)
            # 日志记录（尽量不影响流程）
            try:
                from src.utils.logger import get_logger
                get_logger("config.loader").info(
                    "已加载 EnvConfig",
                    extra={
                        "path": str(path.resolve()),
                        "base_url": cfg.dify.base_url,
                        "has_primary_token": bool(cfg.dify.auth.primary_token.get_secret_value()),
                    },
                )
            except Exception:
                pass
            return cfg
        except Exception as e:
            raise ConfigurationError(f"Failed to validate env_config: {e}")

    def load_catalog(self, path: Path) -> WorkflowCatalog:
        """
        Load workflow repository configuration

        Args:
            path: Path to workflow_repository.yaml

        Returns:
            WorkflowCatalog instance

        Raises:
            ConfigurationError: If loading or validation fails
        """
        raw_data = self.fs_reader.read_yaml(path)
        expanded_data = self._expand_env_vars(raw_data)

        try:
            catalog = WorkflowCatalog(**expanded_data)
            try:
                from src.utils.logger import get_logger
                get_logger("config.loader").info(
                    "已加载 WorkflowCatalog",
                    extra={
                        "path": str(path.resolve()),
                        "workflows": len(catalog.workflows),
                    },
                )
            except Exception:
                pass
            return catalog
        except Exception as e:
            raise ConfigurationError(f"Failed to validate workflow_repository: {e}")

    def load_test_plan(self, path: Path) -> TestPlan:
        """
        Load test plan configuration

        Args:
            path: Path to test_plan.yaml

        Returns:
            TestPlan instance

        Raises:
            ConfigurationError: If loading or validation fails
        """
        raw_data = self.fs_reader.read_yaml(path)
        expanded_data = self._expand_env_vars(raw_data)

        try:
            plan = TestPlan(**expanded_data)
            try:
                from src.utils.logger import get_logger
                get_logger("config.loader").info(
                    "已加载 TestPlan",
                    extra={
                        "path": str(path.resolve()),
                        "workflows": len(plan.workflows),
                        "datasets": len(plan.test_data.datasets if plan.test_data else []),
                    },
                )
            except Exception:
                pass
            return plan
        except Exception as e:
            raise ConfigurationError(f"Failed to validate test_plan: {e}")

    def _expand_env_vars(self, data: Any) -> Any:
        """
        Recursively expand environment variable placeholders in the form ${VAR_NAME}

        Args:
            data: Configuration data (can be dict, list, str, or other)

        Returns:
            Data with environment variables expanded
        """
        if isinstance(data, str):
            # Replace ${VAR_NAME} with environment variable value
            def replacer(match):
                var_name = match.group(1)
                value = os.getenv(var_name)
                if value is None:
                    # Keep placeholder if env var not set
                    return match.group(0)
                return value

            return re.sub(r'\$\{(\w+)\}', replacer, data)

        elif isinstance(data, dict):
            return {key: self._expand_env_vars(value) for key, value in data.items()}

        elif isinstance(data, list):
            return [self._expand_env_vars(item) for item in data]

        else:
            # Return primitive types as-is
            return data
