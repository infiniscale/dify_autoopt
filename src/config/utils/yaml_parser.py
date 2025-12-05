"""
YAML Configuration Module - YAML Parser Utility

Date: 2025-11-13
Author: Rebirthli
Description: Utility for parsing and manipulating YAML trees (especially DSL files)
"""

import yaml
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..utils.exceptions import DSLParseError


class YamlParser:
    """YAML parser for DSL manipulation"""

    def load(self, yaml_text: str) -> Dict[str, Any]:
        """
        Parse YAML text into Python dict

        Args:
            yaml_text: YAML content as string

        Returns:
            Parsed dictionary

        Raises:
            DSLParseError: If YAML is invalid
        """
        try:
            data = yaml.safe_load(yaml_text)
            if not isinstance(data, dict):
                raise DSLParseError(f"Expected dict, got {type(data).__name__}")
            return data
        except yaml.YAMLError as e:
            raise DSLParseError(f"Invalid YAML: {e}")

    def dump(self, data: Dict[str, Any]) -> str:
        """
        Convert Python dict to YAML text

        Args:
            data: Python dictionary

        Returns:
            YAML formatted string
        """
        return yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            indent=2,
            sort_keys=False
        )

    def get_node_by_path(self, tree: Dict[str, Any], path: str) -> Optional[Dict[str, Any]]:
        """
        Get node by JSON pointer path (e.g., '/graph/nodes/0')

        Args:
            tree: YAML tree (dict)
            path: JSON pointer path

        Returns:
            Node dict or None if not found
        """
        if not path.startswith('/'):
            raise ValueError(f"Path must start with '/', got: {path}")

        parts = [p for p in path.split('/') if p]  # Remove empty strings
        current = tree

        try:
            for part in parts:
                # Try as array index first
                if isinstance(current, list):
                    current = current[int(part)]
                else:
                    current = current[part]
            return current
        except (KeyError, IndexError, ValueError, TypeError):
            return None

    def set_field_value(self, node: Dict[str, Any], field_path: str, value: Any) -> None:
        """
        Set a field value in a node (supports nested paths like 'config.prompt')

        Args:
            node: Node dictionary
            field_path: Field path (dot-separated for nested)
            value: New value
        """
        parts = field_path.split('.')
        current = node

        # Navigate to parent
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set final value
        current[parts[-1]] = value

    def get_field_value(self, node: Dict[str, Any], field_path: str) -> Optional[Any]:
        """
        Get a field value from a node (supports nested paths)

        Args:
            node: Node dictionary
            field_path: Field path (dot-separated)

        Returns:
            Field value or None if not found
        """
        parts = field_path.split('.')
        current = node

        try:
            for part in parts:
                current = current[part]
            return current
        except (KeyError, TypeError):
            return None
