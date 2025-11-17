"""
Optimizer Module - Prompt Extractor

Date: 2025-11-17
Author: backend-developer
Description: Extracts prompts from workflow DSL YAML files.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from loguru import logger

from .exceptions import (
    DSLParseError,
    ExtractionError,
    NodeNotFoundError,
    WorkflowNotFoundError,
)
from .models import Prompt


class PromptExtractor:
    """Extract prompts from workflow DSL files.

    Parses workflow DSL YAML files and extracts all LLM node prompts
    with associated metadata.

    Attributes:
        _logger: Loguru logger instance.

    Example:
        >>> extractor = PromptExtractor()
        >>> prompts = extractor.extract_from_workflow(workflow_dict)
        >>> for prompt in prompts:
        ...     print(f"{prompt.id}: {prompt.text[:50]}...")
    """

    def __init__(self, custom_logger: Optional[Any] = None) -> None:
        """Initialize PromptExtractor.

        Args:
            custom_logger: Optional custom logger instance.
        """
        self._logger = custom_logger or logger.bind(module="optimizer.extractor")

    def extract_from_workflow(
        self, workflow_dict: Dict[str, Any], workflow_id: Optional[str] = None
    ) -> List[Prompt]:
        """Extract all prompts from workflow DSL.

        Parses the workflow dictionary and extracts prompts from all LLM nodes.

        Args:
            workflow_dict: Parsed workflow DSL dictionary.
            workflow_id: Optional workflow identifier (used in prompt IDs).

        Returns:
            List of Prompt objects extracted from the workflow.

        Raises:
            ExtractionError: If extraction fails due to DSL structure issues.

        Example:
            >>> workflow = yaml.safe_load(open("workflow.yaml"))
            >>> prompts = extractor.extract_from_workflow(workflow)
            >>> len(prompts)
            3
        """
        self._logger.info(f"Extracting prompts from workflow: {workflow_id}")

        # Determine workflow_id from dict if not provided
        if workflow_id is None:
            workflow_id = workflow_dict.get("id", "unknown_workflow")

        prompts: List[Prompt] = []

        try:
            # Try to find nodes in different possible locations
            nodes = self._find_nodes(workflow_dict)

            self._logger.debug(f"Found {len(nodes)} nodes in workflow")

            for node in nodes:
                prompt = self.extract_from_node(node, workflow_id)
                if prompt:
                    prompts.append(prompt)

            self._logger.info(
                f"Extracted {len(prompts)} prompts from workflow '{workflow_id}'"
            )
            return prompts

        except Exception as e:
            self._logger.error(f"Failed to extract prompts: {str(e)}")
            raise ExtractionError(
                message=f"Failed to extract prompts from workflow '{workflow_id}'",
                error_code="OPT-EXT-010",
                context={"workflow_id": workflow_id, "error": str(e)},
            )

    def extract_from_node(
        self, node: Dict[str, Any], workflow_id: str
    ) -> Optional[Prompt]:
        """Extract prompt from a single workflow node.

        Only extracts from LLM-type nodes. Returns None for other node types.

        Args:
            node: Node dictionary from workflow DSL.
            workflow_id: Parent workflow identifier.

        Returns:
            Prompt object or None if node is not an LLM node.

        Example:
            >>> node = {"id": "llm_1", "type": "llm", "data": {...}}
            >>> prompt = extractor.extract_from_node(node, "wf_001")
        """
        node_id = node.get("id", "unknown_node")
        node_type = self._detect_node_type(node)

        # Only process LLM nodes
        if node_type != "llm":
            self._logger.debug(f"Skipping non-LLM node: {node_id} (type={node_type})")
            return None

        try:
            # Extract prompt text
            prompt_text = self._extract_prompt_text(node)
            if not prompt_text:
                self._logger.warning(f"No prompt text found in LLM node: {node_id}")
                return None

            # Extract variables
            variables = self._detect_variables(prompt_text)

            # Extract context
            context = self._extract_context(node)

            # Determine role
            role = self._extract_role(node)

            # Create unique ID
            prompt_id = f"{workflow_id}_{node_id}"

            prompt = Prompt(
                id=prompt_id,
                workflow_id=workflow_id,
                node_id=node_id,
                node_type=node_type,
                text=prompt_text,
                role=role,
                variables=variables,
                context=context,
                extracted_at=datetime.now(),
            )

            self._logger.debug(
                f"Extracted prompt from node '{node_id}': "
                f"{len(prompt_text)} chars, {len(variables)} variables"
            )

            return prompt

        except Exception as e:
            self._logger.warning(
                f"Failed to extract prompt from node '{node_id}': {str(e)}"
            )
            return None

    def _find_nodes(self, workflow_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find all nodes in workflow DSL.

        Searches for nodes in various possible locations in the DSL structure.

        Args:
            workflow_dict: Workflow DSL dictionary.

        Returns:
            List of node dictionaries.
        """
        # Try different possible locations for nodes
        possible_paths = [
            ["graph", "nodes"],  # Standard path
            ["nodes"],  # Direct nodes
            ["workflow", "nodes"],  # Wrapped format
            ["app", "workflow", "graph", "nodes"],  # App wrapper format
        ]

        for path in possible_paths:
            nodes = self._get_nested_value(workflow_dict, path)
            if nodes and isinstance(nodes, list):
                return nodes

        self._logger.warning("No nodes found in workflow DSL")
        return []

    def _get_nested_value(self, d: Dict[str, Any], path: List[str]) -> Optional[Any]:
        """Get nested value from dictionary.

        Args:
            d: Dictionary to search.
            path: List of keys representing the path.

        Returns:
            Value at path or None if not found.
        """
        current = d
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    def _detect_node_type(self, node: Dict[str, Any]) -> str:
        """Detect the type of workflow node.

        Args:
            node: Node dictionary.

        Returns:
            Node type string (llm, code, tool, etc.).
        """
        # Direct type field
        if "type" in node:
            return node["type"].lower()

        # Nested in data
        if "data" in node and isinstance(node["data"], dict):
            if "type" in node["data"]:
                return node["data"]["type"].lower()

        # Check for LLM-specific indicators
        if "data" in node and isinstance(node["data"], dict):
            data = node["data"]
            if "model" in data or "prompt_template" in data or "llm" in data:
                return "llm"

        return "unknown"

    def _extract_prompt_text(self, node: Dict[str, Any]) -> Optional[str]:
        """Extract prompt text from LLM node.

        Searches for prompt content in various possible locations.

        Args:
            node: LLM node dictionary.

        Returns:
            Prompt text or None if not found.
        """
        # Possible paths for prompt text
        possible_paths = [
            # Path 1: prompt_template.messages[].text
            ["data", "prompt_template", "messages"],
            # Path 2: llm.prompt_template.messages
            ["llm", "prompt_template", "messages"],
            # Path 3: Direct text field
            ["data", "text"],
            ["data", "prompt"],
            ["text"],
            ["prompt"],
        ]

        data = node.get("data", node)

        # Try message-based paths first
        for path in possible_paths[:2]:
            messages = self._get_nested_value(node, path)
            if messages and isinstance(messages, list):
                return self._concatenate_messages(messages)

        # Try direct text paths
        for path in possible_paths[2:]:
            text = self._get_nested_value(node, path)
            if text and isinstance(text, str):
                return text.strip()

        # Special case: Look for jinja2 template field
        if isinstance(data, dict):
            for key in ["template", "jinja", "template_text"]:
                if key in data and isinstance(data[key], str):
                    return data[key].strip()

        return None

    def _concatenate_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Concatenate message texts into single prompt.

        Args:
            messages: List of message dictionaries.

        Returns:
            Concatenated prompt text.
        """
        texts = []
        for msg in messages:
            if isinstance(msg, dict):
                # Try different text field names
                for field in ["text", "content", "value"]:
                    if field in msg:
                        texts.append(msg[field])
                        break
            elif isinstance(msg, str):
                texts.append(msg)

        return "\n\n".join(texts)

    def _extract_role(self, node: Dict[str, Any]) -> str:
        """Extract message role from node.

        Args:
            node: Node dictionary.

        Returns:
            Role string (system, user, assistant).
        """
        # Check messages for role
        messages = self._get_nested_value(node, ["data", "prompt_template", "messages"])
        if messages and isinstance(messages, list) and len(messages) > 0:
            first_msg = messages[0]
            if isinstance(first_msg, dict) and "role" in first_msg:
                return first_msg["role"]

        # Check direct role field
        data = node.get("data", node)
        if isinstance(data, dict) and "role" in data:
            return data["role"]

        # Default to system
        return "system"

    def _detect_variables(self, text: str) -> List[str]:
        """Detect variable placeholders in prompt text.

        Identifies Jinja2-style variables ({{variable}}).

        Args:
            text: Prompt text.

        Returns:
            List of unique variable names.

        Example:
            >>> extractor._detect_variables("Hello {{name}}, your age is {{age}}")
            ['name', 'age']
        """
        # Pattern for {{variable}} syntax
        pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"

        matches = re.findall(pattern, text)

        # Return unique variables while preserving order
        seen = set()
        unique_vars = []
        for var in matches:
            if var not in seen:
                seen.add(var)
                unique_vars.append(var)

        return unique_vars

    def _extract_context(self, node: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context information from node.

        Args:
            node: Node dictionary.

        Returns:
            Context dictionary with metadata.
        """
        context = {}

        # Extract label
        if "title" in node:
            context["label"] = node["title"]
        elif "label" in node:
            context["label"] = node["label"]
        elif "data" in node and isinstance(node["data"], dict):
            context["label"] = node["data"].get("title", node["data"].get("label", ""))

        # Extract model info
        data = node.get("data", {})
        if isinstance(data, dict):
            if "model" in data:
                context["model"] = data["model"]
            if "provider" in data:
                context["provider"] = data["provider"]
            if "temperature" in data:
                context["temperature"] = data["temperature"]
            if "max_tokens" in data:
                context["max_tokens"] = data["max_tokens"]

        # Extract position if available
        if "position" in node:
            context["position"] = node["position"]

        # Extract dependencies (edges)
        if "source_entities" in node:
            context["dependencies"] = node["source_entities"]

        return context

    def load_dsl_file(self, dsl_path: Path) -> Dict[str, Any]:
        """Load and parse DSL YAML file.

        Args:
            dsl_path: Path to DSL YAML file.

        Returns:
            Parsed DSL dictionary.

        Raises:
            DSLParseError: If file cannot be loaded or parsed.

        Example:
            >>> dsl = extractor.load_dsl_file(Path("workflow.yaml"))
        """
        try:
            with open(dsl_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise DSLParseError(dsl_path=str(dsl_path), reason="File not found")
        except yaml.YAMLError as e:
            raise DSLParseError(
                dsl_path=str(dsl_path), reason=f"YAML parsing error: {str(e)}"
            )
        except Exception as e:
            raise DSLParseError(
                dsl_path=str(dsl_path), reason=f"Unexpected error: {str(e)}"
            )
