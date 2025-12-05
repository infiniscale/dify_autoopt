"""
Test Cases for PromptExtractor

Date: 2025-11-17
Author: qa-engineer
Description: Unit tests for prompt extraction from workflow DSL
"""

import pytest
import yaml
from pathlib import Path

from src.optimizer.prompt_extractor import PromptExtractor
from src.optimizer.exceptions import DSLParseError, ExtractionError


class TestPromptExtractorBasic:
    """Basic test cases for PromptExtractor."""

    def test_extractor_initialization(self, extractor):
        """Test that PromptExtractor can be initialized."""
        assert extractor is not None
        assert isinstance(extractor, PromptExtractor)

    def test_extract_from_workflow_with_llm_nodes(
        self, extractor, sample_workflow_dict
    ):
        """Test extracting prompts from workflow with LLM nodes."""
        prompts = extractor.extract_from_workflow(
            sample_workflow_dict, workflow_id="test_workflow_001"
        )
        assert len(prompts) == 2  # Should extract 2 LLM nodes
        assert all(p.node_type == "llm" for p in prompts)

    def test_extract_from_workflow_auto_detects_workflow_id(
        self, extractor, sample_workflow_dict
    ):
        """Test that workflow_id is auto-detected from DSL."""
        prompts = extractor.extract_from_workflow(sample_workflow_dict)
        assert len(prompts) == 2
        assert all(p.workflow_id == "test_workflow_001" for p in prompts)

    def test_extract_from_workflow_with_no_llm_nodes(
        self, extractor, sample_workflow_dict_empty_nodes
    ):
        """Test extracting from workflow with no LLM nodes."""
        prompts = extractor.extract_from_workflow(
            sample_workflow_dict_empty_nodes, workflow_id="empty_workflow"
        )
        assert len(prompts) == 0

    def test_extract_from_workflow_ignores_non_llm_nodes(
        self, extractor, sample_workflow_dict
    ):
        """Test that non-LLM nodes are ignored."""
        prompts = extractor.extract_from_workflow(sample_workflow_dict)
        # Should only extract llm_node_1 and llm_node_2, not code_node_1
        prompt_ids = [p.node_id for p in prompts]
        assert "llm_node_1" in prompt_ids
        assert "llm_node_2" in prompt_ids
        assert "code_node_1" not in prompt_ids


class TestPromptExtractionFromNode:
    """Test cases for extract_from_node method."""

    def test_extract_from_llm_node(self, extractor):
        """Test extracting prompt from a valid LLM node."""
        node = {
            "id": "test_llm",
            "type": "llm",
            "data": {
                "prompt_template": {
                    "messages": [{"role": "system", "text": "Test prompt {{var}}"}]
                }
            },
        }
        prompt = extractor.extract_from_node(node, "workflow_test")
        assert prompt is not None
        assert prompt.node_id == "test_llm"
        assert prompt.text == "Test prompt {{var}}"
        assert prompt.variables == ["var"]

    def test_extract_from_non_llm_node_returns_none(self, extractor):
        """Test that non-LLM nodes return None."""
        node = {"id": "code_node", "type": "code", "data": {"code": "print('hi')"}}
        prompt = extractor.extract_from_node(node, "workflow_test")
        assert prompt is None

    def test_extract_from_node_with_direct_text(self, extractor):
        """Test extracting from node with direct text field."""
        node = {"id": "simple_llm", "type": "llm", "data": {"text": "Simple prompt"}}
        prompt = extractor.extract_from_node(node, "workflow_test")
        assert prompt is not None
        assert prompt.text == "Simple prompt"

    def test_extract_from_node_with_prompt_field(self, extractor):
        """Test extracting from node with prompt field."""
        node = {
            "id": "prompt_field",
            "type": "llm",
            "data": {"prompt": "Prompt text {{x}}"},
        }
        prompt = extractor.extract_from_node(node, "workflow_test")
        assert prompt is not None
        assert prompt.text == "Prompt text {{x}}"

    def test_extract_from_node_with_no_prompt_returns_none(self, extractor):
        """Test that LLM node without prompt returns None."""
        node = {"id": "empty_llm", "type": "llm", "data": {}}
        prompt = extractor.extract_from_node(node, "workflow_test")
        assert prompt is None


class TestVariableDetection:
    """Test cases for variable detection in prompts."""

    @pytest.mark.parametrize(
        "text,expected_vars",
        [
            ("Hello {{user}}", ["user"]),
            ("{{var1}} and {{var2}}", ["var1", "var2"]),
            ("No variables here", []),
            ("{{same}} and {{same}}", ["same"]),  # Duplicates removed
            ("{{valid_name}} {{another_var}}", ["valid_name", "another_var"]),
            ("{{_private}} {{public123}}", ["_private", "public123"]),
            ("{{ spaced }}", ["spaced"]),  # Handles spaces
        ],
    )
    def test_detect_variables_various_cases(self, extractor, text, expected_vars):
        """Test variable detection with different inputs."""
        variables = extractor._detect_variables(text)
        assert variables == expected_vars

    def test_detect_variables_preserves_order(self, extractor):
        """Test that variable order is preserved (first occurrence)."""
        text = "{{first}} {{second}} {{third}} {{first}}"
        variables = extractor._detect_variables(text)
        assert variables == ["first", "second", "third"]

    def test_detect_variables_empty_string(self, extractor):
        """Test variable detection on empty string."""
        assert extractor._detect_variables("") == []

    def test_detect_variables_no_match(self, extractor):
        """Test text without valid variable syntax."""
        text = "This has {single brace} and not {{}} empty"
        variables = extractor._detect_variables(text)
        assert variables == []


class TestContextExtraction:
    """Test cases for context extraction from nodes."""

    def test_extract_context_with_label(self, extractor):
        """Test extracting context with label."""
        node = {"id": "test", "title": "Test Node", "type": "llm", "data": {}}
        context = extractor._extract_context(node)
        assert context["label"] == "Test Node"

    def test_extract_context_with_model_info(self, extractor):
        """Test extracting context with model information."""
        node = {
            "id": "test",
            "type": "llm",
            "data": {
                "model": {"provider": "openai", "name": "gpt-4"},
                "temperature": 0.8,
                "max_tokens": 2000,
            },
        }
        context = extractor._extract_context(node)
        assert "model" in context
        assert context["temperature"] == 0.8
        assert context["max_tokens"] == 2000

    def test_extract_context_with_position(self, extractor):
        """Test extracting context with position."""
        node = {
            "id": "test",
            "type": "llm",
            "position": {"x": 100, "y": 200},
            "data": {},
        }
        context = extractor._extract_context(node)
        assert context["position"] == {"x": 100, "y": 200}

    def test_extract_context_empty_node(self, extractor):
        """Test extracting context from minimal node."""
        node = {"id": "test", "type": "llm"}
        context = extractor._extract_context(node)
        assert isinstance(context, dict)
        # Should not crash, may be empty


class TestDSLStructureVariations:
    """Test cases for different DSL structure formats."""

    def test_extract_from_nested_structure(
        self, extractor, sample_workflow_dict_nested
    ):
        """Test extracting from nested DSL structure."""
        prompts = extractor.extract_from_workflow(sample_workflow_dict_nested)
        assert len(prompts) == 1
        assert prompts[0].text == "Nested prompt {{var}}"

    def test_extract_from_direct_nodes_list(
        self, extractor, sample_workflow_dict_direct_nodes
    ):
        """Test extracting from DSL with direct nodes list."""
        prompts = extractor.extract_from_workflow(sample_workflow_dict_direct_nodes)
        assert len(prompts) == 1
        assert prompts[0].text == "Direct prompt {{input}}"

    def test_extract_handles_multiple_message_formats(self, extractor):
        """Test extracting messages in different formats."""
        workflow = {
            "id": "multi_msg",
            "graph": {
                "nodes": [
                    {
                        "id": "multi",
                        "type": "llm",
                        "data": {
                            "prompt_template": {
                                "messages": [
                                    {"role": "system", "text": "System message"},
                                    {"role": "user", "content": "User message"},
                                ]
                            }
                        },
                    }
                ]
            },
        }
        prompts = extractor.extract_from_workflow(workflow)
        assert len(prompts) == 1
        # Should concatenate messages
        assert "System message" in prompts[0].text
        assert "User message" in prompts[0].text


class TestRoleExtraction:
    """Test cases for role extraction."""

    def test_extract_role_from_first_message(self, extractor):
        """Test extracting role from first message."""
        node = {
            "id": "test",
            "type": "llm",
            "data": {
                "prompt_template": {
                    "messages": [{"role": "user", "text": "Hello"}]
                }
            },
        }
        role = extractor._extract_role(node)
        assert role == "user"

    def test_extract_role_default_system(self, extractor):
        """Test default role is system."""
        node = {"id": "test", "type": "llm", "data": {}}
        role = extractor._extract_role(node)
        assert role == "system"

    def test_extract_role_from_data_field(self, extractor):
        """Test extracting role from direct data field."""
        node = {"id": "test", "type": "llm", "data": {"role": "assistant"}}
        role = extractor._extract_role(node)
        assert role == "assistant"


class TestDSLFileLoading:
    """Test cases for loading DSL files."""

    def test_load_dsl_file_success(self, extractor, tmp_path):
        """Test loading valid YAML DSL file."""
        dsl_content = """
id: test_workflow
graph:
  nodes:
    - id: llm_1
      type: llm
      data:
        text: "Test prompt"
"""
        dsl_file = tmp_path / "test.yaml"
        dsl_file.write_text(dsl_content)

        dsl_dict = extractor.load_dsl_file(dsl_file)
        assert dsl_dict["id"] == "test_workflow"
        assert "graph" in dsl_dict

    def test_load_dsl_file_not_found_raises_error(self, extractor, tmp_path):
        """Test loading non-existent file raises DSLParseError."""
        nonexistent = tmp_path / "nonexistent.yaml"
        with pytest.raises(DSLParseError, match="File not found"):
            extractor.load_dsl_file(nonexistent)

    def test_load_dsl_file_invalid_yaml_raises_error(self, extractor, tmp_path):
        """Test loading invalid YAML raises DSLParseError."""
        invalid_file = tmp_path / "invalid.yaml"
        invalid_file.write_text("{ invalid yaml: [unclosed")

        with pytest.raises(DSLParseError, match="YAML parsing error"):
            extractor.load_dsl_file(invalid_file)


class TestNodeTypeDetection:
    """Test cases for node type detection."""

    @pytest.mark.parametrize(
        "node,expected_type",
        [
            ({"type": "llm"}, "llm"),
            ({"type": "LLM"}, "llm"),  # Case insensitive
            ({"type": "code"}, "code"),
            ({"data": {"type": "llm"}}, "llm"),  # Nested type
            (
                {"data": {"model": {"provider": "openai"}}},
                "llm",
            ),  # Inferred from model
            ({"data": {"prompt_template": {}}}, "llm"),  # Inferred from prompt
            ({"type": "tool"}, "tool"),
            ({}, "unknown"),  # No type info
        ],
    )
    def test_detect_node_type_various_cases(self, extractor, node, expected_type):
        """Test node type detection for various node structures."""
        detected_type = extractor._detect_node_type(node)
        assert detected_type == expected_type


class TestPromptTextExtraction:
    """Test cases for prompt text extraction."""

    def test_extract_prompt_text_from_messages(self, extractor):
        """Test extracting text from messages list."""
        node = {
            "data": {
                "prompt_template": {
                    "messages": [
                        {"text": "First message"},
                        {"text": "Second message"},
                    ]
                }
            }
        }
        text = extractor._extract_prompt_text(node)
        assert text == "First message\n\nSecond message"

    def test_extract_prompt_text_from_direct_text(self, extractor):
        """Test extracting from direct text field."""
        node = {"data": {"text": "Direct text here"}}
        text = extractor._extract_prompt_text(node)
        assert text == "Direct text here"

    def test_extract_prompt_text_from_template(self, extractor):
        """Test extracting from template field."""
        node = {"data": {"template": "Template text {{var}}"}}
        text = extractor._extract_prompt_text(node)
        assert text == "Template text {{var}}"

    def test_extract_prompt_text_not_found_returns_none(self, extractor):
        """Test that missing prompt returns None."""
        node = {"data": {}}
        text = extractor._extract_prompt_text(node)
        assert text is None

    def test_extract_prompt_text_strips_whitespace(self, extractor):
        """Test that extracted text is stripped."""
        node = {"data": {"text": "  \n  Text with spaces  \n  "}}
        text = extractor._extract_prompt_text(node)
        assert text == "Text with spaces"


class TestMessageConcatenation:
    """Test cases for message concatenation."""

    def test_concatenate_messages_with_text_field(self, extractor):
        """Test concatenating messages with text field."""
        messages = [{"text": "First"}, {"text": "Second"}]
        result = extractor._concatenate_messages(messages)
        assert result == "First\n\nSecond"

    def test_concatenate_messages_with_content_field(self, extractor):
        """Test concatenating messages with content field."""
        messages = [{"content": "A"}, {"content": "B"}]
        result = extractor._concatenate_messages(messages)
        assert result == "A\n\nB"

    def test_concatenate_messages_with_value_field(self, extractor):
        """Test concatenating messages with value field."""
        messages = [{"value": "X"}, {"value": "Y"}]
        result = extractor._concatenate_messages(messages)
        assert result == "X\n\nY"

    def test_concatenate_messages_string_list(self, extractor):
        """Test concatenating list of strings."""
        messages = ["String one", "String two"]
        result = extractor._concatenate_messages(messages)
        assert result == "String one\n\nString two"

    def test_concatenate_messages_empty_list(self, extractor):
        """Test concatenating empty list."""
        result = extractor._concatenate_messages([])
        assert result == ""

    def test_concatenate_messages_mixed_formats(self, extractor):
        """Test concatenating messages with mixed formats."""
        messages = [{"text": "A"}, "B", {"content": "C"}]
        result = extractor._concatenate_messages(messages)
        assert "A" in result
        assert "B" in result
        assert "C" in result


class TestNestedValueRetrieval:
    """Test cases for _get_nested_value helper."""

    def test_get_nested_value_success(self, extractor):
        """Test retrieving nested value successfully."""
        d = {"a": {"b": {"c": "value"}}}
        result = extractor._get_nested_value(d, ["a", "b", "c"])
        assert result == "value"

    def test_get_nested_value_not_found(self, extractor):
        """Test retrieving non-existent path returns None."""
        d = {"a": {"b": "value"}}
        result = extractor._get_nested_value(d, ["a", "x", "y"])
        assert result is None

    def test_get_nested_value_empty_path(self, extractor):
        """Test retrieving with empty path."""
        d = {"key": "value"}
        result = extractor._get_nested_value(d, [])
        assert result == d

    def test_get_nested_value_single_level(self, extractor):
        """Test retrieving single-level value."""
        d = {"key": "value"}
        result = extractor._get_nested_value(d, ["key"])
        assert result == "value"


class TestExtractionErrorHandling:
    """Test cases for error handling."""

    def test_extract_from_node_handles_exceptions_gracefully(self, extractor):
        """Test that node extraction exceptions are handled."""
        # Malformed node that might cause errors
        node = {"id": "bad", "type": "llm", "data": None}
        # Should return None instead of raising
        prompt = extractor.extract_from_node(node, "workflow_test")
        # Result may be None or raise, but shouldn't crash
        assert prompt is None  # Malformed node returns None


class TestCompleteExtractionWorkflow:
    """End-to-end test cases for complete extraction workflow."""

    def test_extract_prompts_with_all_metadata(self, extractor):
        """Test extracting prompts with all metadata preserved."""
        workflow = {
            "id": "complete_test",
            "graph": {
                "nodes": [
                    {
                        "id": "rich_node",
                        "type": "llm",
                        "title": "Rich LLM Node",
                        "data": {
                            "prompt_template": {
                                "messages": [
                                    {
                                        "role": "system",
                                        "text": "Complex prompt with {{var1}} and {{var2}}",
                                    }
                                ]
                            },
                            "model": {"provider": "openai", "name": "gpt-4"},
                            "temperature": 0.9,
                            "max_tokens": 1500,
                        },
                        "position": {"x": 50, "y": 100},
                    }
                ]
            },
        }

        prompts = extractor.extract_from_workflow(workflow)
        assert len(prompts) == 1

        p = prompts[0]
        assert p.id == "complete_test_rich_node"
        assert p.workflow_id == "complete_test"
        assert p.node_id == "rich_node"
        assert p.node_type == "llm"
        assert "var1" in p.variables
        assert "var2" in p.variables
        assert p.context["label"] == "Rich LLM Node"
        assert p.context["temperature"] == 0.9
        assert p.role == "system"

    def test_extract_multiple_prompts_preserves_order(self, extractor):
        """Test that extracting multiple prompts preserves node order."""
        workflow = {
            "id": "multi",
            "graph": {
                "nodes": [
                    {"id": "llm_1", "type": "llm", "data": {"text": "First"}},
                    {"id": "code", "type": "code"},
                    {"id": "llm_2", "type": "llm", "data": {"text": "Second"}},
                    {"id": "llm_3", "type": "llm", "data": {"text": "Third"}},
                ]
            },
        }

        prompts = extractor.extract_from_workflow(workflow)
        assert len(prompts) == 3
        assert [p.node_id for p in prompts] == ["llm_1", "llm_2", "llm_3"]
