"""
Test suite for CRITICAL #5: Variable Extraction Missing Dify Syntax Fix

Tests that VariableExtractor supports all Dify variable syntax variants.
"""

import pytest

from src.optimizer.utils.variable_extractor import VariableExtractor
from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.models import Prompt
from datetime import datetime


class TestVariableExtraction:
    """Test suite for comprehensive variable extraction."""

    def test_variable_extraction_standard(self):
        """Test standard {{variable}} syntax."""
        text = "Process {{name}} and {{age}}"
        vars = VariableExtractor.extract(text)

        assert "name" in vars
        assert "age" in vars
        assert len(vars) == 2

    def test_variable_extraction_nested(self):
        """Test nested {{obj.field}} syntax."""
        text = "User: {{user.email}} Query: {{sys.query}}"
        vars = VariableExtractor.extract(text)

        assert "user.email" in vars
        assert "sys.query" in vars

    def test_variable_extraction_context(self):
        """Test context {{#context#}} syntax."""
        text = "Context: {{#context.data#}} {{#workflow.id#}}"
        vars = VariableExtractor.extract(text)

        assert "context.data" in vars
        assert "workflow.id" in vars

    def test_variable_extraction_user(self):
        """Test user {{@user}} syntax."""
        text = "Current: {{@current.user}} Owner: {{@owner.name}}"
        vars = VariableExtractor.extract(text)

        assert "current.user" in vars
        assert "owner.name" in vars

    def test_variable_extraction_system(self):
        """Test system {{$sys}} syntax."""
        text = "Workflow: {{$workflow.id}} Model: {{$config.model}}"
        vars = VariableExtractor.extract(text)

        assert "workflow.id" in vars
        assert "config.model" in vars

    def test_variable_extraction_comprehensive(self):
        """Test all Dify variable syntax variants together."""
        text = """
        Standard: {{name}} {{age}}
        Nested: {{user.email}} {{sys.query}}
        Context: {{#context.data#}}
        User: {{@current.user}}
        System: {{$workflow.id}}
        """

        vars = VariableExtractor.extract(text)

        assert "name" in vars
        assert "age" in vars
        assert "user.email" in vars
        assert "sys.query" in vars
        assert "context.data" in vars
        assert "current.user" in vars
        assert "workflow.id" in vars

    def test_variable_extraction_whitespace_tolerance(self):
        """Test extraction handles whitespace variations."""
        text = "{{ name }} {{  age  }} {{user.email}}"
        vars = VariableExtractor.extract(text)

        assert "name" in vars
        assert "age" in vars
        assert "user.email" in vars

    def test_variable_extraction_uniqueness(self):
        """Test that duplicate variables appear once."""
        text = "{{name}} and {{name}} again {{name}}"
        vars = VariableExtractor.extract(text)

        assert vars.count("name") == 1
        assert len(vars) == 1

    def test_variable_extraction_order_preserved(self):
        """Test that variable order is preserved."""
        text = "{{first}} {{second}} {{third}}"
        vars = VariableExtractor.extract(text)

        assert vars == ["first", "second", "third"]

    def test_variable_validation_missing_detection(self):
        """Test validation detects missing variables."""
        original_text = "Process {{data.input}} with {{config.model}}"
        optimized_text = "Process {{data.input}} only"  # Missing config.model

        original_vars = VariableExtractor.extract(original_text)
        missing = VariableExtractor.validate_variables(optimized_text, original_vars)

        assert "config.model" in missing
        assert "data.input" not in missing

    def test_variable_validation_all_present(self):
        """Test validation passes when all variables present."""
        original_text = "{{name}} {{age}}"
        optimized_text = "Name: {{name}} Age: {{age}}"

        original_vars = VariableExtractor.extract(original_text)
        missing = VariableExtractor.validate_variables(optimized_text, original_vars)

        assert len(missing) == 0

    def test_variable_extraction_empty_text(self):
        """Test extraction from empty text."""
        vars = VariableExtractor.extract("")
        assert vars == []

    def test_variable_extraction_no_variables(self):
        """Test extraction from text without variables."""
        text = "This is plain text with no variables"
        vars = VariableExtractor.extract(text)
        assert vars == []

    def test_variable_extraction_malformed_ignored(self):
        """Test that malformed syntax is ignored."""
        text = "Valid: {{name}} Invalid: { {age} } {missing}"
        vars = VariableExtractor.extract(text)

        assert "name" in vars
        assert len(vars) == 1  # Only valid syntax extracted

    def test_optimization_preserves_variables_integration(self):
        """Integration test: optimization preserves all variables."""
        from src.optimizer.models import Prompt

        service = OptimizerService()

        prompt = Prompt(
            id="test_prompt",
            workflow_id="test_wf",
            node_id="test_node",
            node_type="llm",
            text="Process {{data.input}} with {{config.model}}",
            role="user",
            variables=["data.input", "config.model"],
            context={},
            extracted_at=datetime.now()
        )

        # Mock optimization result
        optimized_text = "Process {{data.input}} using {{config.model}}"

        # Should not raise
        service._validate_optimized_prompt(prompt, optimized_text)

    def test_optimization_detects_missing_variables_integration(self):
        """Integration test: optimization detects lost variables."""
        from src.optimizer.models import Prompt
        from src.optimizer.exceptions import OptimizerError

        service = OptimizerService()

        prompt = Prompt(
            id="test_prompt",
            workflow_id="test_wf",
            node_id="test_node",
            node_type="llm",
            text="Process {{data.input}} with {{config.model}}",
            role="user",
            variables=["data.input", "config.model"],
            context={},
            extracted_at=datetime.now()
        )

        # Optimized text missing config.model
        optimized_text = "Process {{data.input}} only"

        # Should raise OptimizerError
        with pytest.raises(OptimizerError, match="lost required variables"):
            service._validate_optimized_prompt(prompt, optimized_text)

    def test_variable_extraction_complex_nesting(self):
        """Test extraction of deeply nested variables."""
        text = "{{user.profile.settings.theme}} {{sys.config.api.endpoint}}"
        vars = VariableExtractor.extract(text)

        assert "user.profile.settings.theme" in vars
        assert "sys.config.api.endpoint" in vars

    def test_variable_extraction_underscores_numbers(self):
        """Test extraction supports underscores and numbers."""
        text = "{{user_name}} {{value_123}} {{data_v2}}"
        vars = VariableExtractor.extract(text)

        assert "user_name" in vars
        assert "value_123" in vars
        assert "data_v2" in vars
