"""
Test Cases for Optimizer Data Models

Date: 2025-11-17
Author: qa-engineer
Description: Unit tests for Pydantic models in optimizer module
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.optimizer.models import (
    Prompt,
    PromptIssue,
    PromptSuggestion,
    PromptAnalysis,
    OptimizationResult,
    PromptVersion,
    OptimizationConfig,
    IssueSeverity,
    IssueType,
    SuggestionType,
    OptimizationStrategy,
)


class TestPromptModel:
    """Test cases for Prompt model."""

    def test_prompt_creation_with_required_fields(self):
        """Test creating Prompt with required fields only."""
        prompt = Prompt(
            id="test_id",
            workflow_id="wf_001",
            node_id="node_1",
            node_type="llm",
            text="Test prompt text",
        )
        assert prompt.id == "test_id"
        assert prompt.workflow_id == "wf_001"
        assert prompt.node_id == "node_1"
        assert prompt.node_type == "llm"
        assert prompt.text == "Test prompt text"
        assert prompt.role == "system"  # default
        assert prompt.variables == []  # default
        assert prompt.context == {}  # default

    def test_prompt_creation_with_all_fields(self):
        """Test creating Prompt with all fields."""
        extracted_at = datetime(2025, 11, 17, 12, 0, 0)
        prompt = Prompt(
            id="full_id",
            workflow_id="wf_002",
            node_id="node_2",
            node_type="llm",
            text="Hello {{name}}",
            role="user",
            variables=["name"],
            context={"label": "Greeter"},
            extracted_at=extracted_at,
        )
        assert prompt.role == "user"
        assert prompt.variables == ["name"]
        assert prompt.context == {"label": "Greeter"}
        assert prompt.extracted_at == extracted_at

    def test_prompt_empty_text_raises_validation_error(self):
        """Test that empty text raises ValidationError."""
        with pytest.raises(ValidationError, match="Prompt text cannot be empty"):
            Prompt(
                id="invalid",
                workflow_id="wf",
                node_id="n",
                node_type="llm",
                text="",
            )

    def test_prompt_whitespace_only_text_raises_error(self):
        """Test that whitespace-only text raises ValidationError."""
        with pytest.raises(ValidationError, match="Prompt text cannot be empty"):
            Prompt(
                id="invalid",
                workflow_id="wf",
                node_id="n",
                node_type="llm",
                text="   \n\t  ",
            )

    def test_prompt_invalid_variable_name_raises_error(self):
        """Test that empty variable names raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid variable name"):
            Prompt(
                id="test",
                workflow_id="wf",
                node_id="n",
                node_type="llm",
                text="Test",
                variables=["valid", ""],
            )

    def test_prompt_model_dump(self):
        """Test that model_dump() works correctly."""
        prompt = Prompt(
            id="dump_test",
            workflow_id="wf",
            node_id="n",
            node_type="llm",
            text="Text",
            variables=["var1"],
        )
        data = prompt.model_dump()
        assert data["id"] == "dump_test"
        assert data["text"] == "Text"
        assert data["variables"] == ["var1"]
        assert "extracted_at" in data

    def test_prompt_model_validate(self):
        """Test that model_validate() works correctly."""
        data = {
            "id": "validate_test",
            "workflow_id": "wf",
            "node_id": "n",
            "node_type": "llm",
            "text": "Valid text",
            "role": "assistant",
            "variables": ["x", "y"],
            "context": {"key": "value"},
            "extracted_at": datetime.now().isoformat(),
        }
        prompt = Prompt.model_validate(data)
        assert prompt.id == "validate_test"
        assert prompt.role == "assistant"

    def test_prompt_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError):
            Prompt(
                id="test",
                workflow_id="wf",
                node_id="n",
                node_type="llm",
                text="Text",
                extra_field="not allowed",
            )


class TestPromptIssueModel:
    """Test cases for PromptIssue model."""

    def test_issue_creation_with_required_fields(self):
        """Test creating PromptIssue with required fields."""
        issue = PromptIssue(
            severity=IssueSeverity.WARNING,
            type=IssueType.VAGUE_LANGUAGE,
            description="Contains vague terms",
        )
        assert issue.severity == IssueSeverity.WARNING
        assert issue.type == IssueType.VAGUE_LANGUAGE
        assert issue.description == "Contains vague terms"
        assert issue.location is None
        assert issue.suggestion is None

    def test_issue_creation_with_all_fields(self):
        """Test creating PromptIssue with all fields."""
        issue = PromptIssue(
            severity=IssueSeverity.CRITICAL,
            type=IssueType.TOO_LONG,
            description="Prompt is too long",
            location="Lines 1-100",
            suggestion="Reduce length",
        )
        assert issue.location == "Lines 1-100"
        assert issue.suggestion == "Reduce length"

    @pytest.mark.parametrize(
        "severity",
        [IssueSeverity.CRITICAL, IssueSeverity.WARNING, IssueSeverity.INFO],
    )
    def test_issue_all_severity_levels(self, severity):
        """Test all severity levels are valid."""
        issue = PromptIssue(
            severity=severity, type=IssueType.TOO_SHORT, description="Test"
        )
        assert issue.severity == severity

    @pytest.mark.parametrize(
        "issue_type",
        [
            IssueType.TOO_LONG,
            IssueType.TOO_SHORT,
            IssueType.VAGUE_LANGUAGE,
            IssueType.MISSING_STRUCTURE,
            IssueType.REDUNDANCY,
            IssueType.POOR_FORMATTING,
            IssueType.AMBIGUOUS_INSTRUCTIONS,
        ],
    )
    def test_issue_all_types(self, issue_type):
        """Test all issue types are valid."""
        issue = PromptIssue(
            severity=IssueSeverity.INFO, type=issue_type, description="Test"
        )
        assert issue.type == issue_type


class TestPromptSuggestionModel:
    """Test cases for PromptSuggestion model."""

    def test_suggestion_creation(self):
        """Test creating PromptSuggestion."""
        suggestion = PromptSuggestion(
            type=SuggestionType.ADD_STRUCTURE,
            description="Add headers",
            priority=8,
        )
        assert suggestion.type == SuggestionType.ADD_STRUCTURE
        assert suggestion.description == "Add headers"
        assert suggestion.priority == 8

    @pytest.mark.parametrize("priority", [1, 5, 10])
    def test_suggestion_valid_priorities(self, priority):
        """Test valid priority values (1-10)."""
        suggestion = PromptSuggestion(
            type=SuggestionType.CLARIFY_INSTRUCTIONS,
            description="Test",
            priority=priority,
        )
        assert suggestion.priority == priority

    def test_suggestion_priority_too_low_raises_error(self):
        """Test that priority < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            PromptSuggestion(
                type=SuggestionType.REDUCE_LENGTH,
                description="Test",
                priority=0,
            )

    def test_suggestion_priority_too_high_raises_error(self):
        """Test that priority > 10 raises ValidationError."""
        with pytest.raises(ValidationError):
            PromptSuggestion(
                type=SuggestionType.ADD_EXAMPLES,
                description="Test",
                priority=11,
            )


class TestPromptAnalysisModel:
    """Test cases for PromptAnalysis model."""

    def test_analysis_creation_with_required_fields(self):
        """Test creating PromptAnalysis with required fields."""
        analysis = PromptAnalysis(
            prompt_id="test_id",
            overall_score=75.0,
            clarity_score=80.0,
            efficiency_score=68.0,
        )
        assert analysis.prompt_id == "test_id"
        assert analysis.overall_score == 75.0
        assert analysis.clarity_score == 80.0
        assert analysis.efficiency_score == 68.0
        assert analysis.issues == []
        assert analysis.suggestions == []
        assert analysis.metadata == {}

    def test_analysis_creation_with_all_fields(self, sample_analysis):
        """Test creating PromptAnalysis with all fields."""
        assert len(sample_analysis.issues) == 1
        assert len(sample_analysis.suggestions) == 1
        assert "character_count" in sample_analysis.metadata

    @pytest.mark.parametrize("score", [0.0, 50.0, 100.0])
    def test_analysis_valid_score_range(self, score):
        """Test valid score values (0-100)."""
        analysis = PromptAnalysis(
            prompt_id="test",
            overall_score=score,
            clarity_score=score,
            efficiency_score=score,
        )
        assert analysis.overall_score == score

    def test_analysis_score_below_zero_raises_error(self):
        """Test that score < 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            PromptAnalysis(
                prompt_id="test",
                overall_score=-1.0,
                clarity_score=50.0,
                efficiency_score=50.0,
            )

    def test_analysis_score_above_hundred_raises_error(self):
        """Test that score > 100 raises ValidationError."""
        with pytest.raises(ValidationError):
            PromptAnalysis(
                prompt_id="test",
                overall_score=101.0,
                clarity_score=50.0,
                efficiency_score=50.0,
            )

    def test_analysis_clarity_score_invalid(self):
        """Test invalid clarity score."""
        with pytest.raises(ValidationError):
            PromptAnalysis(
                prompt_id="test",
                overall_score=50.0,
                clarity_score=150.0,
                efficiency_score=50.0,
            )

    def test_analysis_efficiency_score_invalid(self):
        """Test invalid efficiency score."""
        with pytest.raises(ValidationError):
            PromptAnalysis(
                prompt_id="test",
                overall_score=50.0,
                clarity_score=50.0,
                efficiency_score=-10.0,
            )


class TestOptimizationResultModel:
    """Test cases for OptimizationResult model."""

    def test_result_creation(self):
        """Test creating OptimizationResult."""
        result = OptimizationResult(
            prompt_id="test",
            original_prompt="Original",
            optimized_prompt="Optimized",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=10.0,
            confidence=0.8,
        )
        assert result.prompt_id == "test"
        assert result.original_prompt == "Original"
        assert result.optimized_prompt == "Optimized"
        assert result.strategy == OptimizationStrategy.CLARITY_FOCUS
        assert result.improvement_score == 10.0
        assert result.confidence == 0.8
        assert result.changes == []

    def test_result_negative_improvement_score_allowed(self):
        """Test that negative improvement scores are allowed."""
        result = OptimizationResult(
            prompt_id="test",
            original_prompt="Good",
            optimized_prompt="Worse",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=-5.0,
            confidence=0.3,
        )
        assert result.improvement_score == -5.0

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_result_valid_confidence_range(self, confidence):
        """Test valid confidence values (0.0-1.0)."""
        result = OptimizationResult(
            prompt_id="test",
            original_prompt="O",
            optimized_prompt="N",
            strategy=OptimizationStrategy.STRUCTURE_FOCUS,
            improvement_score=0.0,
            confidence=confidence,
        )
        assert result.confidence == confidence

    def test_result_confidence_below_zero_raises_error(self):
        """Test that confidence < 0 raises ValidationError."""
        with pytest.raises(ValidationError):
            OptimizationResult(
                prompt_id="test",
                original_prompt="O",
                optimized_prompt="N",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=0.0,
                confidence=-0.1,
            )

    def test_result_confidence_above_one_raises_error(self):
        """Test that confidence > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError):
            OptimizationResult(
                prompt_id="test",
                original_prompt="O",
                optimized_prompt="N",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=0.0,
                confidence=1.5,
            )


class TestPromptVersionModel:
    """Test cases for PromptVersion model."""

    def test_version_creation(self, sample_prompt, sample_analysis):
        """Test creating PromptVersion."""
        version = PromptVersion(
            prompt_id="test",
            version="1.0.0",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )
        assert version.prompt_id == "test"
        assert version.version == "1.0.0"
        assert version.prompt == sample_prompt
        assert version.analysis == sample_analysis
        assert version.optimization_result is None
        assert version.parent_version is None

    @pytest.mark.parametrize(
        "version_str",
        ["1.0.0", "2.5.10", "0.0.1", "10.20.30", "99.99.99"],
    )
    def test_version_valid_semantic_versions(
            self, version_str, sample_prompt, sample_analysis
    ):
        """Test valid semantic version formats."""
        version = PromptVersion(
            prompt_id="test",
            version=version_str,
            prompt=sample_prompt,
            analysis=sample_analysis,
        )
        assert version.version == version_str

    @pytest.mark.parametrize(
        "invalid_version",
        ["1.0", "1", "1.0.0.0", "a.b.c", "1.0.a", "-1.0.0", "1.-1.0"],
    )
    def test_version_invalid_format_raises_error(
            self, invalid_version, sample_prompt, sample_analysis
    ):
        """Test invalid version formats raise ValidationError."""
        with pytest.raises(ValidationError):
            PromptVersion(
                prompt_id="test",
                version=invalid_version,
                prompt=sample_prompt,
                analysis=sample_analysis,
            )

    def test_version_get_version_number(self, sample_prompt, sample_analysis):
        """Test get_version_number() method."""
        version = PromptVersion(
            prompt_id="test",
            version="2.5.10",
            prompt=sample_prompt,
            analysis=sample_analysis,
        )
        assert version.get_version_number() == (2, 5, 10)

    def test_version_is_baseline_true(self, sample_prompt, sample_analysis):
        """Test is_baseline() returns True when no parent."""
        version = PromptVersion(
            prompt_id="test",
            version="1.0.0",
            prompt=sample_prompt,
            analysis=sample_analysis,
            parent_version=None,
        )
        assert version.is_baseline() is True

    def test_version_is_baseline_false(self, sample_prompt, sample_analysis):
        """Test is_baseline() returns False when has parent."""
        version = PromptVersion(
            prompt_id="test",
            version="1.1.0",
            prompt=sample_prompt,
            analysis=sample_analysis,
            parent_version="1.0.0",
        )
        assert version.is_baseline() is False


class TestOptimizationConfigModel:
    """Test cases for OptimizationConfig model."""

    def test_config_default_values(self):
        """Test default values in OptimizationConfig."""
        config = OptimizationConfig()
        assert config.strategies == [OptimizationStrategy.AUTO]
        assert config.min_confidence == 0.6
        assert config.max_iterations == 3
        assert config.analysis_rules == {}
        assert config.metadata == {}

    def test_config_custom_values(self):
        """Test custom values in OptimizationConfig."""
        config = OptimizationConfig(
            strategies=[
                OptimizationStrategy.CLARITY_FOCUS,
                OptimizationStrategy.EFFICIENCY_FOCUS,
            ],
            min_confidence=0.8,
            max_iterations=5,
            analysis_rules={"rule1": "value"},
            metadata={"key": "data"},
        )
        assert len(config.strategies) == 2
        assert config.min_confidence == 0.8
        assert config.max_iterations == 5

    def test_config_empty_strategies_raises_error(self):
        """Test that empty strategies list raises ValidationError."""
        with pytest.raises(
                ValidationError, match="At least one strategy must be specified"
        ):
            OptimizationConfig(strategies=[])

    @pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
    def test_config_valid_confidence_range(self, confidence):
        """Test valid min_confidence values."""
        config = OptimizationConfig(min_confidence=confidence)
        assert config.min_confidence == confidence

    def test_config_confidence_out_of_range_raises_error(self):
        """Test that min_confidence out of range raises error."""
        with pytest.raises(ValidationError):
            OptimizationConfig(min_confidence=1.5)

    def test_config_max_iterations_minimum(self):
        """Test that max_iterations must be >= 1."""
        with pytest.raises(ValidationError):
            OptimizationConfig(max_iterations=0)


class TestEnumerations:
    """Test cases for enumeration classes."""

    def test_issue_severity_values(self):
        """Test IssueSeverity enum values."""
        assert IssueSeverity.CRITICAL.value == "critical"
        assert IssueSeverity.WARNING.value == "warning"
        assert IssueSeverity.INFO.value == "info"

    def test_issue_type_values(self):
        """Test IssueType enum values."""
        assert IssueType.TOO_LONG.value == "too_long"
        assert IssueType.TOO_SHORT.value == "too_short"
        assert IssueType.VAGUE_LANGUAGE.value == "vague_language"
        assert IssueType.MISSING_STRUCTURE.value == "missing_structure"
        assert IssueType.REDUNDANCY.value == "redundancy"
        assert IssueType.POOR_FORMATTING.value == "poor_formatting"
        assert IssueType.AMBIGUOUS_INSTRUCTIONS.value == "ambiguous_instructions"

    def test_suggestion_type_values(self):
        """Test SuggestionType enum values."""
        assert SuggestionType.ADD_STRUCTURE.value == "add_structure"
        assert SuggestionType.CLARIFY_INSTRUCTIONS.value == "clarify_instructions"
        assert SuggestionType.REDUCE_LENGTH.value == "reduce_length"
        assert SuggestionType.ADD_EXAMPLES.value == "add_examples"
        assert SuggestionType.IMPROVE_FORMATTING.value == "improve_formatting"

    def test_optimization_strategy_values(self):
        """Test OptimizationStrategy enum values."""
        assert OptimizationStrategy.CLARITY_FOCUS.value == "clarity_focus"
        assert OptimizationStrategy.EFFICIENCY_FOCUS.value == "efficiency_focus"
        assert OptimizationStrategy.STRUCTURE_FOCUS.value == "structure_focus"
        assert OptimizationStrategy.AUTO.value == "auto"
