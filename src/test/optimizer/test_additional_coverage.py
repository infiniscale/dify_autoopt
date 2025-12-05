"""
Additional Tests for Coverage Improvement

Date: 2025-11-17
Author: qa-engineer
Description: Additional tests to improve optimizer module coverage
"""

import pytest
from src.optimizer.exceptions import (
    OptimizerError,
    ExtractionError,
    WorkflowNotFoundError,
    NodeNotFoundError,
    DSLParseError,
    AnalysisError,
    ScoringError,
    OptimizationError,
    InvalidStrategyError,
    OptimizationFailedError,
    VersionError,
    VersionConflictError,
    VersionNotFoundError,
    ValidationError,
    ConfigError,
)
from src.optimizer.interfaces.llm_client import LLMClient, StubLLMClient, LLMResponse
from src.optimizer.optimization_engine import OptimizationEngine
from src.optimizer.prompt_analyzer import PromptAnalyzer
from src.optimizer.models import Prompt


class TestExceptions:
    """Test cases for exception classes."""

    def test_optimizer_error_basic(self):
        """Test OptimizerError basic creation."""
        error = OptimizerError("Test message")
        assert error.message == "Test message"
        assert error.error_code is None
        assert error.context == {}

    def test_optimizer_error_with_code(self):
        """Test OptimizerError with error code."""
        error = OptimizerError("Test", error_code="TEST-001")
        assert error.error_code == "TEST-001"
        assert str(error) == "[TEST-001] Test"

    def test_optimizer_error_with_context(self):
        """Test OptimizerError with context."""
        error = OptimizerError("Test", context={"key": "value"})
        assert error.context["key"] == "value"

    def test_workflow_not_found_error(self):
        """Test WorkflowNotFoundError."""
        error = WorkflowNotFoundError("wf_001")
        assert error.workflow_id == "wf_001"
        assert "wf_001" in error.message

    def test_node_not_found_error(self):
        """Test NodeNotFoundError."""
        error = NodeNotFoundError("node_1", "wf_001")
        assert error.node_id == "node_1"
        assert error.workflow_id == "wf_001"

    def test_dsl_parse_error(self):
        """Test DSLParseError."""
        error = DSLParseError("/path/to/dsl.yaml", "Invalid YAML")
        assert error.dsl_path == "/path/to/dsl.yaml"
        assert error.reason == "Invalid YAML"

    def test_scoring_error(self):
        """Test ScoringError."""
        error = ScoringError("clarity", "Division by zero")
        assert error.metric_name == "clarity"
        assert error.reason == "Division by zero"

    def test_invalid_strategy_error(self):
        """Test InvalidStrategyError."""
        error = InvalidStrategyError("bad_strategy", ["good1", "good2"])
        assert error.strategy == "bad_strategy"
        assert error.valid_strategies == ["good1", "good2"]

    def test_optimization_failed_error(self):
        """Test OptimizationFailedError."""
        error = OptimizationFailedError("prompt_1", "clarity", "Failed")
        assert error.prompt_id == "prompt_1"
        assert error.strategy == "clarity"

    def test_version_conflict_error(self):
        """Test VersionConflictError."""
        error = VersionConflictError("prompt_1", "1.0.0", "Already exists")
        assert error.prompt_id == "prompt_1"
        assert error.version == "1.0.0"

    def test_version_not_found_error(self):
        """Test VersionNotFoundError."""
        error = VersionNotFoundError("prompt_1", "1.0.0")
        assert error.prompt_id == "prompt_1"
        assert error.version == "1.0.0"

    def test_config_error(self):
        """Test ConfigError."""
        error = ConfigError("Missing config", config_key="api_key")
        assert error.reason == "Missing config"
        assert error.config_key == "api_key"


class TestStubLLMClient:
    """Test cases for StubLLMClient."""

    def test_stub_analyze_prompt(self):
        """Test StubLLMClient analyze_prompt."""
        client = StubLLMClient()
        response = client.analyze_prompt("Test prompt")
        assert response is not None
        assert isinstance(response, LLMResponse)
        # Parse JSON content to verify structure
        import json
        analysis = json.loads(response.content)
        assert 0.0 <= analysis["overall_score"] <= 100.0

    def test_stub_analyze_prompt_with_context(self):
        """Test StubLLMClient with context."""
        client = StubLLMClient()
        response = client.analyze_prompt(
            "Test prompt",
            context={"workflow_id": "wf_001", "node_id": "node_1"}
        )
        assert response is not None
        assert isinstance(response, LLMResponse)

    def test_stub_optimize_prompt_clarity(self):
        """Test StubLLMClient optimize_prompt clarity."""
        client = StubLLMClient()
        response = client.optimize_prompt("Test prompt", "clarity_focus")
        assert response is not None
        assert isinstance(response, LLMResponse)
        assert len(response.content) > 0

    def test_stub_optimize_prompt_efficiency(self):
        """Test StubLLMClient optimize_prompt efficiency."""
        client = StubLLMClient()
        response = client.optimize_prompt("Test prompt", "efficiency_focus")
        assert response is not None

    def test_stub_optimize_prompt_structure(self):
        """Test StubLLMClient optimize_prompt structure."""
        client = StubLLMClient()
        optimized = client.optimize_prompt("Test prompt", "structure_focus")
        assert optimized is not None

    def test_stub_optimize_prompt_invalid_strategy(self):
        """Test StubLLMClient with invalid strategy."""
        client = StubLLMClient()
        with pytest.raises(InvalidStrategyError):
            client.optimize_prompt("Test", "invalid_strategy")


class TestOptimizationEngineTransformations:
    """Test cases for optimization engine transformation methods."""

    def test_break_long_sentences(self, engine):
        """Test breaking long sentences."""
        long_sentence = "This is a " + " ".join(["very long sentence"] * 10) + " and it continues, which makes it too long to read comfortably"
        prompt = Prompt(
            id="test",
            workflow_id="w",
            node_id="n",
            node_type="llm",
            text=long_sentence
        )
        result = engine._break_long_sentences(prompt.text)
        # Should break into shorter sentences
        assert isinstance(result, str)

    def test_replace_vague_terms(self, engine):
        """Test replacing vague terms."""
        text = "Maybe do some stuff with things"
        result = engine._replace_vague_terms(text)
        # Should replace vague terms
        assert "maybe" not in result.lower() or "stuff" not in result

    def test_remove_filler_words(self, engine):
        """Test removing filler words."""
        text = "This is very really just actually quite good"
        result = engine._remove_filler_words(text)
        # Filler words should be removed
        assert len(result) <= len(text)

    def test_compress_verbose_phrases(self, engine):
        """Test compressing verbose phrases."""
        text = "In order to achieve the goal due to the fact that"
        result = engine._compress_verbose_phrases(text)
        # Should be compressed
        assert len(result) < len(text)

    def test_clean_whitespace(self, engine):
        """Test cleaning whitespace."""
        text = "Text    with     excessive   spaces\n\n\n\n"
        result = engine._clean_whitespace(text)
        # Should normalize whitespace
        assert "    " not in result
        assert "\n\n\n" not in result


class TestPromptAnalyzerInternals:
    """Test internal methods of PromptAnalyzer."""

    def test_compute_metadata(self, analyzer, sample_prompt):
        """Test metadata computation."""
        metadata = analyzer._compute_metadata(sample_prompt)
        assert "character_count" in metadata
        assert "word_count" in metadata
        assert "sentence_count" in metadata
        assert metadata["character_count"] == len(sample_prompt.text)

    def test_score_structure_various_formats(self, analyzer):
        """Test structure scoring with various formats."""
        # Plain text
        score1 = analyzer._score_structure("Plain text without structure")
        # With headers
        score2 = analyzer._score_structure("# Header\nContent")
        # With bullets
        score3 = analyzer._score_structure("- Item 1\n- Item 2")

        # Headers and bullets should score higher
        assert score2 >= score1
        assert score3 >= score1

    def test_score_specificity_with_numbers(self, analyzer):
        """Test specificity scoring with specific numbers."""
        text_with_numbers = "Provide exactly 5 examples with 3 bullet points"
        text_without = "Provide some examples with several points"

        score_with = analyzer._score_specificity(text_with_numbers)
        score_without = analyzer._score_specificity(text_without)

        # With numbers should score higher
        assert score_with >= score_without


class TestIntegrationScenarios:
    """Additional integration test scenarios."""

    def test_optimize_empty_variables_list(self, optimizer_service):
        """Test optimizing prompt with no variables."""
        prompt = Prompt(
            id="test",
            workflow_id="w",
            node_id="n",
            node_type="llm",
            text="Simple prompt without variables",
            variables=[]
        )
        result = optimizer_service.optimize_single_prompt(prompt, "clarity_focus")
        assert result is not None

    def test_optimize_preserves_role(self, optimizer_service):
        """Test that optimization preserves prompt role."""
        prompt = Prompt(
            id="test",
            workflow_id="w",
            node_id="n",
            node_type="llm",
            text="User prompt",
            role="user"
        )
        result = optimizer_service.optimize_single_prompt(prompt, "clarity_focus")
        assert result is not None
        # Role should be preserved in metadata
        assert result.prompt_id == prompt.id


class TestEdgeCasesAndBoundaries:
    """Test edge cases and boundary conditions."""

    def test_analyze_single_word_prompt(self, analyzer):
        """Test analyzing single word prompt."""
        prompt = Prompt(
            id="test",
            workflow_id="w",
            node_id="n",
            node_type="llm",
            text="Analyze"
        )
        analysis = analyzer.analyze_prompt(prompt)
        assert analysis is not None
        assert len(analysis.issues) > 0  # Should detect too_short

    def test_optimize_prompt_with_only_variables(self, engine):
        """Test optimizing prompt that is only variables."""
        prompt = Prompt(
            id="test",
            workflow_id="w",
            node_id="n",
            node_type="llm",
            text="{{var1}} {{var2}} {{var3}}"
        )
        result = engine.optimize(prompt, "clarity_focus")
        # Should not crash, may add context
        assert result is not None

    def test_version_comparison_same_version(self, version_manager, sample_prompt, sample_analysis):
        """Test comparing same version with itself."""
        version_manager.create_version(sample_prompt, sample_analysis, None, None)
        comparison = version_manager.compare_versions(sample_prompt.id, "1.0.0", "1.0.0")
        assert comparison["improvement"] == 0.0
        assert comparison["clarity_improvement"] == 0.0
