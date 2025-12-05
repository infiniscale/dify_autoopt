"""
Test Cases for OptimizerService

Date: 2025-11-17
Author: qa-engineer
Description: Unit tests for high-level optimizer service
"""

import pytest
from unittest.mock import MagicMock

from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.exceptions import WorkflowNotFoundError, OptimizerError


class TestOptimizerServiceBasic:
    """Basic test cases for OptimizerService."""

    def test_service_initialization(self, optimizer_service):
        """Test that OptimizerService can be initialized."""
        assert optimizer_service is not None
        assert isinstance(optimizer_service, OptimizerService)

    def test_optimize_single_prompt(self, optimizer_service, sample_prompt):
        """Test optimizing single prompt."""
        result = optimizer_service.optimize_single_prompt(sample_prompt, "clarity_focus")
        assert result is not None
        assert result.prompt_id == sample_prompt.id
        assert len(result.optimized_prompt) > 0

    def test_optimize_single_prompt_auto_strategy(self, optimizer_service, sample_prompt):
        """Test optimizing with auto strategy selection."""
        result = optimizer_service.optimize_single_prompt(sample_prompt, "auto")
        assert result is not None
        assert result.strategy.value in ["clarity_focus", "efficiency_focus", "structure_focus"]


class TestOptimizationCycle:
    """Test cases for full optimization cycle."""

    def test_run_optimization_cycle_with_catalog(self, mock_catalog):
        """Test running optimization cycle with catalog."""
        service = OptimizerService(catalog=mock_catalog)
        patches = service.run_optimization_cycle(
            workflow_id="test_workflow_001",
            strategy="clarity_focus"
        )
        assert isinstance(patches, list)

    def test_run_optimization_cycle_workflow_not_found(self, mock_catalog):
        """Test that missing workflow raises error."""
        mock_catalog.get_workflow.return_value = None
        service = OptimizerService(catalog=mock_catalog)

        with pytest.raises(WorkflowNotFoundError):
            service.run_optimization_cycle(
                workflow_id="nonexistent",
                strategy="clarity_focus"
            )

    def test_run_optimization_cycle_no_catalog_raises_error(self, optimizer_service):
        """Test that running cycle without catalog raises error."""
        with pytest.raises(OptimizerError):
            optimizer_service.run_optimization_cycle(
                workflow_id="test",
                strategy="clarity_focus"
            )


class TestAnalyzeWorkflow:
    """Test cases for workflow analysis."""

    def test_analyze_workflow(self, mock_catalog):
        """Test analyzing workflow without optimization."""
        service = OptimizerService(catalog=mock_catalog)
        report = service.analyze_workflow("test_workflow_001")

        assert "workflow_id" in report
        assert "prompt_count" in report
        assert "average_score" in report
        assert "prompts" in report

    def test_analyze_workflow_with_prompts(self, mock_catalog):
        """Test workflow analysis returns prompt details."""
        service = OptimizerService(catalog=mock_catalog)
        report = service.analyze_workflow("test_workflow_001")

        assert report["prompt_count"] >= 0
        if report["prompt_count"] > 0:
            assert len(report["prompts"]) > 0


class TestVersionHistory:
    """Test cases for version history retrieval."""

    def test_get_version_history(self, optimizer_service, sample_prompt, sample_analysis):
        """Test getting version history through service."""
        # Create a version
        optimizer_service._version_manager.create_version(
            sample_prompt, sample_analysis, None, None
        )

        history = optimizer_service.get_version_history(sample_prompt.id)
        assert len(history) > 0
        assert "version" in history[0]
        assert "score" in history[0]


class TestStrategySelection:
    """Test cases for automatic strategy selection."""

    def test_select_strategy_low_clarity(self, optimizer_service, sample_analysis_low_score):
        """Test strategy selection for low clarity."""
        # Modify analysis to have lower clarity than efficiency
        sample_analysis_low_score.clarity_score = 40.0
        sample_analysis_low_score.efficiency_score = 60.0

        strategy = optimizer_service._select_strategy(sample_analysis_low_score, None)
        assert strategy == "clarity_focus"

    def test_select_strategy_low_efficiency(self, optimizer_service, sample_analysis):
        """Test strategy selection for low efficiency."""
        sample_analysis.clarity_score = 80.0
        sample_analysis.efficiency_score = 50.0

        strategy = optimizer_service._select_strategy(sample_analysis, None)
        assert strategy == "efficiency_focus"


class TestShouldOptimize:
    """Test cases for optimization decision logic."""

    def test_should_optimize_low_score(self, optimizer_service, sample_analysis_low_score):
        """Test that low score prompts should be optimized."""
        should_optimize = optimizer_service._should_optimize(sample_analysis_low_score, None, None)
        assert should_optimize is True

    def test_should_not_optimize_high_score(self, optimizer_service, sample_analysis_high_score):
        """Test that high score prompts may not need optimization."""
        should_optimize = optimizer_service._should_optimize(sample_analysis_high_score, None, None)
        # High score should not need optimization
        assert should_optimize is False


class TestPromptPatchCreation:
    """Test cases for PromptPatch creation."""

    def test_create_prompt_patch(self, optimizer_service, sample_prompt, sample_optimization_result):
        """Test creating PromptPatch from optimization result."""
        patch = optimizer_service._create_prompt_patch(sample_prompt, sample_optimization_result)

        assert patch is not None
        assert patch.selector.by_id == sample_prompt.node_id
        assert patch.strategy.mode == "replace"
        assert patch.strategy.content == sample_optimization_result.optimized_prompt
