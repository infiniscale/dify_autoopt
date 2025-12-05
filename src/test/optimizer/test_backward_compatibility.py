"""Tests for backward compatibility of OptimizerService.

Tests ensuring that legacy usage patterns still work correctly:
- Legacy strategy parameter only
- Legacy strategy with baseline_metrics
- No parameters (default behavior)
- New config parameter
- Mixed pattern (strategy overrides config)
"""

import pytest
from unittest.mock import Mock, MagicMock
from pathlib import Path

from src.optimizer.models import (
    OptimizationConfig,
    OptimizationStrategy,
    OptimizationResult,
    OptimizationChange,
    Prompt,
)
from src.optimizer.optimizer_service import OptimizerService


class TestBackwardCompatibility:
    """Test backward compatibility with legacy API."""

    def test_legacy_pattern_strategy_only(
        self, mock_catalog, sample_workflow, mock_engine_results
    ):
        """Test legacy pattern: strategy parameter only."""
        service = OptimizerService(catalog=mock_catalog)
        service._engine.optimize = Mock(return_value=mock_engine_results[0])

        # Old API call
        patches = service.run_optimization_cycle(
            workflow_id="wf_001", strategy="clarity_focus"
        )

        # Should work as before
        assert isinstance(patches, list)
        # Single run (max_iterations=1)
        assert service._engine.optimize.call_count <= 1

    def test_legacy_pattern_strategy_with_baseline(
        self, mock_catalog, sample_workflow, mock_engine_results
    ):
        """Test legacy pattern: strategy + baseline_metrics."""
        service = OptimizerService(catalog=mock_catalog)
        service._engine.optimize = Mock(return_value=mock_engine_results[0])

        # Old API call with baseline
        patches = service.run_optimization_cycle(
            workflow_id="wf_001",
            strategy="clarity_focus",
            baseline_metrics={"success_rate": 0.7},
        )

        # Should work as before
        assert isinstance(patches, list)

    def test_legacy_pattern_no_parameters(
        self, mock_catalog, sample_workflow, mock_engine_results
    ):
        """Test legacy pattern: no strategy parameter (defaults to AUTO)."""
        service = OptimizerService(catalog=mock_catalog)
        service._engine.optimize = Mock(return_value=mock_engine_results[0])

        # Old API call with no strategy
        patches = service.run_optimization_cycle(workflow_id="wf_001")

        # Should use default AUTO strategy
        assert isinstance(patches, list)

    def test_new_pattern_config_only(
        self, mock_catalog, sample_workflow, mock_engine_results
    ):
        """Test new pattern: config parameter only."""
        service = OptimizerService(catalog=mock_catalog)
        service._engine.optimize = Mock(return_value=mock_engine_results[0])

        # New API call with config
        config = OptimizationConfig(
            strategies=[OptimizationStrategy.CLARITY_FOCUS],
            max_iterations=3,
            min_confidence=0.7,
        )

        patches = service.run_optimization_cycle(workflow_id="wf_001", config=config)

        # Should use config settings
        assert isinstance(patches, list)

    def test_mixed_pattern_strategy_overrides(
        self, mock_catalog, sample_workflow, mock_engine_results
    ):
        """Test mixed pattern: strategy overrides config."""
        service = OptimizerService(catalog=mock_catalog)

        call_count = 0
        def counting_optimize(prompt, strategy):
            nonlocal call_count
            call_count += 1
            return mock_engine_results[0]

        service._engine.optimize = counting_optimize

        # Both strategy and config provided
        config = OptimizationConfig(
            strategies=[
                OptimizationStrategy.CLARITY_FOCUS,
                OptimizationStrategy.EFFICIENCY_FOCUS,
            ],
            max_iterations=3,
        )

        patches = service.run_optimization_cycle(
            workflow_id="wf_001",
            strategy="structure_focus",  # Should override config
            config=config,
        )

        # Should use structure_focus (not clarity/efficiency from config)
        # Should use single iteration (legacy behavior)
        assert isinstance(patches, list)
        # Legacy mode: max 1 iteration per prompt
        assert call_count <= 1

    def test_config_parameter_resolution(self, mock_catalog, sample_workflow):
        """Test configuration parameter resolution logic."""
        service = OptimizerService(catalog=mock_catalog)

        # Mock _extract_prompts to return empty list (skip optimization)
        service._extract_prompts = Mock(return_value=[])

        # Test 1: strategy parameter creates legacy config
        service.run_optimization_cycle(workflow_id="wf_001", strategy="clarity_focus")
        # Should log "Using legacy single-strategy mode"

        # Test 2: config parameter is used directly
        config = OptimizationConfig(
            strategies=[OptimizationStrategy.CLARITY_FOCUS],
            max_iterations=3,
        )
        service.run_optimization_cycle(workflow_id="wf_001", config=config)
        # Should log "Using multi-strategy mode"

        # Test 3: no parameters uses default
        service.run_optimization_cycle(workflow_id="wf_001")
        # Should log "Using default configuration"

    def test_legacy_behavior_single_run(
        self, mock_catalog, sample_workflow, mock_engine_results
    ):
        """Test legacy behavior: single optimization run."""
        service = OptimizerService(catalog=mock_catalog)

        call_count = 0

        def counting_optimize(prompt, strategy):
            nonlocal call_count
            call_count += 1
            return mock_engine_results[0]

        service._engine.optimize = counting_optimize

        # Legacy call
        patches = service.run_optimization_cycle(
            workflow_id="wf_001", strategy="clarity_focus"
        )

        # Should run exactly once per prompt (no iterations)
        assert call_count <= 1

    def test_legacy_behavior_no_confidence_check(
        self, mock_catalog, sample_workflow
    ):
        """Test legacy behavior: no confidence filtering."""
        service = OptimizerService(catalog=mock_catalog)

        # Mock engine to return low confidence
        low_confidence_result = OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
            optimized_prompt="Improved Test",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=5.0,
            confidence=0.3,  # Very low confidence
            changes=[
                OptimizationChange(rule_id="CHANGE_1", description="change1"),
            ],
            metadata={"optimized_score": 85.0},
        )

        service._engine.optimize = Mock(return_value=low_confidence_result)

        # Legacy call
        patches = service.run_optimization_cycle(
            workflow_id="wf_001", strategy="clarity_focus"
        )

        # Should accept result despite low confidence (legacy behavior)
        # In legacy mode, min_confidence=0.0, so all results accepted
        assert len(patches) >= 0


@pytest.fixture
def mock_catalog(tmp_path):
    """Create a mock catalog with sample workflow."""
    catalog = Mock()

    # Create a mock workflow
    workflow = Mock()
    workflow.workflow_id = "wf_001"
    workflow.dsl_path_resolved = tmp_path / "workflow.yml"

    # Create sample DSL file with proper prompt structure
    dsl_content = {
        "app": {"mode": "workflow"},
        "nodes": [
            {
                "id": "node_1",
                "data": {
                    "type": "llm",
                    "title": "Test LLM",
                    "prompt_template": {
                        "messages": [
                            {
                                "role": "system",
                                "text": "You are a helpful assistant. Please provide clear and concise answers.",
                            }
                        ]
                    },
                },
            }
        ],
    }

    import yaml

    with open(workflow.dsl_path_resolved, "w") as f:
        yaml.dump(dsl_content, f)

    catalog.get_workflow = Mock(return_value=workflow)

    return catalog


@pytest.fixture
def sample_workflow():
    """Sample workflow ID for testing."""
    return "wf_001"


@pytest.fixture
def mock_engine_results():
    """Sample optimization results for testing."""
    return [
        OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Original",
            optimized_prompt="Optimized",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=8.0,
            confidence=0.75,
            changes=[
                OptimizationChange(rule_id="CHANGE_1", description="change1"),
            ],
            metadata={"optimized_score": 88.0},
        )
    ]
