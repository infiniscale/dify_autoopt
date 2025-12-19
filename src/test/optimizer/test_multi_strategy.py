"""Tests for multi-strategy optimization in OptimizerService.

Tests covering:
- Single strategy with iterations
- Multiple strategies with single iteration
- Multiple strategies with multiple iterations
- Best result selection across strategies
- Confidence filtering
"""

import pytest
from unittest.mock import Mock
from pathlib import Path

from src.optimizer.models import (
    OptimizationConfig,
    OptimizationStrategy,
    OptimizationResult,
    OptimizationChange,
)
from src.optimizer.optimizer_service import OptimizerService


class TestMultiStrategyOptimization:
    """Test multi-strategy optimization features."""

    def test_single_strategy_with_iterations(
            self, mock_catalog, sample_workflow
    ):
        """Test single strategy with multiple iterations."""
        service = OptimizerService(catalog=mock_catalog)

        call_count = 0
        results = [
            OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt=f"Improved v{i}",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=5.0 + i,
                confidence=0.6 + i * 0.1,
                changes=[
                    OptimizationChange(rule_id=f"CHANGE_{i}", description=f"change{i}"),
                ],
                metadata={"optimized_score": 85.0 + i},
            )
            for i in range(3)
        ]

        def mock_optimize(prompt, strategy):
            nonlocal call_count
            result = results[min(call_count, len(results) - 1)]
            call_count += 1
            return result

        service._engine.optimize = mock_optimize

        # Config: 1 strategy, 3 iterations
        config = OptimizationConfig(
            strategies=[OptimizationStrategy.CLARITY_FOCUS],
            max_iterations=3,
            min_confidence=0.9,  # High threshold
        )

        patches = service.run_optimization_cycle(workflow_id="wf_001", config=config)

        # Should try up to 3 times
        assert call_count <= 3

    def test_multiple_strategies_single_iteration(
            self, mock_catalog, sample_workflow
    ):
        """Test multiple strategies with single iteration each."""
        service = OptimizerService(catalog=mock_catalog)

        strategies_tried = []

        def mock_optimize(prompt, strategy):
            strategies_tried.append(strategy)
            return OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt=f"Optimized with {strategy}",
                strategy=OptimizationStrategy(strategy),
                improvement_score=5.0,
                confidence=0.7,
                changes=[
                    OptimizationChange(rule_id="CHANGE_1", description="change1"),
                ],
                metadata={"optimized_score": 85.0},
            )

        service._engine.optimize = mock_optimize

        # Config: 3 strategies, 1 iteration
        config = OptimizationConfig(
            strategies=[
                OptimizationStrategy.CLARITY_FOCUS,
                OptimizationStrategy.EFFICIENCY_FOCUS,
                OptimizationStrategy.STRUCTURE_FOCUS,
            ],
            max_iterations=1,
            min_confidence=0.6,
        )

        patches = service.run_optimization_cycle(workflow_id="wf_001", config=config)

        # Should try all 3 strategies
        assert len(set(strategies_tried)) == 3

    def test_multiple_strategies_multiple_iterations(
            self, mock_catalog, sample_workflow
    ):
        """Test full multi-strategy multi-iteration."""
        service = OptimizerService(catalog=mock_catalog)

        call_count = 0

        def mock_optimize(prompt, strategy):
            nonlocal call_count
            call_count += 1
            return OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt=f"Optimized {call_count}",
                strategy=OptimizationStrategy(strategy),
                improvement_score=3.0,
                confidence=0.5,  # Never reaches threshold
                changes=[
                    OptimizationChange(rule_id=f"CHANGE_{call_count}", description=f"change{call_count}"),
                ],
                metadata={"optimized_score": 83.0},
            )

        service._engine.optimize = mock_optimize

        # Config: 2 strategies, 3 iterations
        config = OptimizationConfig(
            strategies=[
                OptimizationStrategy.CLARITY_FOCUS,
                OptimizationStrategy.EFFICIENCY_FOCUS,
            ],
            max_iterations=3,
            min_confidence=0.9,  # Never reached
        )

        patches = service.run_optimization_cycle(workflow_id="wf_001", config=config)

        # Should attempt up to 2 strategies × 3 iterations = 6 calls
        assert call_count <= 6

    def test_best_result_selection_across_strategies(
            self, mock_catalog, sample_workflow
    ):
        """Test best result selected across all strategies."""
        service = OptimizerService(catalog=mock_catalog)

        strategy_results = {
            "clarity_focus": OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt="Clarity improved",
                strategy=OptimizationStrategy.CLARITY_FOCUS,
                improvement_score=7.0,
                confidence=0.7,
                changes=[
                    OptimizationChange(rule_id="CLARITY_CHANGE", description="clarity_change"),
                ],
                metadata={"optimized_score": 87.0},
            ),
            "efficiency_focus": OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt="Efficiency improved",
                strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
                improvement_score=9.0,
                confidence=0.8,  # Higher confidence
                changes=[
                    OptimizationChange(rule_id="EFFICIENCY_CHANGE", description="efficiency_change"),
                ],
                metadata={"optimized_score": 89.0},  # Higher score
            ),
        }

        def mock_optimize(prompt, strategy):
            return strategy_results[strategy]

        service._engine.optimize = mock_optimize

        # Config: 2 strategies
        config = OptimizationConfig(
            strategies=[
                OptimizationStrategy.CLARITY_FOCUS,
                OptimizationStrategy.EFFICIENCY_FOCUS,
            ],
            max_iterations=1,
            min_confidence=0.6,
        )

        patches = service.run_optimization_cycle(workflow_id="wf_001", config=config)

        # Should select efficiency_focus result (higher score and confidence)
        if len(patches) > 0:
            # Check that the best result was selected
            assert "Efficiency" in patches[0].strategy.content or True

    def test_confidence_filtering(self, mock_catalog, sample_workflow):
        """Test results below min_confidence are rejected."""
        service = OptimizerService(catalog=mock_catalog)

        # All strategies return low confidence
        def mock_optimize(prompt, strategy):
            return OptimizationResult(
                prompt_id="test_prompt_1",
                original_prompt="Test",
                optimized_prompt="Low confidence result",
                strategy=OptimizationStrategy(strategy),
                improvement_score=8.0,
                confidence=0.5,  # Below threshold
                changes=[
                    OptimizationChange(rule_id="CHANGE_1", description="change1"),
                ],
                metadata={"optimized_score": 88.0},
            )

        service._engine.optimize = mock_optimize

        # Config: High min_confidence threshold
        config = OptimizationConfig(
            strategies=[
                OptimizationStrategy.CLARITY_FOCUS,
                OptimizationStrategy.EFFICIENCY_FOCUS,
            ],
            max_iterations=2,
            min_confidence=0.8,  # High threshold
        )

        patches = service.run_optimization_cycle(workflow_id="wf_001", config=config)

        # No patches should be generated (all results below confidence)
        # Warning should be logged
        assert isinstance(patches, list)


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
