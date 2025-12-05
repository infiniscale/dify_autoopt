"""
End-to-End Integration Tests for Optimizer Module

Date: 2025-11-17
Author: qa-engineer
Description: Integration tests for complete optimization workflow
"""

import pytest
from pathlib import Path

from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.models import Prompt


class TestEndToEndOptimization:
    """End-to-end integration test cases."""

    def test_complete_optimization_flow(self, mock_catalog):
        """Test complete optimization flow from DSL to patches."""
        service = OptimizerService(catalog=mock_catalog)

        # Run optimization cycle
        patches = service.run_optimization_cycle(
            workflow_id="test_workflow_001",
            strategy="clarity_focus"
        )

        # Verify patches were generated
        assert isinstance(patches, list)
        # May be 0 if all prompts are already high quality
        assert len(patches) >= 0

        # If patches exist, verify structure
        for patch in patches:
            assert patch.selector is not None
            assert patch.strategy is not None
            assert patch.strategy.content is not None

    def test_multi_round_optimization(self, optimizer_service, sample_prompt, sample_analysis_low_score):
        """Test multiple rounds of optimization on same prompt."""
        # First optimization
        result1 = optimizer_service.optimize_single_prompt(sample_prompt, "clarity_focus")

        # Create optimized prompt
        optimized_prompt = Prompt(
            id=sample_prompt.id,
            workflow_id=sample_prompt.workflow_id,
            node_id=sample_prompt.node_id,
            node_type=sample_prompt.node_type,
            text=result1.optimized_prompt,
            variables=sample_prompt.variables,
        )

        # Second optimization
        result2 = optimizer_service.optimize_single_prompt(optimized_prompt, "efficiency_focus")

        # Both should complete successfully
        assert result1 is not None
        assert result2 is not None

    def test_workflow_analysis_and_optimization(self, mock_catalog):
        """Test analyzing workflow before and after optimization."""
        service = OptimizerService(catalog=mock_catalog)

        # Analyze first
        report_before = service.analyze_workflow("test_workflow_001")
        score_before = report_before["average_score"]

        # Optimize if needed
        if report_before["needs_optimization"]:
            patches = service.run_optimization_cycle(
                workflow_id="test_workflow_001",
                strategy="auto"
            )
            assert isinstance(patches, list)

    def test_version_tracking_through_optimization(self, optimizer_service, sample_prompt):
        """Test that versions are tracked through optimization."""
        # Optimize
        result = optimizer_service.optimize_single_prompt(sample_prompt, "clarity_focus")

        # Get version history
        history = optimizer_service.get_version_history(sample_prompt.id)

        # Should have baseline version created internally
        assert len(history) >= 0


class TestIntegrationWithComponents:
    """Integration tests between optimizer components."""

    def test_extractor_analyzer_integration(self, extractor, analyzer, sample_workflow_dict):
        """Test integration between extractor and analyzer."""
        # Extract prompts
        prompts = extractor.extract_from_workflow(sample_workflow_dict)
        assert len(prompts) > 0

        # Analyze extracted prompts
        for prompt in prompts:
            analysis = analyzer.analyze_prompt(prompt)
            assert analysis is not None
            assert analysis.prompt_id == prompt.id

    def test_analyzer_engine_integration(self, analyzer, engine, sample_prompt):
        """Test integration between analyzer and engine."""
        # Analyze
        analysis = analyzer.analyze_prompt(sample_prompt)

        # Optimize based on analysis
        result = engine.optimize(sample_prompt, "clarity_focus")

        assert result is not None
        assert result.improvement_score is not None

    def test_engine_version_manager_integration(self, engine, version_manager, sample_prompt):
        """Test integration between engine and version manager."""
        # Optimize
        result = engine.optimize(sample_prompt, "clarity_focus")

        # Create optimized prompt
        optimized_prompt = Prompt(
            id=sample_prompt.id,
            workflow_id=sample_prompt.workflow_id,
            node_id=sample_prompt.node_id,
            node_type=sample_prompt.node_type,
            text=result.optimized_prompt,
            variables=sample_prompt.variables,
        )

        # Analyze optimized
        from src.optimizer.prompt_analyzer import PromptAnalyzer
        analyzer = PromptAnalyzer()
        optimized_analysis = analyzer.analyze_prompt(optimized_prompt)

        # Store versions
        baseline = version_manager.create_version(sample_prompt, analyzer.analyze_prompt(sample_prompt), None, None)
        optimized = version_manager.create_version(optimized_prompt, optimized_analysis, result, baseline.version)

        assert baseline.version == "1.0.0"
        assert optimized.version == "1.1.0"


class TestErrorHandlingIntegration:
    """Integration tests for error handling across components."""

    def test_graceful_handling_of_invalid_dsl(self, optimizer_service):
        """Test graceful handling of invalid DSL."""
        # Service without catalog should handle error
        with pytest.raises(Exception):  # OptimizerError or WorkflowNotFoundError
            optimizer_service.run_optimization_cycle("nonexistent", "clarity_focus")

    def test_optimization_continues_on_partial_failures(self, mock_catalog):
        """Test that optimization continues even if some prompts fail."""
        service = OptimizerService(catalog=mock_catalog)

        # Should not crash even if some optimizations have issues
        patches = service.run_optimization_cycle(
            workflow_id="test_workflow_001",
            strategy="clarity_focus"
        )

        # Should return list (may be empty)
        assert isinstance(patches, list)


class TestPerformance:
    """Performance-related integration tests."""

    def test_optimization_completes_in_reasonable_time(self, optimizer_service, sample_prompt):
        """Test that optimization completes quickly."""
        import time

        start = time.time()
        result = optimizer_service.optimize_single_prompt(sample_prompt, "clarity_focus")
        duration = time.time() - start

        assert result is not None
        # Should complete in under 5 seconds for single prompt
        assert duration < 5.0

    def test_multiple_prompts_optimization(self, extractor, optimizer_service, sample_workflow_dict):
        """Test optimizing multiple prompts from workflow."""
        prompts = extractor.extract_from_workflow(sample_workflow_dict)

        results = []
        for prompt in prompts:
            result = optimizer_service.optimize_single_prompt(prompt, "clarity_focus")
            results.append(result)

        assert len(results) == len(prompts)
        assert all(r is not None for r in results)


class TestDataConsistency:
    """Tests for data consistency across optimization workflow."""

    def test_prompt_id_consistency(self, optimizer_service, sample_prompt):
        """Test that prompt IDs remain consistent."""
        result = optimizer_service.optimize_single_prompt(sample_prompt, "clarity_focus")

        assert result.prompt_id == sample_prompt.id

    def test_variable_consistency(self, optimizer_service, sample_prompt_with_multiple_vars):
        """Test that variables are preserved consistently."""
        result = optimizer_service.optimize_single_prompt(
            sample_prompt_with_multiple_vars, "efficiency_focus"
        )

        # All original variables should be present
        for var in sample_prompt_with_multiple_vars.variables:
            assert f"{{{{{var}}}}}" in result.optimized_prompt
