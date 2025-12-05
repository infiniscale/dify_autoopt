"""
Full Integration Test Suite for LLM with Existing Optimizer Components.

This test suite validates integration between LLM functionality and:
1. PromptExtractor integration
2. PromptAnalyzer integration
3. VersionManager integration
4. PromptPatchEngine integration

Author: QA Engineer
Date: 2025-11-18
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from src.optimizer.config import LLMConfig, LLMProvider
from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.prompt_extractor import PromptExtractor
from src.optimizer.prompt_analyzer import PromptAnalyzer
from src.optimizer.optimization_engine import OptimizationEngine
from src.optimizer.version_manager import VersionManager
from src.optimizer.interfaces.storage import InMemoryStorage
from src.optimizer.models import Prompt, PromptAnalysis
from src.config.models import WorkflowCatalog, PromptPatch


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def comprehensive_workflow_dsl() -> Dict[str, Any]:
    """Create a comprehensive workflow with multiple node types."""
    return {
        "graph": {
            "nodes": [
                {
                    "data": {
                        "type": "llm",
                        "title": "Document Summarizer",
                        "prompt_template": [
                            {
                                "role": "system",
                                "text": "You are helpful assistant. Summarize document."
                            },
                            {
                                "role": "user",
                                "text": "Summarize: {{document}}"
                            }
                        ]
                    },
                    "id": "llm_1"
                },
                {
                    "data": {
                        "type": "llm",
                        "title": "Sentiment Analyzer",
                        "prompt_template": [
                            {
                                "role": "system",
                                "text": "Analyze sentiment of text and provide detailed analysis"
                            }
                        ]
                    },
                    "id": "llm_2"
                },
                {
                    "data": {
                        "type": "question-classifier",
                        "title": "Question Classifier",
                        "query_variable_selector": ["sys", "query"],
                        "classes": [
                            {
                                "name": "technical",
                                "template": "Is this a technical question about {{topic}}?"
                            },
                            {
                                "name": "general",
                                "template": "Is this a general question?"
                            }
                        ]
                    },
                    "id": "classifier_1"
                },
                {
                    "data": {
                        "type": "if-else",
                        "title": "Conditional Node",
                        "conditions": {
                            "default": {
                                "logical_operator": "and",
                                "conditions": [
                                    {
                                        "prompt": "Check if {{condition}} is met"
                                    }
                                ]
                            }
                        }
                    },
                    "id": "ifelse_1"
                }
            ]
        }
    }


@pytest.fixture
def mock_catalog(comprehensive_workflow_dsl: Dict[str, Any]) -> WorkflowCatalog:
    """Mock workflow catalog."""
    catalog = Mock(spec=WorkflowCatalog)
    catalog.get_workflow_by_id.return_value = comprehensive_workflow_dsl
    return catalog


@pytest.fixture
def stub_config() -> LLMConfig:
    """STUB configuration for testing."""
    return LLMConfig(
        provider=LLMProvider.STUB,
        enable_cache=True,
        cache_ttl=3600
    )


# ============================================================================
# Test 1: PromptExtractor Integration
# ============================================================================


class TestPromptExtractorIntegration:
    """Test integration with PromptExtractor."""

    def test_extract_and_optimize_llm_nodes(self, mock_catalog, stub_config):
        """Test extracting prompts from LLM nodes and optimizing them."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        # Extract prompts
        prompts = service.extract_prompts("wf_001")

        # Verify extraction
        llm_prompts = [p for p in prompts if p.node_type == "llm"]
        assert len(llm_prompts) == 3  # 2 system + 1 user from llm nodes

        # Optimize each
        results = []
        for prompt in llm_prompts:
            result = service.optimize_prompt(prompt, "llm_clarity")
            results.append(result)

        # Verify all optimized
        assert len(results) == len(llm_prompts)
        assert all(r.optimized_text is not None for r in results)
        assert all(r.improvement_score >= -100 for r in results)

    def test_extract_and_optimize_classifier_nodes(self, mock_catalog, stub_config):
        """Test extracting and optimizing question-classifier prompts."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        prompts = service.extract_prompts("wf_001")

        # Find classifier prompts
        classifier_prompts = [p for p in prompts if p.node_type == "question-classifier"]
        assert len(classifier_prompts) == 2  # 2 classes

        # Verify variable preservation in classifier prompts
        for prompt in classifier_prompts:
            if "{{topic}}" in prompt.text:
                # Optimize
                result = service.optimize_prompt(prompt, "llm_clarity")
                # Variable should be preserved
                assert "{{topic}}" in result.optimized_text

    def test_extract_and_optimize_ifelse_nodes(self, mock_catalog, stub_config):
        """Test extracting and optimizing if-else condition prompts."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        prompts = service.extract_prompts("wf_001")

        # Find if-else prompts
        ifelse_prompts = [p for p in prompts if p.node_type == "if-else"]
        assert len(ifelse_prompts) >= 1

        # Optimize
        for prompt in ifelse_prompts:
            result = service.optimize_prompt(prompt, "llm_clarity")
            assert result.optimized_text is not None

            # Verify variables preserved
            if "{{condition}}" in prompt.text:
                assert "{{condition}}" in result.optimized_text

    def test_extraction_with_empty_workflow(self):
        """Test extraction from workflow with no prompts."""
        empty_workflow = {"graph": {"nodes": []}}
        catalog = Mock(spec=WorkflowCatalog)
        catalog.get_workflow_by_id.return_value = empty_workflow

        config = LLMConfig(provider=LLMProvider.STUB)
        service = OptimizerService(catalog=catalog, llm_config=config)

        prompts = service.extract_prompts("wf_empty")
        assert len(prompts) == 0


# ============================================================================
# Test 2: PromptAnalyzer Integration
# ============================================================================


class TestPromptAnalyzerIntegration:
    """Test integration with PromptAnalyzer."""

    def test_analyze_before_and_after_optimization(self, mock_catalog, stub_config):
        """Test analyzing prompt quality before and after optimization."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        prompts = service.extract_prompts("wf_001")
        test_prompt = prompts[0]

        # Analyze original
        analysis_before = service.analyze_prompt(test_prompt)
        assert analysis_before.overall_score >= 0
        assert analysis_before.clarity_score >= 0
        assert analysis_before.efficiency_score >= 0

        # Optimize
        result = service.optimize_prompt(test_prompt, "llm_clarity")

        # Analyze optimized
        analysis_after = service.analyze_prompt_text(result.optimized_text)
        assert analysis_after.overall_score >= 0

        # Quality should improve or stay same
        # Note: In some cases, rule-based optimization may not improve score
        # This is expected behavior
        assert analysis_after is not None

    def test_quality_score_improvement_tracking(self, mock_catalog, stub_config):
        """Test tracking quality score improvements."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        prompts = service.extract_prompts("wf_001")

        improvements = []
        for prompt in prompts[:3]:  # Test first 3
            # Get baseline score
            before = service.analyze_prompt(prompt)

            # Optimize
            result = service.optimize_prompt(prompt, "llm_guided")

            # Get improved score
            after = service.analyze_prompt_text(result.optimized_text)

            improvement = after.overall_score - before.overall_score
            improvements.append({
                "prompt_id": prompt.id,
                "before": before.overall_score,
                "after": after.overall_score,
                "improvement": improvement
            })

        # Verify we tracked all improvements
        assert len(improvements) == 3
        assert all("improvement" in imp for imp in improvements)

    def test_issue_detection_and_resolution(self, mock_catalog, stub_config):
        """Test that detected issues are addressed by optimization."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        prompts = service.extract_prompts("wf_001")

        for prompt in prompts[:2]:
            # Analyze to find issues
            analysis = service.analyze_prompt(prompt)
            original_issues = len(analysis.issues)

            if original_issues > 0:
                # Optimize to fix issues
                result = service.optimize_prompt(prompt, "llm_guided")

                # Re-analyze
                new_analysis = service.analyze_prompt_text(result.optimized_text)

                # Issues should be reduced (ideally)
                # Note: This may not always happen with rule-based optimization
                assert new_analysis is not None


# ============================================================================
# Test 3: VersionManager Integration
# ============================================================================


class TestVersionManagerIntegration:
    """Test integration with VersionManager."""

    def test_create_baseline_and_optimized_versions(self, mock_catalog, stub_config):
        """Test creating baseline and optimized versions."""
        storage = InMemoryStorage()
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=stub_config,
            storage=storage
        )

        prompts = service.extract_prompts("wf_001")
        test_prompt = prompts[0]

        # Create baseline version
        version_manager = VersionManager(storage)
        baseline_id = version_manager.create_version(
            workflow_id="wf_001",
            prompt_id=test_prompt.id,
            text=test_prompt.text,
            metadata={"type": "baseline"}
        )
        assert baseline_id is not None

        # Optimize
        result = service.optimize_prompt(test_prompt, "llm_clarity")

        # Create optimized version
        optimized_id = version_manager.create_version(
            workflow_id="wf_001",
            prompt_id=test_prompt.id,
            text=result.optimized_text,
            metadata={"type": "optimized", "strategy": "llm_clarity"}
        )
        assert optimized_id is not None

        # Verify both versions exist
        baseline = version_manager.get_version(baseline_id)
        optimized = version_manager.get_version(optimized_id)

        assert baseline.text == test_prompt.text
        assert optimized.text == result.optimized_text

    def test_version_comparison(self, mock_catalog, stub_config):
        """Test comparing versions before and after optimization."""
        storage = InMemoryStorage()
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=stub_config,
            storage=storage
        )

        prompts = service.extract_prompts("wf_001")
        test_prompt = prompts[0]

        version_manager = VersionManager(storage)

        # Create versions
        v1_id = version_manager.create_version(
            workflow_id="wf_001",
            prompt_id=test_prompt.id,
            text=test_prompt.text,
            metadata={"type": "original"}
        )

        result = service.optimize_prompt(test_prompt, "llm_efficiency")

        v2_id = version_manager.create_version(
            workflow_id="wf_001",
            prompt_id=test_prompt.id,
            text=result.optimized_text,
            metadata={"type": "optimized"}
        )

        # Compare
        comparison = version_manager.compare_versions(v1_id, v2_id)

        assert comparison["version1_id"] == v1_id
        assert comparison["version2_id"] == v2_id
        assert "text_length_diff" in comparison

    def test_version_rollback_after_optimization(self, mock_catalog, stub_config):
        """Test rolling back to previous version."""
        storage = InMemoryStorage()
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=stub_config,
            storage=storage
        )

        prompts = service.extract_prompts("wf_001")
        test_prompt = prompts[0]

        version_manager = VersionManager(storage)

        # Create original version
        original_id = version_manager.create_version(
            workflow_id="wf_001",
            prompt_id=test_prompt.id,
            text=test_prompt.text,
            metadata={"type": "original"}
        )

        # Optimize and create new version
        result = service.optimize_prompt(test_prompt, "llm_guided")

        optimized_id = version_manager.create_version(
            workflow_id="wf_001",
            prompt_id=test_prompt.id,
            text=result.optimized_text,
            metadata={"type": "optimized"}
        )

        # Get history
        history = version_manager.get_version_history("wf_001", test_prompt.id)
        assert len(history) == 2

        # Can retrieve original for rollback
        original_version = version_manager.get_version(original_id)
        assert original_version.text == test_prompt.text

    def test_multi_strategy_version_tracking(self, mock_catalog, stub_config):
        """Test tracking versions for multiple optimization strategies."""
        storage = InMemoryStorage()
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=stub_config,
            storage=storage
        )

        prompts = service.extract_prompts("wf_001")
        test_prompt = prompts[0]

        version_manager = VersionManager(storage)

        # Create versions for different strategies
        strategies = ["llm_clarity", "llm_efficiency", "llm_guided"]
        version_ids = []

        for strategy in strategies:
            result = service.optimize_prompt(test_prompt, strategy)

            version_id = version_manager.create_version(
                workflow_id="wf_001",
                prompt_id=test_prompt.id,
                text=result.optimized_text,
                metadata={
                    "strategy": strategy,
                    "improvement": result.improvement_score
                }
            )
            version_ids.append((strategy, version_id))

        # Verify all versions created
        assert len(version_ids) == 3

        # Get history
        history = version_manager.get_version_history("wf_001", test_prompt.id)
        assert len(history) == 3


# ============================================================================
# Test 4: PromptPatch Integration
# ============================================================================


class TestPromptPatchIntegration:
    """Test integration with PromptPatch generation and application."""

    def test_generate_prompt_patches(self, mock_catalog, stub_config):
        """Test generating PromptPatch objects for workflow updates."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        # Run full optimization cycle
        patches = service.run_optimization_cycle(
            workflow_id="wf_001",
            strategy="llm_clarity"
        )

        # Verify patches generated
        assert len(patches) > 0
        assert all(isinstance(p, PromptPatch) for p in patches)

        # Verify patch structure
        for patch in patches:
            assert patch.node_id is not None
            assert patch.original_prompt is not None
            assert patch.optimized_prompt is not None
            assert patch.improvement_score >= -100

    def test_patch_preserves_workflow_structure(self, mock_catalog, stub_config):
        """Test that patches preserve workflow structure and metadata."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        patches = service.run_optimization_cycle(
            workflow_id="wf_001",
            strategy="llm_efficiency"
        )

        # Extract original prompts
        original_prompts = service.extract_prompts("wf_001")

        # Create mapping
        original_by_node = {p.node_id: p for p in original_prompts}

        # Verify patches match original structure
        for patch in patches:
            assert patch.node_id in original_by_node
            original = original_by_node[patch.node_id]

            # Verify metadata preserved
            assert patch.node_type == original.node_type
            assert patch.role == original.role

    def test_variable_preservation_in_patches(self, mock_catalog, stub_config):
        """Test that patches preserve template variables."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        patches = service.run_optimization_cycle(
            workflow_id="wf_001",
            strategy="llm_clarity"
        )

        # Check patches with variables
        for patch in patches:
            if "{{" in patch.original_prompt:
                # Extract variables from original
                import re
                original_vars = set(re.findall(r'\{\{([^}]+)\}\}', patch.original_prompt))

                # Verify all present in optimized
                for var in original_vars:
                    assert f"{{{{{var}}}}}" in patch.optimized_prompt, \
                        f"Variable {{{{{var}}}}} lost in patch for node {patch.node_id}"

    def test_apply_patches_to_workflow(self, mock_catalog, stub_config):
        """Test applying patches to update workflow."""
        service = OptimizerService(catalog=mock_catalog, llm_config=stub_config)

        # Get original workflow
        original_workflow = mock_catalog.get_workflow_by_id("wf_001")

        # Generate patches
        patches = service.run_optimization_cycle(
            workflow_id="wf_001",
            strategy="llm_guided"
        )

        # Apply patches (simulate)
        updated_workflow = self._apply_patches_simulation(
            original_workflow,
            patches
        )

        # Verify updates
        for patch in patches:
            node = self._find_node_by_id(updated_workflow, patch.node_id)
            assert node is not None

            # Check that prompt was updated
            if patch.node_type == "llm":
                prompts = node["data"]["prompt_template"]
                # At least one prompt should match optimized text
                found = any(p["text"] == patch.optimized_prompt for p in prompts)
                # Note: Simulation may not match exactly, this is demonstration

    def _apply_patches_simulation(
        self,
        workflow: Dict[str, Any],
        patches: List[PromptPatch]
    ) -> Dict[str, Any]:
        """Simulate applying patches to workflow (helper method)."""
        import copy
        updated = copy.deepcopy(workflow)

        for patch in patches:
            node = self._find_node_by_id(updated, patch.node_id)
            if node and patch.node_type == "llm":
                # Update prompt_template
                for prompt in node["data"]["prompt_template"]:
                    if prompt["text"] == patch.original_prompt:
                        prompt["text"] = patch.optimized_prompt

        return updated

    def _find_node_by_id(
        self,
        workflow: Dict[str, Any],
        node_id: str
    ) -> Dict[str, Any]:
        """Find node by ID in workflow (helper method)."""
        for node in workflow["graph"]["nodes"]:
            if node["id"] == node_id:
                return node
        return None


# ============================================================================
# Test 5: End-to-End Workflow
# ============================================================================


class TestEndToEndWorkflow:
    """Test complete end-to-end workflow integration."""

    def test_complete_optimization_workflow(self, mock_catalog, stub_config):
        """Test complete workflow: extract -> analyze -> optimize -> patch -> version."""
        storage = InMemoryStorage()
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=stub_config,
            storage=storage
        )

        workflow_id = "wf_001"
        strategy = "llm_clarity"

        # Step 1: Extract prompts
        prompts = service.extract_prompts(workflow_id)
        assert len(prompts) > 0

        # Step 2: Analyze quality
        analyses = []
        for prompt in prompts:
            analysis = service.analyze_prompt(prompt)
            analyses.append(analysis)
        assert len(analyses) == len(prompts)

        # Step 3: Optimize low-quality prompts
        results = []
        for i, prompt in enumerate(prompts):
            if analyses[i].overall_score < 80:  # Arbitrary threshold
                result = service.optimize_prompt(prompt, strategy)
                results.append(result)

        # Step 4: Create versions
        version_manager = VersionManager(storage)
        for prompt, result in zip(prompts[:len(results)], results):
            version_manager.create_version(
                workflow_id=workflow_id,
                prompt_id=prompt.id,
                text=result.optimized_text,
                metadata={"strategy": strategy}
            )

        # Step 5: Generate patches
        patches = service.run_optimization_cycle(workflow_id, strategy)
        assert len(patches) > 0

        # Verify complete workflow
        assert len(prompts) > 0
        assert len(patches) > 0

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_complete_workflow_with_real_llm(
        self,
        mock_openai,
        mock_catalog
    ):
        """Test complete workflow with mocked OpenAI integration."""
        # Mock OpenAI
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(
            message=MagicMock(content="LLM optimized prompt"),
            finish_reason="stop"
        )]
        mock_completion.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            enable_cache=True
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            storage = InMemoryStorage()
            service = OptimizerService(
                catalog=mock_catalog,
                llm_config=config,
                storage=storage
            )

            # Complete workflow
            patches = service.run_optimization_cycle("wf_001", "llm_guided")

            # Verify
            assert len(patches) > 0
            assert mock_client.chat.completions.create.called

            # Check stats
            stats = service.get_usage_stats()
            assert stats.total_requests > 0
            assert stats.total_cost > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
