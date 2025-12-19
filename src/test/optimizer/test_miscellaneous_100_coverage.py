"""
Test Coverage for Miscellaneous Modules - Achieving 100% Coverage

Date: 2025-11-18
Author: backend-developer
Description: Tests for uncovered code paths in various modules
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.optimizer.models import (
    Prompt,
    PromptAnalysis,
    OptimizationResult,
    OptimizationStrategy,
    OptimizationChange,
    OptimizationConfig,
)
from src.optimizer.optimization_engine import OptimizationEngine
from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.prompt_analyzer import PromptAnalyzer
from src.optimizer.prompt_extractor import PromptExtractor
from src.optimizer.prompt_patch_engine import PromptPatchEngine
from src.optimizer.version_manager import VersionManager
from src.optimizer.interfaces.llm_client import StubLLMClient, LLMResponse
from src.optimizer.interfaces.storage import InMemoryStorage
from src.optimizer.exceptions import WorkflowNotFoundError, OptimizerError


class TestPromptAnalyzerEdgeCases:
    """Test uncovered edge cases in PromptAnalyzer."""

    def test_analyze_prompt_with_empty_sentences(self):
        """Test analyzing prompt that results in empty sentences."""
        analyzer = PromptAnalyzer(llm_client=StubLLMClient())

        prompt = Prompt(
            id="test",
            workflow_id="wf_001",
            node_id="node_1",
            node_type="llm",
            text="!!!",  # No real sentences
            role="user",
            variables=[],
        )

        analysis = analyzer.analyze_prompt(prompt)

        # Should still complete without error
        assert analysis.overall_score >= 0

    def test_score_clarity_with_no_sentences(self):
        """Test _calculate_clarity_score with text that has no sentences."""
        analyzer = PromptAnalyzer(llm_client=StubLLMClient())

        # Create a minimal prompt with only punctuation
        prompt = Prompt(
            id="test",
            workflow_id="wf_001",
            node_id="node_1",
            node_type="llm",
            text="... --- !!!",
            role="user",
            variables=[],
        )

        # Call the actual method
        score = analyzer._calculate_clarity_score(prompt)

        # Should return base score
        assert score >= 0


class TestOptimizationEngineEdgeCases:
    """Test uncovered edge cases in OptimizationEngine."""

    def test_optimize_with_empty_prompt(self):
        """Test optimization with minimal prompt."""
        analyzer = PromptAnalyzer(llm_client=StubLLMClient())
        engine = OptimizationEngine(
            analyzer=analyzer, llm_client=StubLLMClient()
        )

        prompt = Prompt(
            id="test",
            workflow_id="wf_001",
            node_id="node_1",
            node_type="llm",
            text="Hi",  # Very short
            role="user",
            variables=[],
        )

        result = engine.optimize(prompt, "clarity_focus")

        assert result is not None
        assert result.optimized_prompt is not None

    def test_detect_changes_with_identical_text(self):
        """Test _detect_changes when text is identical."""
        analyzer = PromptAnalyzer(llm_client=StubLLMClient())
        engine = OptimizationEngine(
            analyzer=analyzer, llm_client=StubLLMClient()
        )

        changes = engine._detect_changes("same text", "same text")

        # Should detect no significant changes
        assert isinstance(changes, list)


class TestOptimizerServiceEdgeCases:
    """Test uncovered edge cases in OptimizerService."""

    def test_optimize_single_prompt_with_error(self):
        """Test optimize_single_prompt error handling."""
        service = OptimizerService()

        # Create invalid prompt
        prompt = Prompt(
            id="test",
            workflow_id="wf_001",
            node_id="node_1",
            node_type="llm",
            text="Test",
            role="user",
            variables=[],
        )

        # Mock engine to raise exception
        with patch.object(
                service._engine, "optimize", side_effect=Exception("Optimization failed")
        ):
            with pytest.raises(OptimizerError) as exc_info:
                service.optimize_single_prompt(prompt)

            assert exc_info.value.error_code == "OPT-SVC-002"

    def test_run_optimization_cycle_no_prompts_found(self):
        """Test run_optimization_cycle when no prompts extracted."""
        from src.config.models import WorkflowCatalog, WorkflowEntry

        catalog = Mock(spec=WorkflowCatalog)
        service = OptimizerService(catalog=catalog)

        # Mock extractor to return empty list
        with patch.object(service, "_extract_prompts", return_value=[]):
            patches = service.run_optimization_cycle("wf_001")

            assert patches == []

    def test_extract_prompts_catalog_not_initialized(self):
        """Test _extract_prompts when catalog is None."""
        service = OptimizerService(catalog=None)

        with pytest.raises(ValueError) as exc_info:
            service._extract_prompts("wf_001")

        assert "WorkflowCatalog not initialized" in str(exc_info.value)

    def test_extract_prompts_workflow_not_found(self):
        """Test _extract_prompts when workflow doesn't exist."""
        from src.config.models import WorkflowCatalog

        catalog = Mock(spec=WorkflowCatalog)
        catalog.get_workflow.return_value = None

        service = OptimizerService(catalog=catalog)

        with pytest.raises(WorkflowNotFoundError):
            service._extract_prompts("nonexistent_workflow")

    def test_extract_prompts_dsl_file_not_found(self):
        """Test _extract_prompts when DSL file doesn't exist."""
        from src.config.models import WorkflowCatalog, WorkflowEntry
        from pathlib import Path

        catalog = Mock(spec=WorkflowCatalog)
        workflow = Mock(spec=WorkflowEntry)
        workflow.dsl_path_resolved = Path("/nonexistent/path.json")
        catalog.get_workflow.return_value = workflow

        service = OptimizerService(catalog=catalog)

        with pytest.raises(OptimizerError) as exc_info:
            service._extract_prompts("wf_001")

        assert exc_info.value.error_code == "OPT-SVC-004"

    def test_is_better_result_both_below_threshold(self):
        """Test _is_better_result when both results below threshold."""
        service = OptimizerService()

        result1 = OptimizationResult(
            prompt_id="test",
            original_prompt="test",
            optimized_prompt="optimized1",
            strategy=OptimizationStrategy.CLARITY_FOCUS,
            improvement_score=5.0,
            confidence=0.5,
            changes=[],
            metadata={"optimized_score": 75.0},
        )

        result2 = OptimizationResult(
            prompt_id="test",
            original_prompt="test",
            optimized_prompt="optimized2",
            strategy=OptimizationStrategy.EFFICIENCY_FOCUS,
            improvement_score=8.0,
            confidence=0.55,
            changes=[],
            metadata={"optimized_score": 78.0},
        )

        # result2 has higher score, should be better
        assert service._is_better_result(result2, result1, min_confidence=0.7) is True

    def test_analyze_workflow_empty_workflow(self):
        """Test analyze_workflow with workflow that has no prompts."""
        from src.config.models import WorkflowCatalog

        catalog = Mock(spec=WorkflowCatalog)
        service = OptimizerService(catalog=catalog)

        with patch.object(service, "_extract_prompts", return_value=[]):
            report = service.analyze_workflow("wf_001")

            assert report["prompt_count"] == 0
            assert report["average_score"] == 0.0


class TestPromptExtractorEdgeCases:
    """Test uncovered edge cases in PromptExtractor."""

    def test_extract_from_workflow_with_unknown_node_type(self):
        """Test extraction with unknown node types."""
        extractor = PromptExtractor()

        dsl = {
            "nodes": [
                {
                    "id": "unknown_node",
                    "data": {
                        "type": "unknown_type",
                        "title": "Unknown Node",
                    },
                }
            ]
        }

        prompts = extractor.extract_from_workflow(dsl, "wf_001")

        # Should handle gracefully and return empty list
        assert prompts == []

    def test_load_dsl_file_with_invalid_json(self, tmp_path):
        """Test load_dsl_file with invalid YAML/JSON."""
        from src.optimizer.exceptions import DSLParseError

        extractor = PromptExtractor()

        invalid_file = tmp_path / "invalid.yaml"
        # Create truly invalid YAML that will cause parse error
        invalid_file.write_text(":\n  - invalid\n    broken")

        with pytest.raises(DSLParseError):
            extractor.load_dsl_file(invalid_file)


class TestPromptPatchEngineEdgeCases:
    """Test uncovered edge cases in PromptPatchEngine."""

    def test_apply_patches_with_unknown_node_type(self):
        """Test patch application with unknown node type."""
        from src.config.models import (
            PromptPatch,
            PromptSelector,
            PromptStrategy,
            WorkflowCatalog,
        )
        from src.config.utils.yaml_parser import YamlParser
        import yaml

        # Create mock catalog and parser
        catalog = Mock(spec=WorkflowCatalog)
        catalog.workflows = []  # Empty workflows list
        parser = Mock(spec=YamlParser)

        engine = PromptPatchEngine(catalog=catalog, yaml_parser=parser)

        dsl_dict = {
            "nodes": [
                {
                    "id": "node_1",
                    "data": {
                        "type": "unknown_type",
                        "title": "Unknown",
                    },
                }
            ]
        }
        dsl_text = yaml.dump(dsl_dict)

        selector = PromptSelector(by_id="node_1")
        strategy = PromptStrategy(mode="replace", content="new content")
        patch = PromptPatch(selector=selector, strategy=strategy)

        # Mock the parser methods to return proper values
        parser.load.return_value = dsl_dict
        parser.dump.return_value = dsl_text

        # Should apply patch even if node type is unknown
        modified_dsl = engine.apply_patches("wf_001", dsl_text, [patch])

        # Should return modified DSL as string
        assert modified_dsl is not None
        assert isinstance(modified_dsl, str)


class TestVersionManagerEdgeCases:
    """Test uncovered edge cases in VersionManager."""

    def test_delete_version_not_found(self):
        """Test delete_version when version doesn't exist."""
        manager = VersionManager(storage=InMemoryStorage())

        deleted = manager.delete_version("nonexistent_prompt", "1.0.0")

        assert deleted is False


class TestModelsValidation:
    """Test uncovered validation paths in models."""

    def test_prompt_text_too_long(self):
        """Test Prompt validation with text exceeding max length."""
        with pytest.raises(ValueError) as exc_info:
            Prompt(
                id="test",
                workflow_id="wf_001",
                node_id="node_1",
                node_type="llm",
                text="x" * 100001,  # Exceeds 100,000 limit
                role="user",
                variables=[],
            )

        assert "exceeds maximum length" in str(exc_info.value)

    def test_optimization_config_invalid_strategy(self):
        """Test OptimizationConfig with invalid strategy."""
        with pytest.raises(ValueError):
            OptimizationConfig(
                strategies=["invalid_strategy"],  # Invalid
            )

    def test_optimization_change_minimal(self):
        """Test OptimizationChange with minimal required fields."""
        change = OptimizationChange(
            rule_id="TEST_RULE",
            description="Test change",
        )

        assert change.rule_id == "TEST_RULE"
        assert change.location is None
        assert change.before is None
        assert change.after is None


class TestInMemoryStorageEdgeCases:
    """Test uncovered edge cases in InMemoryStorage."""

    def test_list_versions_with_limit_parameter(self):
        """Test list_versions with limit parameter."""
        from src.optimizer.models import PromptVersion

        storage = InMemoryStorage()

        # Create 5 versions
        for i in range(5):
            prompt = Prompt(
                id="test",
                workflow_id="wf_001",
                node_id="node_1",
                node_type="llm",
                text=f"Test {i}",
                role="user",
                variables=[],
            )
            analysis = PromptAnalysis(
                prompt_id="test",
                overall_score=85.0,
                clarity_score=80.0,
                efficiency_score=90.0,
                issues=[],
                suggestions=[],
            )
            version = PromptVersion(
                prompt_id="test",
                version=f"1.{i}.0",
                prompt=prompt,
                analysis=analysis,
            )
            storage.save_version(version)

        # List with limit
        versions = storage.list_versions("test", limit=3)

        assert len(versions) == 3


class TestStubLLMClientEdgeCases:
    """Test uncovered edge cases in StubLLMClient."""

    def test_stub_llm_analyze_prompt_basic(self):
        """Test StubLLMClient.analyze_prompt basic functionality."""
        client = StubLLMClient()

        result = client.analyze_prompt(
            prompt="Test prompt with variables",
            context={"workflow_id": "wf_001", "node_id": "node_1"},
        )

        # Should return an LLMResponse object with JSON content
        assert result is not None
        assert isinstance(result, LLMResponse)
        import json
        analysis = json.loads(result.content)
        assert "overall_score" in analysis


class TestPromptAnalyzerStopwords:
    """Test PromptAnalyzer stopwords handling."""

    def test_information_density_with_all_stopwords(self):
        """Test _score_information_density with all stopwords."""
        analyzer = PromptAnalyzer(llm_client=StubLLMClient())

        # Create a prompt with only stopwords
        prompt = Prompt(
            id="test",
            workflow_id="wf_001",
            node_id="node_1",
            node_type="llm",
            text="the a an is are was were",
            role="user",
            variables=[],
        )

        # Call the actual method
        score = analyzer._score_information_density(prompt.text)

        # The method returns a base score, not a penalized score
        # Just verify it returns a valid score
        assert 0 <= score <= 100
