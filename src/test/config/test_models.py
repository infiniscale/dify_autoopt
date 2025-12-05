"""
Unit Tests for Pydantic Models

Tests model properties, methods, and validators.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config.models import WorkflowCatalog, WorkflowEntry, NodeMeta, TestPlan, PromptStrategy


class TestWorkflowEntryProperties:
    """Tests for WorkflowEntry properties and methods"""

    def test_dsl_path_resolved_with_relative_path(self):
        """Test dsl_path_resolved with relative path"""
        entry = WorkflowEntry(
            id="test_workflow",
            label="Test Workflow",
            type="workflow",
            dsl_path=Path("relative/path/workflow.yaml"),
            nodes=[]
        )

        resolved = entry.dsl_path_resolved
        assert resolved.is_absolute()

    def test_dsl_path_resolved_with_absolute_path(self):
        """Test dsl_path_resolved with absolute path"""
        with TemporaryDirectory() as tmp_dir:
            abs_path = Path(tmp_dir) / "workflow.yaml"
            abs_path.touch()

            entry = WorkflowEntry(
                id="test_workflow",
                label="Test Workflow",
                type="workflow",
                dsl_path=abs_path,
                nodes=[]
            )

            resolved = entry.dsl_path_resolved
            assert resolved == abs_path
            assert resolved.is_absolute()


class TestWorkflowCatalogMethods:
    """Tests for WorkflowCatalog methods"""

    def test_get_workflow_found(self):
        """Test get_workflow returns correct workflow"""
        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[
                WorkflowEntry(
                    id="workflow_1",
                    label="Workflow 1",
                    type="workflow",
                    dsl_path=Path("workflow1.yaml"),
                    nodes=[]
                ),
                WorkflowEntry(
                    id="workflow_2",
                    label="Workflow 2",
                    type="chatflow",
                    dsl_path=Path("workflow2.yaml"),
                    nodes=[]
                )
            ]
        )

        result = catalog.get_workflow("workflow_1")
        assert result is not None
        assert result.id == "workflow_1"
        assert result.label == "Workflow 1"

    def test_get_workflow_not_found(self):
        """Test get_workflow returns None when workflow not found"""
        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[
                WorkflowEntry(
                    id="workflow_1",
                    label="Workflow 1",
                    type="workflow",
                    dsl_path=Path("workflow1.yaml"),
                    nodes=[]
                )
            ]
        )

        result = catalog.get_workflow("nonexistent")
        assert result is None

    def test_get_workflows_by_tag_found(self):
        """Test get_workflows_by_tag returns matching workflows"""
        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[
                WorkflowEntry(
                    id="workflow_1",
                    label="Workflow 1",
                    type="workflow",
                    dsl_path=Path("workflow1.yaml"),
                    nodes=[],
                    tags=["production", "critical"]
                ),
                WorkflowEntry(
                    id="workflow_2",
                    label="Workflow 2",
                    type="chatflow",
                    dsl_path=Path("workflow2.yaml"),
                    nodes=[],
                    tags=["development"]
                ),
                WorkflowEntry(
                    id="workflow_3",
                    label="Workflow 3",
                    type="workflow",
                    dsl_path=Path("workflow3.yaml"),
                    nodes=[],
                    tags=["production"]
                )
            ]
        )

        results = catalog.get_workflows_by_tag("production")
        assert len(results) == 2
        assert all(wf.id in ["workflow_1", "workflow_3"] for wf in results)

    def test_get_workflows_by_tag_not_found(self):
        """Test get_workflows_by_tag returns empty list when tag not found"""
        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[
                WorkflowEntry(
                    id="workflow_1",
                    label="Workflow 1",
                    type="workflow",
                    dsl_path=Path("workflow1.yaml"),
                    nodes=[],
                    tags=["production"]
                )
            ]
        )

        results = catalog.get_workflows_by_tag("nonexistent")
        assert results == []


class TestTestPlanMethods:
    """Tests for TestPlan methods"""

    def test_get_dataset_found(self):
        """Test get_dataset returns correct dataset"""
        from src.config.models import Dataset, TestDataConfig, ExecutionPolicy, RateLimit

        plan = TestPlan(
            meta={"plan_id": "test_plan"},
            workflows=[],
            test_data=TestDataConfig(
                datasets=[
                    Dataset(name="dataset_1", scenario="normal"),
                    Dataset(name="dataset_2", scenario="boundary")
                ]
            ),
            execution=ExecutionPolicy(
                concurrency=1,
                rate_control=RateLimit(per_minute=10, burst=5)
            )
        )

        result = plan.get_dataset("dataset_1")
        assert result is not None
        assert result.name == "dataset_1"
        assert result.scenario == "normal"

    def test_get_dataset_not_found(self):
        """Test get_dataset returns None when dataset not found"""
        from src.config.models import Dataset, TestDataConfig, ExecutionPolicy, RateLimit

        plan = TestPlan(
            meta={"plan_id": "test_plan"},
            workflows=[],
            test_data=TestDataConfig(
                datasets=[
                    Dataset(name="dataset_1", scenario="normal")
                ]
            ),
            execution=ExecutionPolicy(
                concurrency=1,
                rate_control=RateLimit(per_minute=10, burst=5)
            )
        )

        result = plan.get_dataset("nonexistent")
        assert result is None


class TestPromptStrategyValidation:
    """Tests for PromptStrategy validation"""

    def test_validate_mode_invalid(self):
        """Test that invalid mode raises ValueError"""
        from src.config.models import PromptTemplate

        with pytest.raises(ValueError, match="Invalid mode"):
            PromptStrategy(
                mode="invalid_mode",
                content="test content"
            )

    def test_validate_mode_valid_replace(self):
        """Test that valid mode 'replace' is accepted"""
        strategy = PromptStrategy(
            mode="replace",
            content="test content"
        )
        assert strategy.mode == "replace"

    def test_validate_mode_valid_prepend(self):
        """Test that valid mode 'prepend' is accepted"""
        strategy = PromptStrategy(
            mode="prepend",
            content="test content"
        )
        assert strategy.mode == "prepend"

    def test_validate_mode_valid_append(self):
        """Test that valid mode 'append' is accepted"""
        strategy = PromptStrategy(
            mode="append",
            content="test content"
        )
        assert strategy.mode == "append"

    def test_validate_mode_valid_template(self):
        """Test that valid mode 'template' is accepted"""
        from src.config.models import PromptTemplate

        strategy = PromptStrategy(
            mode="template",
            template=PromptTemplate(inline="test template")
        )
        assert strategy.mode == "template"


class TestEnvConfigPathValidation:
    """Tests for EnvConfig path validation"""

    def test_validate_paths_with_non_dict_input(self):
        """Test that validate_paths handles non-dict input by returning it as-is"""
        from src.config.models import EnvConfig

        # Test the validator directly by calling it with non-dict input
        # This tests line 58: return value
        result = EnvConfig.validate_paths("not_a_dict")
        assert result == "not_a_dict"

    def test_validate_paths_with_dict_input(self):
        """Test that validate_paths converts string paths to Path objects"""
        from src.config.models import EnvConfig
        from pathlib import Path

        # Test with dict containing string paths
        input_dict = {
            "output": "path/to/output",
            "input": "path/to/input",
            "already_path": Path("existing/path")
        }

        result = EnvConfig.validate_paths(input_dict)

        assert isinstance(result, dict)
        assert isinstance(result["output"], Path)
        assert isinstance(result["input"], Path)
        assert isinstance(result["already_path"], Path)
        assert result["output"] == Path("path/to/output")


class TestDifyConfigUrlValidation:
    """Tests for DifyConfig URL validation"""

    def test_validate_url_invalid_protocol(self):
        """Test that invalid URL protocol raises ValueError"""
        from src.config.models import DifyConfig, AuthConfig, RateLimit
        from pydantic import ValidationError

        # This tests line 37: raise ValueError for invalid URL
        with pytest.raises(ValidationError, match="Invalid URL format"):
            DifyConfig(
                base_url="ftp://invalid.com",
                auth=AuthConfig(primary_token="token"),
                rate_limits=RateLimit(per_minute=60)
            )

    def test_validate_url_valid_https(self):
        """Test that valid HTTPS URL is accepted"""
        from src.config.models import DifyConfig, AuthConfig, RateLimit

        config = DifyConfig(
            base_url="https://api.dify.ai",
            auth=AuthConfig(primary_token="token"),
            rate_limits=RateLimit(per_minute=60)
        )
        assert config.base_url == "https://api.dify.ai"

    def test_validate_url_valid_http(self):
        """Test that valid HTTP URL is accepted"""
        from src.config.models import DifyConfig, AuthConfig, RateLimit

        config = DifyConfig(
            base_url="http://localhost:8080",
            auth=AuthConfig(primary_token="token"),
            rate_limits=RateLimit(per_minute=60)
        )
        assert config.base_url == "http://localhost:8080"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
