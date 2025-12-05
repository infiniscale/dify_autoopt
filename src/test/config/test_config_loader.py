"""
Unit Tests for ConfigLoader

Tests configuration file loading with environment variable expansion.
"""

import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config.loaders import ConfigLoader, FileSystemReader
from src.config.models import EnvConfig, WorkflowCatalog, TestPlan
from src.config.utils.exceptions import ConfigurationError


class TestFileSystemReader:
    """Tests for FileSystemReader"""

    def test_read_yaml_success(self, tmp_path):
        """Test successful YAML file reading"""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value\nnumber: 42")

        reader = FileSystemReader()
        data = reader.read_yaml(yaml_file)

        assert data == {"key": "value", "number": 42}

    def test_read_yaml_file_not_found(self):
        """Test reading non-existent file raises ConfigurationError"""
        reader = FileSystemReader()

        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            reader.read_yaml(Path("nonexistent.yaml"))

    def test_read_yaml_invalid_syntax(self, tmp_path):
        """Test reading invalid YAML raises ConfigurationError"""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("key: value\n  bad: [unclosed")  # Actually invalid YAML

        reader = FileSystemReader()

        with pytest.raises(ConfigurationError, match="Invalid YAML format"):
            reader.read_yaml(yaml_file)

    def test_read_yaml_non_dict_root(self, tmp_path):
        """Test reading YAML with non-dict root raises error"""
        yaml_file = tmp_path / "list.yaml"
        yaml_file.write_text("- item1\n- item2")

        reader = FileSystemReader()

        with pytest.raises(ConfigurationError, match="must contain a dictionary"):
            reader.read_yaml(yaml_file)

    def test_read_yaml_empty_file(self, tmp_path):
        """Test reading empty YAML file returns empty dict"""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        reader = FileSystemReader()
        data = reader.read_yaml(yaml_file)

        assert data == {}


class TestConfigLoaderEnvExpansion:
    """Tests for environment variable expansion"""

    def test_expand_env_vars_string_replacement(self):
        """Test environment variable replacement in strings"""
        os.environ["TEST_VAR"] = "test_value"

        loader = ConfigLoader()
        data = {"key": "${TEST_VAR}"}
        result = loader._expand_env_vars(data)

        assert result == {"key": "test_value"}
        del os.environ["TEST_VAR"]

    def test_expand_env_vars_nested_dict(self):
        """Test env var expansion in nested dictionaries"""
        os.environ["NESTED_VAR"] = "nested_value"

        loader = ConfigLoader()
        data = {
            "level1": {
                "level2": {
                    "key": "${NESTED_VAR}"
                }
            }
        }
        result = loader._expand_env_vars(data)

        assert result["level1"]["level2"]["key"] == "nested_value"
        del os.environ["NESTED_VAR"]

    def test_expand_env_vars_in_list(self):
        """Test env var expansion in lists"""
        os.environ["LIST_VAR"] = "list_value"

        loader = ConfigLoader()
        data = {"items": ["${LIST_VAR}", "static", "${LIST_VAR}"]}
        result = loader._expand_env_vars(data)

        assert result["items"] == ["list_value", "static", "list_value"]
        del os.environ["LIST_VAR"]

    def test_expand_env_vars_missing_var(self):
        """Test that missing env vars are left as-is"""
        loader = ConfigLoader()
        data = {"key": "${NONEXISTENT_VAR}"}
        result = loader._expand_env_vars(data)

        assert result == {"key": "${NONEXISTENT_VAR}"}

    def test_expand_env_vars_multiple_in_string(self):
        """Test multiple env vars in single string"""
        os.environ["VAR1"] = "hello"
        os.environ["VAR2"] = "world"

        loader = ConfigLoader()
        data = {"message": "${VAR1} ${VAR2}!"}
        result = loader._expand_env_vars(data)

        assert result == {"message": "hello world!"}
        del os.environ["VAR1"]
        del os.environ["VAR2"]

    def test_expand_env_vars_non_string_types(self):
        """Test that non-string types are preserved"""
        loader = ConfigLoader()
        data = {
            "number": 42,
            "boolean": True,
            "none": None,
            "float": 3.14
        }
        result = loader._expand_env_vars(data)

        assert result == data


class TestConfigLoaderLoadEnv:
    """Tests for load_env"""

    def test_load_env_success(self, tmp_path):
        """Test successful env config loading"""
        config_file = tmp_path / "env.yaml"
        config_file.write_text("""
meta:
  version: "1.0"
  environment: "test"
dify:
  base_url: "https://api.dify.ai"
  auth:
    primary_token: "test_token"
  rate_limits:
    per_minute: 60
    burst: 10
model_evaluator:
  provider: "openai"
  model_name: "gpt-4"
io_paths: {}
logging:
  level: "INFO"
""")

        loader = ConfigLoader()
        env = loader.load_env(config_file)

        assert isinstance(env, EnvConfig)
        assert env.meta["version"] == "1.0"
        assert env.dify.base_url == "https://api.dify.ai"

    def test_load_env_with_env_vars(self, tmp_path):
        """Test env config loading with environment variable expansion"""
        os.environ["DIFY_TOKEN"] = "secret_token_123"

        config_file = tmp_path / "env.yaml"
        config_file.write_text("""
meta:
  version: "1.0"
dify:
  base_url: "https://api.dify.ai"
  auth:
    primary_token: "${DIFY_TOKEN}"
  rate_limits:
    per_minute: 60
model_evaluator:
  provider: "openai"
  model_name: "gpt-4"
io_paths: {}
logging: {}
""")

        loader = ConfigLoader()
        env = loader.load_env(config_file)

        assert env.dify.auth.primary_token.get_secret_value() == "secret_token_123"
        del os.environ["DIFY_TOKEN"]

    def test_load_env_validation_error(self, tmp_path):
        """Test load_env raises ConfigurationError on validation failure"""
        config_file = tmp_path / "invalid_env.yaml"
        config_file.write_text("""
meta:
  version: "1.0"
dify:
  base_url: "https://api.dify.ai"
  # Missing required auth field
""")

        loader = ConfigLoader()

        with pytest.raises(ConfigurationError, match="Failed to validate env_config"):
            loader.load_env(config_file)


class TestConfigLoaderLoadCatalog:
    """Tests for load_catalog"""

    def test_load_catalog_success(self, tmp_path):
        """Test successful catalog loading"""
        catalog_file = tmp_path / "catalog.yaml"
        catalog_file.write_text("""
meta:
  version: "1.0"
  source: "test"
workflows:
  - id: "workflow_1"
    label: "Test Workflow"
    type: "chatflow"
    dsl_path: "workflows/test.yaml"
    nodes: []
""")

        loader = ConfigLoader()
        catalog = loader.load_catalog(catalog_file)

        assert isinstance(catalog, WorkflowCatalog)
        assert len(catalog.workflows) == 1
        assert catalog.workflows[0].id == "workflow_1"

    def test_load_catalog_with_env_vars(self, tmp_path):
        """Test catalog loading with environment variable expansion"""
        os.environ["WORKFLOW_PATH"] = "custom/path"

        catalog_file = tmp_path / "catalog.yaml"
        catalog_file.write_text("""
meta:
  version: "1.0"
workflows:
  - id: "workflow_1"
    label: "Test"
    type: "workflow"
    dsl_path: "${WORKFLOW_PATH}/test.yaml"
    nodes: []
""")

        loader = ConfigLoader()
        catalog = loader.load_catalog(catalog_file)

        # Use Path for platform-independent comparison
        assert catalog.workflows[0].dsl_path == Path("custom/path/test.yaml")
        del os.environ["WORKFLOW_PATH"]

    def test_load_catalog_validation_error(self, tmp_path):
        """Test load_catalog raises ConfigurationError on validation failure"""
        catalog_file = tmp_path / "invalid_catalog.yaml"
        catalog_file.write_text("""
meta:
  version: "1.0"
workflows:
  - id: "workflow_1"
    # Missing required fields
""")

        loader = ConfigLoader()

        with pytest.raises(ConfigurationError, match="Failed to validate workflow_repository"):
            loader.load_catalog(catalog_file)


class TestConfigLoaderLoadTestPlan:
    """Tests for load_test_plan"""

    def test_load_test_plan_success(self, tmp_path):
        """Test successful test plan loading"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""
meta:
  plan_id: "test_plan_1"
workflows:
  - catalog_id: "workflow_1"
    dataset_refs: ["dataset_1"]
test_data:
  datasets:
    - name: "dataset_1"
      scenario: "normal"
execution:
  concurrency: 2
  rate_control:
    per_minute: 30
""")

        loader = ConfigLoader()
        plan = loader.load_test_plan(plan_file)

        assert isinstance(plan, TestPlan)
        assert plan.meta["plan_id"] == "test_plan_1"
        assert len(plan.workflows) == 1

    def test_load_test_plan_with_env_vars(self, tmp_path):
        """Test test plan loading with environment variable expansion"""
        os.environ["PLAN_ID"] = "dynamic_plan"

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""
meta:
  plan_id: "${PLAN_ID}"
workflows:
  - catalog_id: "workflow_1"
    dataset_refs: []
test_data:
  datasets: []
execution:
  concurrency: 1
  rate_control:
    per_minute: 10
""")

        loader = ConfigLoader()
        plan = loader.load_test_plan(plan_file)

        assert plan.meta["plan_id"] == "dynamic_plan"
        del os.environ["PLAN_ID"]

    def test_load_test_plan_validation_error(self, tmp_path):
        """Test load_test_plan raises ConfigurationError on validation failure"""
        plan_file = tmp_path / "invalid_plan.yaml"
        plan_file.write_text("""
meta:
  plan_id: "test"
# Missing required workflows and execution fields
""")

        loader = ConfigLoader()

        with pytest.raises(ConfigurationError, match="Failed to validate test_plan"):
            loader.load_test_plan(plan_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
