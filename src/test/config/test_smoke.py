"""
Smoke Tests for Config Module

Basic tests to verify the refactored architecture works correctly.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config.models import RateLimit, EnvConfig, DifyConfig, AuthConfig, ModelEvaluator
from src.config.loaders import ConfigLoader


class TestModelsSmokeTest:
    """Basic smoke tests for models"""

    def test_rate_limit_creation(self):
        """Test RateLimit model can be created"""
        rl = RateLimit(per_minute=60, burst=10)
        assert rl.per_minute == 60
        assert rl.burst == 10

    def test_env_config_creation(self):
        """Test EnvConfig model can be created"""
        env = EnvConfig(
            meta={"version": "1.0"},
            dify=DifyConfig(
                base_url="https://test.com",
                auth=AuthConfig(primary_token="token"),
                rate_limits=RateLimit(per_minute=60)
            ),
            model_evaluator=ModelEvaluator(provider="openai", model_name="gpt-4"),
            io_paths={},
            logging={}
        )
        assert env.meta["version"] == "1.0"
        assert env.dify.base_url == "https://test.com"


class TestLoaderSmokeTest:
    """Basic smoke tests for loaders"""

    def test_config_loader_exists(self):
        """Test ConfigLoader can be imported and instantiated"""
        loader = ConfigLoader()
        assert loader is not None

    def test_load_env_config(self):
        """Test loading a simple env config"""
        with TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "env.yaml"
            config_path.write_text("""
meta:
  version: "1.0"
dify:
  base_url: "https://test.com"
  auth:
    primary_token: "test_token"
  rate_limits:
    per_minute: 60
model_evaluator:
  provider: "openai"
  model_name: "gpt-4"
io_paths: {}
logging: {}
""")

            loader = ConfigLoader()
            env = loader.load_env(config_path)

            assert env.dify.base_url == "https://test.com"
            assert env.dify.auth.primary_token.get_secret_value() == "test_token"


class TestExecutorSmokeTest:
    """Basic smoke tests for executor module"""

    def test_pairwise_engine_import(self):
        """Test PairwiseEngine can be imported"""
        from src.executor import PairwiseEngine
        engine = PairwiseEngine()
        assert engine is not None

    def test_pairwise_simple_generation(self):
        """Test PairwiseEngine can generate combinations"""
        from src.executor import PairwiseEngine

        engine = PairwiseEngine()
        dimensions = {
            "param1": ["a", "b"],
            "param2": [1, 2]
        }

        result = engine.generate(dimensions, seed=42)
        assert len(result) > 0
        assert all("param1" in case and "param2" in case for case in result)


class TestOptimizerSmokeTest:
    """Basic smoke tests for optimizer module"""

    def test_prompt_patch_engine_import(self):
        """Test PromptPatchEngine can be imported"""
        from src.optimizer import PromptPatchEngine
        # Just verify it can be imported
        assert PromptPatchEngine is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
