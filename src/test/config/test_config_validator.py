"""
Unit Tests for ConfigValidator

Tests validation logic for cross-file configuration consistency.
"""

import pytest
from pathlib import Path

from src.config.models import (
    EnvConfig, DifyConfig, AuthConfig, RateLimit, ModelEvaluator,
    WorkflowCatalog, WorkflowEntry, NodeMeta,
    TestPlan, WorkflowPlanEntry, TestDataConfig, Dataset, ExecutionPolicy,
    PromptVariant, PromptPatch, PromptSelector, PromptStrategy
)
from src.config.loaders import ConfigValidator
from src.config.utils.exceptions import ConfigurationError, ConfigReferenceError


class TestConfigValidatorEnv:
    """Tests for validate_env"""

    def test_validate_env_success(self):
        """Test successful environment validation"""
        env = EnvConfig(
            meta={"version": "1.0", "environment": "test"},
            dify=DifyConfig(
                base_url="https://api.dify.ai",
                auth=AuthConfig(primary_token="test_token_123"),
                rate_limits=RateLimit(per_minute=60, burst=10)
            ),
            model_evaluator=ModelEvaluator(provider="openai", model_name="gpt-4"),
            io_paths={},
            logging={}
        )

        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[]
        )
        validator = ConfigValidator(catalog)
        validator.validate_env(env)  # Should not raise

    def test_validate_env_invalid_url_format(self):
        """Test validation catches invalid URL at validator level"""
        # Create an EnvConfig with invalid URL by bypassing Pydantic validation
        # This tests the additional check in ConfigValidator.validate_env()
        from pydantic import SecretStr

        # Use model_construct to bypass validation
        dify_config = DifyConfig.model_construct(
            base_url="ftp://invalid-protocol.com",  # Invalid protocol
            auth=AuthConfig(primary_token=SecretStr("token")),
            rate_limits=RateLimit(per_minute=60)
        )

        env = EnvConfig.model_construct(
            meta={"version": "1.0"},
            dify=dify_config,
            model_evaluator=ModelEvaluator(provider="openai", model_name="gpt-4"),
            io_paths={},
            logging={}
        )

        catalog = WorkflowCatalog(meta={"version": "1.0"}, workflows=[])
        validator = ConfigValidator(catalog)

        with pytest.raises(ConfigurationError, match="Invalid Dify URL format"):
            validator.validate_env(env)

    def test_validate_env_empty_token(self):
        """Test validation fails for empty primary token"""
        env = EnvConfig(
            meta={"version": "1.0"},
            dify=DifyConfig(
                base_url="https://api.dify.ai",
                auth=AuthConfig(primary_token=""),  # Empty token
                rate_limits=RateLimit(per_minute=60)
            ),
            model_evaluator=ModelEvaluator(provider="openai", model_name="gpt-4"),
            io_paths={},
            logging={}
        )

        catalog = WorkflowCatalog(meta={"version": "1.0"}, workflows=[])
        validator = ConfigValidator(catalog)

        with pytest.raises(ConfigurationError, match="Primary token cannot be empty"):
            validator.validate_env(env)

    def test_validate_env_warns_missing_paths(self, tmp_path, caplog):
        """Test validation warns for non-existent paths"""
        env = EnvConfig(
            meta={"version": "1.0"},
            dify=DifyConfig(
                base_url="https://api.dify.ai",
                auth=AuthConfig(primary_token="token"),
                rate_limits=RateLimit(per_minute=60)
            ),
            model_evaluator=ModelEvaluator(provider="openai", model_name="gpt-4"),
            io_paths={"output": tmp_path / "nonexistent"},  # Non-existent path
            logging={}
        )

        catalog = WorkflowCatalog(meta={"version": "1.0"}, workflows=[])
        validator = ConfigValidator(catalog)

        validator.validate_env(env)
        assert "does not exist" in caplog.text


class TestConfigValidatorPlan:
    """Tests for validate_plan"""

    def test_validate_plan_success(self):
        """Test successful plan validation"""
        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[
                WorkflowEntry(
                    id="workflow_1",
                    label="Test Workflow",
                    type="chatflow",
                    dsl_path=Path("test.yaml"),
                    nodes=[]
                )
            ]
        )

        plan = TestPlan(
            meta={"plan_id": "test_plan"},
            workflows=[
                WorkflowPlanEntry(
                    catalog_id="workflow_1",
                    dataset_refs=["dataset_1"]
                )
            ],
            test_data=TestDataConfig(
                datasets=[
                    Dataset(name="dataset_1", scenario="normal")
                ]
            ),
            execution=ExecutionPolicy(
                concurrency=1,
                rate_control=RateLimit(per_minute=10)
            )
        )

        validator = ConfigValidator(catalog)
        validator.validate_plan(plan)  # Should not raise

    def test_validate_plan_workflow_not_in_catalog(self):
        """Test validation fails when workflow not in catalog"""
        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[]  # Empty catalog
        )

        plan = TestPlan(
            meta={"plan_id": "test_plan"},
            workflows=[
                WorkflowPlanEntry(
                    catalog_id="nonexistent_workflow",  # Does not exist
                    dataset_refs=[]
                )
            ],
            test_data=TestDataConfig(datasets=[]),
            execution=ExecutionPolicy(
                concurrency=1,
                rate_control=RateLimit(per_minute=10)
            )
        )

        validator = ConfigValidator(catalog)

        with pytest.raises(ConfigReferenceError, match="not found in workflow catalog"):
            validator.validate_plan(plan)

    def test_validate_plan_dataset_not_defined(self):
        """Test validation fails when referenced dataset not defined"""
        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[
                WorkflowEntry(
                    id="workflow_1",
                    label="Test",
                    type="chatflow",
                    dsl_path=Path("test.yaml"),
                    nodes=[]
                )
            ]
        )

        plan = TestPlan(
            meta={"plan_id": "test_plan"},
            workflows=[
                WorkflowPlanEntry(
                    catalog_id="workflow_1",
                    dataset_refs=["nonexistent_dataset"]  # Does not exist
                )
            ],
            test_data=TestDataConfig(datasets=[]),  # Empty datasets
            execution=ExecutionPolicy(
                concurrency=1,
                rate_control=RateLimit(per_minute=10)
            )
        )

        validator = ConfigValidator(catalog)

        with pytest.raises(ConfigReferenceError, match="not defined in test_data.datasets"):
            validator.validate_plan(plan)


class TestConfigValidatorPromptVariants:
    """Tests for validate_prompt_variants"""

    def test_validate_prompt_variants_success(self):
        """Test successful prompt variant validation"""
        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[
                WorkflowEntry(
                    id="workflow_1",
                    label="Test",
                    type="chatflow",
                    dsl_path=Path("test.yaml"),
                    nodes=[]
                )
            ]
        )

        plan = TestPlan(
            meta={"plan_id": "test_plan"},
            workflows=[
                WorkflowPlanEntry(
                    catalog_id="workflow_1",
                    dataset_refs=[],
                    prompt_optimization=[
                        PromptVariant(
                            variant_id="variant_1",
                            nodes=[
                                PromptPatch(
                                    selector=PromptSelector(by_id="node_1"),
                                    strategy=PromptStrategy(mode="replace", content="New prompt")
                                )
                            ]
                        ),
                        PromptVariant(
                            variant_id="variant_2",
                            nodes=[],
                            fallback_variant="variant_1"  # Valid reference
                        )
                    ]
                )
            ],
            test_data=TestDataConfig(datasets=[]),
            execution=ExecutionPolicy(
                concurrency=1,
                rate_control=RateLimit(per_minute=10)
            )
        )

        validator = ConfigValidator(catalog)
        validator.validate_prompt_variants(plan)  # Should not raise

    def test_validate_prompt_variants_duplicate_id(self):
        """Test validation fails for duplicate variant IDs"""
        catalog = WorkflowCatalog(meta={"version": "1.0"}, workflows=[])

        plan = TestPlan(
            meta={"plan_id": "test_plan"},
            workflows=[
                WorkflowPlanEntry(
                    catalog_id="workflow_1",
                    dataset_refs=[],
                    prompt_optimization=[
                        PromptVariant(
                            variant_id="duplicate",
                            nodes=[]
                        ),
                        PromptVariant(
                            variant_id="duplicate",  # Duplicate!
                            nodes=[]
                        )
                    ]
                )
            ],
            test_data=TestDataConfig(datasets=[]),
            execution=ExecutionPolicy(
                concurrency=1,
                rate_control=RateLimit(per_minute=10)
            )
        )

        validator = ConfigValidator(catalog)

        with pytest.raises(ConfigurationError, match="Duplicate variant_id"):
            validator.validate_prompt_variants(plan)

    def test_validate_prompt_variants_invalid_fallback(self):
        """Test validation fails for non-existent fallback variant"""
        catalog = WorkflowCatalog(meta={"version": "1.0"}, workflows=[])

        plan = TestPlan(
            meta={"plan_id": "test_plan"},
            workflows=[
                WorkflowPlanEntry(
                    catalog_id="workflow_1",
                    dataset_refs=[],
                    prompt_optimization=[
                        PromptVariant(
                            variant_id="variant_1",
                            nodes=[],
                            fallback_variant="nonexistent"  # Does not exist
                        )
                    ]
                )
            ],
            test_data=TestDataConfig(datasets=[]),
            execution=ExecutionPolicy(
                concurrency=1,
                rate_control=RateLimit(per_minute=10)
            )
        )

        validator = ConfigValidator(catalog)

        with pytest.raises(ConfigurationError, match="does not exist"):
            validator.validate_prompt_variants(plan)


class TestConfigValidatorAll:
    """Tests for validate_all"""

    def test_validate_all_success(self):
        """Test successful validation of all components"""
        catalog = WorkflowCatalog(
            meta={"version": "1.0"},
            workflows=[
                WorkflowEntry(
                    id="workflow_1",
                    label="Test",
                    type="chatflow",
                    dsl_path=Path("test.yaml"),
                    nodes=[]
                )
            ]
        )

        env = EnvConfig(
            meta={"version": "1.0"},
            dify=DifyConfig(
                base_url="https://api.dify.ai",
                auth=AuthConfig(primary_token="token"),
                rate_limits=RateLimit(per_minute=60)
            ),
            model_evaluator=ModelEvaluator(provider="openai", model_name="gpt-4"),
            io_paths={},
            logging={}
        )

        plan = TestPlan(
            meta={"plan_id": "test_plan"},
            workflows=[
                WorkflowPlanEntry(catalog_id="workflow_1", dataset_refs=["ds1"])
            ],
            test_data=TestDataConfig(
                datasets=[Dataset(name="ds1", scenario="normal")]
            ),
            execution=ExecutionPolicy(
                concurrency=1,
                rate_control=RateLimit(per_minute=10)
            )
        )

        validator = ConfigValidator(catalog)
        validator.validate_all(env, plan)  # Should not raise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
