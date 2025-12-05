"""
YAML Configuration Module - Configuration Validator

Date: 2025-11-13
Author: Rebirthli
Description: Validates configuration consistency across multiple files
"""

import logging
from pathlib import Path
from typing import Set

from ..models import EnvConfig, WorkflowCatalog, TestPlan
from ..utils.exceptions import ConfigurationError, ConfigReferenceError

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Cross-file configuration consistency validator"""

    def __init__(self, catalog: WorkflowCatalog):
        """
        Initialize validator

        Args:
            catalog: WorkflowCatalog to validate against
        """
        self.catalog = catalog

    def validate_env(self, env: EnvConfig) -> None:
        """
        Validate environment configuration

        Args:
            env: EnvConfig instance

        Raises:
            ConfigurationError: If validation fails
        """
        # Validate URL format
        if not env.dify.base_url.startswith(('http://', 'https://')):
            raise ConfigurationError(f"Invalid Dify URL format: {env.dify.base_url}")

        # Validate paths exist (warn only)
        for path_key, path_val in env.io_paths.items():
            if not path_val.exists():
                logger.warning(f"Path '{path_key}' does not exist: {path_val}")

        # Validate tokens are non-empty
        if not env.dify.auth.primary_token.get_secret_value():
            raise ConfigurationError("Primary token cannot be empty")

        logger.info(f"EnvConfig validation passed: {env.meta.get('environment', 'unknown')}")

    def validate_plan(self, plan: TestPlan) -> None:
        """
        Validate test plan against catalog

        Args:
            plan: TestPlan instance

        Raises:
            ConfigReferenceError: If referenced workflows or datasets don't exist
        """
        # Build catalog workflow IDs
        catalog_ids: Set[str] = {wf.id for wf in self.catalog.workflows}

        # Build defined datasets set once for all workflows
        defined_datasets = {ds.name for ds in plan.datasets}

        # Validate workflow references
        for wf_entry in plan.workflows:
            if wf_entry.catalog_id not in catalog_ids:
                raise ConfigReferenceError(
                    f"Workflow '{wf_entry.catalog_id}' referenced in test plan "
                    f"not found in workflow catalog"
                )

            # Validate dataset references
            for ds_ref in wf_entry.dataset_refs:
                if ds_ref not in defined_datasets:
                    raise ConfigReferenceError(
                        f"Dataset '{ds_ref}' referenced by workflow '{wf_entry.catalog_id}' "
                        f"not defined in test_data.datasets"
                    )

        logger.info(f"TestPlan validation passed: {plan.meta.get('plan_id', 'unknown')}")

    def validate_prompt_variants(self, plan: TestPlan) -> None:
        """
        Validate prompt variant configurations

        Args:
            plan: TestPlan instance

        Raises:
            ConfigurationError: If variant configuration is invalid
        """
        for wf_entry in plan.workflows:
            if not wf_entry.prompt_optimization:
                continue

            # First pass: collect all variant IDs and check for duplicates
            variant_ids = set()
            for variant in wf_entry.prompt_optimization:
                if variant.variant_id in variant_ids:
                    raise ConfigurationError(
                        f"Duplicate variant_id '{variant.variant_id}' "
                        f"in workflow '{wf_entry.catalog_id}'"
                    )
                variant_ids.add(variant.variant_id)

            # Second pass: validate fallback variant references
            for variant in wf_entry.prompt_optimization:
                if variant.fallback_variant and variant.fallback_variant not in variant_ids:
                    raise ConfigurationError(
                        f"Fallback variant '{variant.fallback_variant}' for variant "
                        f"'{variant.variant_id}' does not exist in workflow '{wf_entry.catalog_id}'"
                    )

        logger.info("Prompt variant validation passed")

    def validate_all(self, env: EnvConfig, plan: TestPlan) -> None:
        """
        Run all validations

        Args:
            env: EnvConfig instance
            plan: TestPlan instance

        Raises:
            ConfigurationError, ConfigReferenceError: If any validation fails
        """
        self.validate_env(env)
        self.validate_plan(plan)
        self.validate_prompt_variants(plan)
        logger.info("All validations passed")
