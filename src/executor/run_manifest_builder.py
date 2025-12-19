"""
Executor Module - RunManifest Builder

Date: 2025-11-13
Author: Rebirthli
Description: Builds RunManifest objects for execution
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from src.config.models import (
    EnvConfig,
    WorkflowCatalog,
    TestPlan,
    RunManifest,
    TestCase,
)
from src.optimizer.prompt_patch_engine import PromptPatchEngine
from src.executor.test_case_generator import TestCaseGenerator
from src.config.utils.exceptions import ManifestError

logger = logging.getLogger(__name__)


class RunManifestBuilder:
    """Builds RunManifest objects from configuration and test cases"""

    def __init__(
            self,
            env: EnvConfig,
            catalog: WorkflowCatalog,
            plan: TestPlan,
            patch_engine: PromptPatchEngine,
            case_generator: TestCaseGenerator
    ):
        """
        Initialize RunManifestBuilder

        Args:
            env: Environment configuration
            catalog: Workflow catalog
            plan: Test plan
            patch_engine: PromptPatchEngine for DSL patching
            case_generator: TestCaseGenerator for generating test cases
        """
        self.env = env
        self.catalog = catalog
        self.plan = plan
        self.patch_engine = patch_engine
        self.case_generator = case_generator

    def build_all(self) -> List[RunManifest]:
        """
        Build all RunManifests for the test plan

        Returns:
            List of RunManifest objects

        Raises:
            ManifestError: If building fails
        """
        manifests = []

        # Generate all test cases
        all_cases = self.case_generator.generate_all_cases()

        for wf_entry in self.plan.workflows:
            if not wf_entry.enabled:
                continue

            workflow_id = wf_entry.catalog_id
            workflow = self._get_workflow(workflow_id)

            if workflow is None:
                logger.error(f"Workflow '{workflow_id}' not found in catalog")
                continue

            cases = all_cases.get(workflow_id, [])
            if not cases:
                logger.warning(f"No test cases generated for workflow '{workflow_id}'")
                continue

            # Group cases by prompt variant
            variant_groups = {}
            for case in cases:
                variant_id = case.prompt_variant or 'default'
                variant_groups.setdefault(variant_id, []).append(case)

            # Build manifest for each variant
            for variant_id, variant_cases in variant_groups.items():
                try:
                    # Load original DSL
                    original_dsl = self._load_dsl(workflow.dsl_path)

                    # Apply prompt patches if variant is not default
                    if variant_id != 'default':
                        variant = self._get_variant(wf_entry, variant_id)
                        if variant is None:
                            logger.error(f"Variant '{variant_id}' not found")
                            continue

                        patched_dsl = self.patch_engine.apply_patches(
                            workflow_id,
                            original_dsl,
                            variant.nodes,
                            context={'dataset': variant_cases[0].dataset if variant_cases else ''}
                        )
                    else:
                        patched_dsl = original_dsl

                    # Create RunManifest
                    manifest = RunManifest(
                        workflow_id=workflow_id,
                        workflow_version=workflow.version or 'unknown',
                        prompt_variant=variant_id if variant_id != 'default' else None,
                        dsl_payload=patched_dsl,
                        cases=variant_cases,
                        execution_policy=self.plan.execution,
                        rate_limits=self.env.dify.rate_limits,
                        evaluator=self.env.model_evaluator,
                        metadata={
                            'plan_id': self.plan.meta.get('plan_id'),
                            'environment': self.env.meta.get('environment'),
                            'created_at': datetime.utcnow().isoformat()
                        }
                    )

                    manifests.append(manifest)
                    logger.info(
                        f"Built manifest for workflow '{workflow_id}', "
                        f"variant '{variant_id}', {len(variant_cases)} cases"
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to build manifest for workflow '{workflow_id}', "
                        f"variant '{variant_id}': {e}"
                    )
                    raise ManifestError(
                        f"Failed to build manifest for {workflow_id}/{variant_id}: {e}"
                    )

        logger.info(f"Built {len(manifests)} total manifests")
        return manifests

    def _get_workflow(self, workflow_id: str):
        """Get workflow from catalog"""
        return self.catalog.get_workflow(workflow_id)

    def _get_variant(self, wf_entry, variant_id: str):
        """Get prompt variant by ID"""
        if not wf_entry.prompt_optimization:
            return None

        for variant in wf_entry.prompt_optimization:
            if variant.variant_id == variant_id:
                return variant
        return None

    def _load_dsl(self, dsl_path: Path) -> str:
        """Load DSL YAML file as string"""
        try:
            with open(dsl_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ManifestError(f"Failed to load DSL from {dsl_path}: {e}")
