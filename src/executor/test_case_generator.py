"""
Executor Module - Test Case Generator

Date: 2025-11-13
Author: Rebirthli
Description: Generates test cases from datasets using various strategies
"""

import logging
import itertools
from typing import Any, Dict, List, Optional

from ..config.models import TestPlan, WorkflowCatalog, Dataset, InputParameter, TestCase
from .pairwise_engine import PairwiseEngine
from ..config.utils.exceptions import CaseGenerationError

logger = logging.getLogger(__name__)


class TestCaseGenerator:
    """Generates test cases from datasets"""

    def __init__(
        self,
        plan: TestPlan,
        catalog: WorkflowCatalog,
        pairwise_engine: PairwiseEngine
    ):
        """
        Initialize TestCaseGenerator

        Args:
            plan: TestPlan configuration
            catalog: WorkflowCatalog for workflow metadata
            pairwise_engine: PairwiseEngine for combinatorial generation
        """
        self.plan = plan
        self.catalog = catalog
        self.pairwise_engine = pairwise_engine

    def generate_all_cases(self) -> Dict[str, List[TestCase]]:
        """
        Generate all test cases for all workflows in plan

        Returns:
            {workflow_id: [TestCase, ...]}
        """
        all_cases = {}

        for wf_entry in self.plan.workflows:
            if not wf_entry.enabled:
                logger.info(f"Skipping disabled workflow: {wf_entry.catalog_id}")
                continue

            # Get bound datasets
            datasets = self._get_datasets(wf_entry.dataset_refs)

            # Get prompt variants (or None for no optimization)
            variants = wf_entry.prompt_optimization or [None]

            cases = []
            for variant in variants:
                for dataset in datasets:
                    cases.extend(
                        self._generate_cases_for_dataset(
                            wf_entry.catalog_id,
                            dataset,
                            variant
                        )
                    )

            all_cases[wf_entry.catalog_id] = cases
            logger.info(
                f"Generated {len(cases)} test cases for workflow '{wf_entry.catalog_id}'"
            )

        return all_cases

    def _get_datasets(self, dataset_refs: List[str]) -> List[Dataset]:
        """Get datasets by name references"""
        datasets = []
        for ref in dataset_refs:
            dataset = self.plan.get_dataset(ref)
            if dataset:
                datasets.append(dataset)
            else:
                logger.warning(f"Dataset '{ref}' not found in test plan")
        return datasets

    def _generate_cases_for_dataset(
        self,
        workflow_id: str,
        dataset: Dataset,
        variant: Optional[Any]  # PromptVariant or None
    ) -> List[TestCase]:
        """Generate test cases for a single dataset"""
        cases = []
        strategy_config = self.plan.test_data.get('combination_strategy', {})
        mode = strategy_config.get('mode', 'pairwise')
        seed = strategy_config.get('random_seed')

        # If dataset has conversation flows, use them directly
        if dataset.conversation_flows:
            for flow in dataset.conversation_flows:
                cases.append(TestCase(
                    workflow_id=workflow_id,
                    dataset=dataset.name,
                    scenario=dataset.scenario,
                    parameters=self._extract_flow_parameters(dataset, flow),
                    conversation_flow=flow,
                    prompt_variant=variant.variant_id if variant else None,
                    seed=seed
                ))

        # Otherwise, generate parameter combinations
        else:
            if mode == 'scenario_only':
                # Use first value of each parameter
                param_combo = self._get_first_values(dataset.parameters)
                cases.append(TestCase(
                    workflow_id=workflow_id,
                    dataset=dataset.name,
                    scenario=dataset.scenario,
                    parameters=param_combo,
                    conversation_flow=None,
                    prompt_variant=variant.variant_id if variant else None,
                    seed=seed
                ))

            elif mode == 'pairwise':
                # Extract pairwise dimensions
                dimensions = self._extract_pairwise_dimensions(dataset)
                if not dimensions:
                    logger.warning(f"No pairwise dimensions for dataset '{dataset.name}', using first values")
                    param_combo = self._get_first_values(dataset.parameters)
                    cases.append(TestCase(
                        workflow_id=workflow_id,
                        dataset=dataset.name,
                        scenario=dataset.scenario,
                        parameters=param_combo,
                        conversation_flow=None,
                        prompt_variant=variant.variant_id if variant else None,
                        seed=seed
                    ))
                else:
                    # Generate pairwise combinations
                    combos = self.pairwise_engine.generate(
                        dimensions,
                        seed=seed,
                        max_cases=strategy_config.get('max_cases')
                    )

                    for combo in combos:
                        cases.append(TestCase(
                            workflow_id=workflow_id,
                            dataset=dataset.name,
                            scenario=dataset.scenario,
                            parameters=combo,
                            conversation_flow=None,
                            prompt_variant=variant.variant_id if variant else None,
                            seed=seed
                        ))

            elif mode == 'cartesian':
                # Full cartesian product (dangerous!)
                param_names = list(dataset.parameters.keys())
                param_values_lists = [
                    self._expand_parameter(dataset.parameters[k])
                    for k in param_names
                ]

                max_cases = strategy_config.get('max_cases', 1000)
                count = 0

                for combo_values in itertools.product(*param_values_lists):
                    if count >= max_cases:
                        logger.warning(
                            f"Cartesian product exceeded {max_cases} cases, truncating"
                        )
                        break

                    combo = dict(zip(param_names, combo_values))
                    cases.append(TestCase(
                        workflow_id=workflow_id,
                        dataset=dataset.name,
                        scenario=dataset.scenario,
                        parameters=combo,
                        conversation_flow=None,
                        prompt_variant=variant.variant_id if variant else None,
                        seed=seed
                    ))
                    count += 1

        return cases

    def _extract_pairwise_dimensions(self, dataset: Dataset) -> Dict[str, List[Any]]:
        """Extract pairwise dimensions from dataset"""
        dimensions = {}

        for param_name in dataset.pairwise_dimensions:
            if param_name not in dataset.parameters:
                logger.warning(f"Pairwise dimension '{param_name}' not in dataset parameters")
                continue

            param = dataset.parameters[param_name]
            values = self._expand_parameter(param)
            if values:
                dimensions[param_name] = values

        return dimensions

    def _expand_parameter(self, param: InputParameter) -> List[Any]:
        """Expand parameter to all possible values"""
        if param.values:
            return param.values

        if param.range:
            # Generate numeric range
            import numpy as np
            return list(np.arange(
                param.range['min'],
                param.range['max'] + param.range.get('step', 1),
                param.range.get('step', 1)
            ))

        if param.default is not None:
            return [param.default]

        return []

    def _get_first_values(self, parameters: Dict[str, InputParameter]) -> Dict[str, Any]:
        """Get first value of each parameter"""
        result = {}
        for name, param in parameters.items():
            values = self._expand_parameter(param)
            if values:
                result[name] = values[0]
            else:
                result[name] = None
        return result

    def _extract_flow_parameters(self, dataset: Dataset, flow: Any) -> Dict[str, Any]:
        """Extract parameters from conversation flow (use first values)"""
        return self._get_first_values(dataset.parameters)
