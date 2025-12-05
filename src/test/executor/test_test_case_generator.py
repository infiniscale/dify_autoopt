"""
Unit tests for TestCaseGenerator

These tests use lightweight stub objects instead of full config models to
exercise all strategy modes: scenario_only, pairwise, cartesian, and
conversation-flow based generation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.executor.test_case_generator import TestCaseGenerator
from src.config.models import TestCase


@dataclass
class StubInputParameter:
    values: Optional[List[Any]] = None
    range: Optional[Dict[str, Any]] = None
    default: Optional[Any] = None


@dataclass
class StubDataset:
    name: str
    scenario: str
    parameters: Dict[str, StubInputParameter]
    pairwise_dimensions: List[str]
    conversation_flows: Optional[List[Any]] = None


@dataclass
class StubWorkflowEntry:
    catalog_id: str
    enabled: bool
    dataset_refs: List[str]
    prompt_optimization: Optional[List[Any]] = None


class StubPlan:
    def __init__(self, workflows, datasets, strategy_config: Dict[str, Any]) -> None:
        self.workflows = workflows
        self._datasets = {d.name: d for d in datasets}
        self._strategy_config = strategy_config
        self.test_data = {"combination_strategy": strategy_config}

    def get_dataset(self, name: str):
        return self._datasets.get(name)


class StubPairwiseEngine:
    def __init__(self) -> None:
        self.last_dimensions = None
        self.last_seed = None
        self.last_max_cases = None

    def generate(self, dimensions, seed=None, max_cases=None):
        self.last_dimensions = dimensions
        self.last_seed = seed
        self.last_max_cases = max_cases
        # Return a simple predictable list
        return [{"x": 1}, {"x": 2}]


class TestTestCaseGenerator:
    """Test suite for TestCaseGenerator using stubs."""

    def test_generate_all_cases_scenario_only(self) -> None:
        """Scenario-only mode should take the first value of each parameter."""
        dataset = StubDataset(
            name="ds1",
            scenario="normal",
            parameters={
                "p1": StubInputParameter(values=[1, 2, 3]),
                "p2": StubInputParameter(default="v"),
            },
            pairwise_dimensions=[],
            conversation_flows=None,
        )
        workflow_entry = StubWorkflowEntry(
            catalog_id="wf1",
            enabled=True,
            dataset_refs=["ds1"],
        )

        plan = StubPlan(
            workflows=[workflow_entry],
            datasets=[dataset],
            strategy_config={"mode": "scenario_only"},
        )

        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=StubPairwiseEngine())

        all_cases = generator.generate_all_cases()

        assert "wf1" in all_cases
        cases = all_cases["wf1"]
        assert len(cases) == 1
        assert isinstance(cases[0], TestCase)
        assert cases[0].parameters == {"p1": 1, "p2": "v"}

    def test_generate_all_cases_pairwise_dimensions(self) -> None:
        """Pairwise mode should call pairwise_engine with extracted dimensions."""
        dataset = StubDataset(
            name="ds2",
            scenario="normal",
            parameters={
                "a": StubInputParameter(values=[1, 2]),
                "b": StubInputParameter(values=["x", "y"]),
            },
            pairwise_dimensions=["a", "b"],
            conversation_flows=None,
        )
        workflow_entry = StubWorkflowEntry(
            catalog_id="wf2",
            enabled=True,
            dataset_refs=["ds2"],
        )

        strategy_config = {"mode": "pairwise", "random_seed": 42, "max_cases": 10}
        plan = StubPlan(
            workflows=[workflow_entry],
            datasets=[dataset],
            strategy_config=strategy_config,
        )

        pairwise_engine = StubPairwiseEngine()
        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=pairwise_engine)

        all_cases = generator.generate_all_cases()

        assert "wf2" in all_cases
        cases = all_cases["wf2"]
        assert len(cases) == 2
        # Ensure pairwise_engine was called with correct dimensions
        assert set(pairwise_engine.last_dimensions.keys()) == {"a", "b"}
        assert pairwise_engine.last_seed == 42
        assert pairwise_engine.last_max_cases == 10

    def test_generate_all_cases_pairwise_without_dimensions_falls_back_first_values(self) -> None:
        """When pairwise_dimensions is empty, generator should fall back to first values."""
        dataset = StubDataset(
            name="ds3",
            scenario="normal",
            parameters={"a": StubInputParameter(values=[10, 20])},
            pairwise_dimensions=[],
            conversation_flows=None,
        )
        workflow_entry = StubWorkflowEntry(
            catalog_id="wf3",
            enabled=True,
            dataset_refs=["ds3"],
        )

        plan = StubPlan(
            workflows=[workflow_entry],
            datasets=[dataset],
            strategy_config={"mode": "pairwise"},
        )

        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=StubPairwiseEngine())

        all_cases = generator.generate_all_cases()

        cases = all_cases["wf3"]
        assert len(cases) == 1
        assert cases[0].parameters == {"a": 10}

    def test_generate_all_cases_cartesian_respects_max_cases(self) -> None:
        """Cartesian mode should truncate when exceeding max_cases."""
        dataset = StubDataset(
            name="ds4",
            scenario="normal",
            parameters={
                "p": StubInputParameter(values=[1, 2, 3]),
                "q": StubInputParameter(values=["a", "b"]),
            },
            pairwise_dimensions=[],
            conversation_flows=None,
        )
        workflow_entry = StubWorkflowEntry(
            catalog_id="wf4",
            enabled=True,
            dataset_refs=["ds4"],
        )

        plan = StubPlan(
            workflows=[workflow_entry],
            datasets=[dataset],
            strategy_config={"mode": "cartesian", "max_cases": 3},
        )

        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=StubPairwiseEngine())

        all_cases = generator.generate_all_cases()

        cases = all_cases["wf4"]
        # Total combinations would be 3 * 2 = 6, but truncated to 3
        assert len(cases) == 3

    def test_generate_cases_with_conversation_flows(self) -> None:
        """Datasets with conversation_flows should use flows directly."""
        dataset = StubDataset(
            name="ds5",
            scenario="chat",
            parameters={
                "query": StubInputParameter(values=["hello", "bye"]),
            },
            pairwise_dimensions=[],
            # Use minimal objects compatible with config.models.ConversationFlow
            conversation_flows=[
                {"title": "flow1", "steps": [], "expected_outcome": "ok"},
                {"title": "flow2", "steps": [], "expected_outcome": "ok"},
            ],
        )
        workflow_entry = StubWorkflowEntry(
            catalog_id="wf5",
            enabled=True,
            dataset_refs=["ds5"],
        )

        plan = StubPlan(
            workflows=[workflow_entry],
            datasets=[dataset],
            strategy_config={"mode": "pairwise"},
        )

        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=StubPairwiseEngine())

        all_cases = generator.generate_all_cases()

        cases = all_cases["wf5"]
        assert len(cases) == 2
        assert all(case.conversation_flow is not None for case in cases)

    def test_generate_all_cases_skips_disabled_workflow(self) -> None:
        """Disabled workflows should be skipped without generating cases."""
        dataset = StubDataset(
            name="ds_disabled",
            scenario="normal",
            parameters={"p": StubInputParameter(values=[1])},
            pairwise_dimensions=[],
            conversation_flows=None,
        )
        disabled_entry = StubWorkflowEntry(
            catalog_id="wf_disabled",
            enabled=False,
            dataset_refs=["ds_disabled"],
        )

        plan = StubPlan(
            workflows=[disabled_entry],
            datasets=[dataset],
            strategy_config={"mode": "scenario_only"},
        )

        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=StubPairwiseEngine())

        all_cases = generator.generate_all_cases()

        # No entries should be produced for disabled workflow
        assert all_cases == {}

    def test_get_datasets_warns_on_missing_dataset(self) -> None:
        """_get_datasets should skip missing dataset references."""
        plan = StubPlan(workflows=[], datasets=[], strategy_config={})
        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=StubPairwiseEngine())

        datasets = generator._get_datasets(["missing_ds"])

        assert datasets == []

    def test_extract_pairwise_dimensions_skips_unknown_params(self) -> None:
        """_extract_pairwise_dimensions should ignore unknown parameter names."""
        dataset = StubDataset(
            name="ds_pairwise_unknown",
            scenario="normal",
            parameters={"a": StubInputParameter(values=[1])},
            pairwise_dimensions=["a", "missing_param"],
            conversation_flows=None,
        )
        plan = StubPlan(workflows=[], datasets=[dataset], strategy_config={})
        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=StubPairwiseEngine())

        dimensions = generator._extract_pairwise_dimensions(dataset)

        assert dimensions == {"a": [1]}

    def test_expand_parameter_range_and_default_and_empty(self) -> None:
        """_expand_parameter should support range, default, and empty parameters."""
        plan = StubPlan(workflows=[], datasets=[], strategy_config={})
        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=StubPairwiseEngine())

        # Range-based parameter
        range_param = StubInputParameter(range={"min": 1, "max": 3, "step": 1})
        values = generator._expand_parameter(range_param)
        assert values == [1, 2, 3]

        # Default-only parameter
        default_param = StubInputParameter(default="x")
        assert generator._expand_parameter(default_param) == ["x"]

        # Completely empty parameter
        empty_param = StubInputParameter()
        assert generator._expand_parameter(empty_param) == []

    def test_get_first_values_uses_none_for_missing_values(self) -> None:
        """_get_first_values should return None when a parameter has no candidates."""
        plan = StubPlan(workflows=[], datasets=[], strategy_config={})
        generator = TestCaseGenerator(plan=plan, catalog=None, pairwise_engine=StubPairwiseEngine())

        parameters = {
            "a": StubInputParameter(values=[1, 2]),
            "b": StubInputParameter(),  # no values/range/default
        }

        first_values = generator._get_first_values(parameters)

        assert first_values["a"] == 1
        assert first_values["b"] is None
