"""
Unit tests for RunManifestBuilder

These tests use stubbed EnvConfig/WorkflowCatalog/TestPlan/PromptPatchEngine
and TestCaseGenerator to validate RunManifestBuilder behaviour without
depending on full YAML config models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

import src.executor.run_manifest_builder as builder_module
from src.executor.run_manifest_builder import RunManifestBuilder
from src.config.utils.exceptions import ManifestError


@dataclass
class StubWorkflow:
    workflow_id: str
    dsl_path: Path
    version: str = "1.0.0"


class StubCatalog:
    def __init__(self, workflows: Dict[str, StubWorkflow]) -> None:
        self._workflows = workflows

    def get_workflow(self, workflow_id: str) -> Optional[StubWorkflow]:
        return self._workflows.get(workflow_id)


@dataclass
class StubPromptVariant:
    variant_id: str
    nodes: List[Dict[str, Any]]


@dataclass
class StubWorkflowEntry:
    catalog_id: str
    enabled: bool
    dataset_refs: List[str]
    prompt_optimization: Optional[List[StubPromptVariant]] = None


class StubPlan:
    def __init__(self, workflows: List[StubWorkflowEntry]) -> None:
        self.workflows = workflows
        self.execution = {"execution_policy": "value"}
        self.meta = {"plan_id": "plan_001"}


class StubDifyConfig:
    def __init__(self) -> None:
        self.rate_limits = {"per_minute": 60}


class StubEnv:
    def __init__(self) -> None:
        self.dify = StubDifyConfig()
        self.model_evaluator = {"provider": "openai"}
        self.meta = {"environment": "test"}


class StubCase:
    def __init__(self, workflow_id: str, dataset: str, scenario: str, prompt_variant: Optional[str]) -> None:
        self.workflow_id = workflow_id
        self.dataset = dataset
        self.scenario = scenario
        self.prompt_variant = prompt_variant


class StubCaseGenerator:
    def __init__(self, cases_by_workflow: Dict[str, List[StubCase]]) -> None:
        self._cases_by_workflow = cases_by_workflow

    def generate_all_cases(self) -> Dict[str, List[StubCase]]:
        return self._cases_by_workflow


class StubPatchEngine:
    def __init__(self) -> None:
        self.last_calls: List[Dict[str, Any]] = []

    def apply_patches(self, workflow_id: str, dsl: str, nodes: List[Dict[str, Any]], context: Dict[str, Any]) -> str:
        self.last_calls.append(
            {"workflow_id": workflow_id, "dsl": dsl, "nodes": nodes, "context": context}
        )
        return f"patched:{dsl}"


@dataclass
class CapturedRunManifest:
    workflow_id: str
    workflow_version: str
    prompt_variant: Optional[str]
    dsl_payload: str
    cases: List[Any]
    execution_policy: Any
    rate_limits: Any
    evaluator: Any
    metadata: Dict[str, Any]


class TestRunManifestBuilder:
    """Test suite for RunManifestBuilder using stub dependencies."""

    def test_build_all_creates_manifests_for_default_and_variant(self, tmp_path, monkeypatch) -> None:
        """Builder should create manifests per workflow variant and apply patches correctly."""
        # Prepare DSL file
        dsl_path = tmp_path / "workflow.yaml"
        dsl_path.write_text("dsl: original", encoding="utf-8")

        workflows = {
            "wf1": StubWorkflow(workflow_id="wf1", dsl_path=dsl_path),
        }
        catalog = StubCatalog(workflows)

        variant_v1 = StubPromptVariant(variant_id="v1", nodes=[{"op": "patch"}])
        wf_entry = StubWorkflowEntry(
            catalog_id="wf1",
            enabled=True,
            dataset_refs=["ds1"],
            prompt_optimization=[variant_v1],
        )
        plan = StubPlan(workflows=[wf_entry])

        # Cases: one default, one variant v1
        cases_by_workflow = {
            "wf1": [
                StubCase("wf1", "ds1", "normal", prompt_variant=None),
                StubCase("wf1", "ds1", "normal", prompt_variant="v1"),
                # Unknown variant to exercise variant-not-found branch
                StubCase("wf1", "ds1", "normal", prompt_variant="missing"),
            ]
        }
        case_generator = StubCaseGenerator(cases_by_workflow)
        patch_engine = StubPatchEngine()
        env = StubEnv()

        # Replace RunManifest with a simple capturing stub
        captured_manifests: List[CapturedRunManifest] = []

        def stub_run_manifest(**kwargs):
            manifest = CapturedRunManifest(**kwargs)
            captured_manifests.append(manifest)
            return manifest

        monkeypatch.setattr(builder_module, "RunManifest", stub_run_manifest)

        builder = RunManifestBuilder(
            env=env,
            catalog=catalog,
            plan=plan,
            patch_engine=patch_engine,
            case_generator=case_generator,
        )

        manifests = builder.build_all()

        # Two variants: default and v1
        assert len(manifests) == 2
        variants = {m.prompt_variant for m in captured_manifests}
        assert variants == {None, "v1"}

        # Default variant should use original DSL
        default_manifest = next(m for m in captured_manifests if m.prompt_variant is None)
        assert default_manifest.dsl_payload == "dsl: original"

        # v1 variant should use patched DSL
        v1_manifest = next(m for m in captured_manifests if m.prompt_variant == "v1")
        assert v1_manifest.dsl_payload == "patched:dsl: original"

        # Metadata should include plan/environment
        for manifest in captured_manifests:
            assert manifest.metadata["plan_id"] == "plan_001"
            assert manifest.metadata["environment"] == "test"

    def test_build_all_skips_disabled_and_missing_workflows(self, tmp_path, monkeypatch) -> None:
        """Disabled workflows and missing catalog entries should be skipped."""
        dsl_path = tmp_path / "wf.yaml"
        dsl_path.write_text("dsl: wf", encoding="utf-8")

        workflows = {
            "wf_enabled": StubWorkflow(workflow_id="wf_enabled", dsl_path=dsl_path),
            # Workflow present in catalog but with no cases
            "wf_no_cases": StubWorkflow(workflow_id="wf_no_cases", dsl_path=dsl_path),
            # "wf_missing" intentionally not present
        }
        catalog = StubCatalog(workflows)

        enabled_entry = StubWorkflowEntry(
            catalog_id="wf_enabled",
            enabled=True,
            dataset_refs=["ds"],
        )
        disabled_entry = StubWorkflowEntry(
            catalog_id="wf_disabled",
            enabled=False,
            dataset_refs=["ds"],
        )
        missing_entry = StubWorkflowEntry(
            catalog_id="wf_missing",
            enabled=True,
            dataset_refs=["ds"],
        )
        no_cases_entry = StubWorkflowEntry(
            catalog_id="wf_no_cases",
            enabled=True,
            dataset_refs=["ds"],
        )

        plan = StubPlan(workflows=[enabled_entry, disabled_entry, missing_entry, no_cases_entry])

        cases_by_workflow = {
            "wf_enabled": [StubCase("wf_enabled", "ds", "normal", prompt_variant=None)],
            # No cases for wf_missing or wf_no_cases -> should be skipped with warning
        }
        case_generator = StubCaseGenerator(cases_by_workflow)

        env = StubEnv()
        patch_engine = StubPatchEngine()

        captured_manifests: List[CapturedRunManifest] = []

        def stub_run_manifest(**kwargs):
            manifest = CapturedRunManifest(**kwargs)
            captured_manifests.append(manifest)
            return manifest

        monkeypatch.setattr(builder_module, "RunManifest", stub_run_manifest)

        builder = RunManifestBuilder(
            env=env,
            catalog=catalog,
            plan=plan,
            patch_engine=patch_engine,
            case_generator=case_generator,
        )

        manifests = builder.build_all()

        # Only enabled workflow with cases should produce a manifest
        assert len(manifests) == 1
        assert captured_manifests[0].workflow_id == "wf_enabled"

    def test_build_all_raises_manifest_error_on_dsl_load_failure(self, tmp_path) -> None:
        """Failure to load DSL should raise ManifestError with context."""
        # Prepare catalog with a workflow whose DSL path does not exist
        missing_path = tmp_path / "missing.yaml"
        workflows = {
            "wf1": StubWorkflow(workflow_id="wf1", dsl_path=missing_path),
        }
        catalog = StubCatalog(workflows)

        wf_entry = StubWorkflowEntry(
            catalog_id="wf1",
            enabled=True,
            dataset_refs=["ds"],
        )
        plan = StubPlan(workflows=[wf_entry])

        cases_by_workflow = {
            "wf1": [StubCase("wf1", "ds", "normal", prompt_variant=None)],
        }
        case_generator = StubCaseGenerator(cases_by_workflow)
        env = StubEnv()
        patch_engine = StubPatchEngine()

        builder = RunManifestBuilder(
            env=env,
            catalog=catalog,
            plan=plan,
            patch_engine=patch_engine,
            case_generator=case_generator,
        )

        with pytest.raises(ManifestError):
            builder.build_all()

    def test_get_variant_handles_missing_and_unknown_variants(self, tmp_path) -> None:
        """_get_variant should return None for missing configuration or unknown IDs."""
        catalog = StubCatalog(workflows={})
        env = StubEnv()

        # Entry without prompt_optimization
        entry_no_opt = StubWorkflowEntry(
            catalog_id="wf_no_opt",
            enabled=True,
            dataset_refs=[],
            prompt_optimization=None,
        )

        # Entry with prompt_optimization but missing requested ID
        entry_with_opt = StubWorkflowEntry(
            catalog_id="wf_with_opt",
            enabled=True,
            dataset_refs=[],
            prompt_optimization=[StubPromptVariant(variant_id="v1", nodes=[])],
        )

        plan = StubPlan(workflows=[entry_no_opt, entry_with_opt])
        case_generator = StubCaseGenerator(cases_by_workflow={})
        patch_engine = StubPatchEngine()

        builder = RunManifestBuilder(
            env=env,
            catalog=catalog,
            plan=plan,
            patch_engine=patch_engine,
            case_generator=case_generator,
        )

        assert builder._get_variant(entry_no_opt, "any") is None
        assert builder._get_variant(entry_with_opt, "missing") is None
