from pathlib import Path

import pytest
import yaml

from src.optimizer.prompt_optimizer import PromptOptimizer, ReferenceSpec, PromptPatch


def _sample_yaml():
    return {
        "graph": {
            "nodes": [
                {
                    "id": "llm_node_1",
                    "type": "llm",
                    "data": {
                        "prompt_template": {
                            "messages": [
                                {"role": "system", "text": "You are a helpful assistant."}
                            ]
                        }
                    },
                }
            ]
        }
    }


def _nested_workflow_yaml():
    return {"workflow": _sample_yaml()}


@pytest.fixture()
def mock_dspy(monkeypatch):
    def _activate(suffix="Return JSON"):
        def fake_init(self, cfg):
            self.cfg = cfg
            self.available = True

        def fake_optimize_prompts(self, workflow_id, prompts, samples, reference_texts, constraints, workflow_context=None):
            patches = []
            for prompt in prompts:
                new_text = f"{prompt.text}\n\n- {suffix}"
                patches.append(
                    PromptPatch(
                        workflow_id=workflow_id,
                        node_id=prompt.node_id,
                        field_path=prompt.path,
                        old=prompt.text,
                        new=new_text,
                        rationale="mock dspy patch",
                        confidence=0.9,
                        evidence_runs=[s.index for s in samples],
                    )
                )
            return patches

        def fake_optimize_blocks(
                self,
                workflow_id,
                blocks,
                samples,
                reference_texts,
                constraints,
                fail_signal=None,
                workflow_context=None,
        ):
            patches = []
            for block in blocks:
                for msg in block.messages:
                    old_text = str(msg.get("text") or "")
                    new_text = f"{old_text}\n\n- {suffix}"
                    patches.append(
                        PromptPatch(
                            workflow_id=workflow_id,
                            node_id=block.node_id,
                            field_path=msg.get("path", ""),
                            old=old_text,
                            new=new_text,
                            rationale="mock dspy patch",
                            confidence=0.9,
                            evidence_runs=[s.index for s in samples],
                        )
                    )
            return patches

        monkeypatch.setattr("src.optimizer.prompt_optimizer._DspyPromptOptimizer.__init__", fake_init)
        monkeypatch.setattr(
            "src.optimizer.prompt_optimizer._DspyPromptOptimizer.optimize_prompts",
            fake_optimize_prompts,
        )
        monkeypatch.setattr(
            "src.optimizer.prompt_optimizer._DspyPromptOptimizer.optimize_blocks",
            fake_optimize_blocks,
        )

    return _activate


def test_generates_patch_when_constraint_missing(tmp_path: Path, mock_dspy):
    mock_dspy()
    optimizer = PromptOptimizer(llm_config={"use_dspy_optimizer": True}, validate_llm=False)
    run_results = [
        {"status": "success", "output": "plain text without json"},
        {"status": "success", "output": "still missing constraint"},
    ]
    reference = ReferenceSpec(constraints=["Return JSON"])

    report = optimizer.optimize_from_runs(
        workflow_id="wf_123",
        run_results=run_results,
        workflow_yaml=(_sample_yaml(), tmp_path / "app_wf_123.yml"),
        reference_spec=reference,
    )

    assert report.patches, "Expected at least one patch"
    assert report.patches[0].node_id == "llm_node_1"
    assert "Return JSON" in report.patches[0].new
    assert report.issues, "Should record detected issues"


def test_low_similarity_issue_creates_patch(tmp_path: Path, mock_dspy):
    mock_dspy()
    optimizer = PromptOptimizer(llm_config={"use_dspy_optimizer": True}, validate_llm=False)
    run_results = [{"status": "success", "output": "unrelated answer"}]
    reference = ReferenceSpec(expected_outputs=["expected deterministic answer"], similarity_threshold=0.9)

    report = optimizer.optimize_from_runs(
        workflow_id="wf_456",
        run_results=run_results,
        workflow_yaml=(_sample_yaml(), tmp_path / "app_wf_456.yml"),
        reference_spec=reference,
    )

    assert any(issue.kind == "low_similarity" for issue in report.issues)
    assert report.patches, "Patch should be suggested for low similarity"


def test_creates_patched_copy_when_applying(tmp_path: Path, mock_dspy):
    mock_dspy()
    optimizer = PromptOptimizer(default_output_root=tmp_path, llm_config={"use_dspy_optimizer": True}, validate_llm=False)
    run_results = [{"status": "success", "output": "missing constraint"}]
    reference = ReferenceSpec(constraints=["Return JSON"])

    report = optimizer.optimize_from_runs(
        workflow_id="wf_789",
        run_results=run_results,
        workflow_yaml=(_sample_yaml(), tmp_path / "app_wf_789.yml"),
        reference_spec=reference,
        output_root=tmp_path,
        apply_patches=True,
    )

    assert report.patched_path is not None
    assert report.patched_path.exists()


def test_nested_workflow_patch_updates_in_place(tmp_path: Path, mock_dspy):
    mock_dspy()
    optimizer = PromptOptimizer(default_output_root=tmp_path, llm_config={"use_dspy_optimizer": True}, validate_llm=False)
    run_results = [{"status": "success", "output": "missing constraint"}]
    reference = ReferenceSpec(constraints=["Return JSON"])
    workflow_tree = _nested_workflow_yaml()
    workflow_path = tmp_path / "app_wf_nested_patch.yml"

    report = optimizer.optimize_from_runs(
        workflow_id="wf_nested_patch",
        run_results=run_results,
        workflow_yaml=(workflow_tree, workflow_path),
        reference_spec=reference,
        output_root=tmp_path,
        apply_patches=True,
    )

    assert report.patched_path and report.patched_path.exists()
    patched = yaml.safe_load(report.patched_path.read_text())
    assert "graph" not in patched, "Should not create a new top-level graph key"
    prompt_text = patched["workflow"]["graph"]["nodes"][0]["data"]["prompt_template"]["messages"][0]["text"]
    assert "Return JSON" in prompt_text


def test_status_parsing_handles_nested_result_data(tmp_path: Path):
    optimizer = PromptOptimizer()
    run_results = [
        {"result": {"data": {"status": "succeeded"}}, "output": "fine"},
        {"result": {"data": {"status": "success"}}, "output": "fine too"},
    ]

    report = optimizer.optimize_from_runs(
        workflow_id="wf_nested_status",
        run_results=run_results,
        workflow_yaml=(_sample_yaml(), tmp_path / "app_wf_nested_status.yml"),
    )

    assert report.stats["failures"] == 0
    assert not report.issues or all(it.kind != "high_failure_rate" for it in report.issues)


def test_reads_prompts_from_workflow_graph_section(tmp_path: Path, mock_dspy):
    mock_dspy()
    optimizer = PromptOptimizer(llm_config={"use_dspy_optimizer": True}, validate_llm=False)
    run_results = [{"status": "success", "output": "plain"}]
    reference = ReferenceSpec(constraints=["Return JSON"])

    report = optimizer.optimize_from_runs(
        workflow_id="wf_nested_graph",
        run_results=run_results,
        workflow_yaml=(_nested_workflow_yaml(), tmp_path / "app_wf_nested_graph.yml"),
        reference_spec=reference,
    )

    assert report.patches, "Expected patches when constraints are missing in nested workflow graph"
