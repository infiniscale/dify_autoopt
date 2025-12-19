from pathlib import Path

import pytest


def test_loads_yaml_from_sanitized_directory(tmp_path: Path):
    from src.optimizer.yaml_loader import load_workflow_yaml

    workflow_id = "wf/001"
    safe_dir = tmp_path / "wf_001"
    safe_dir.mkdir(parents=True)
    target = safe_dir / "app_wf_001.yaml"
    target.write_text("name: demo\n", encoding="utf-8")

    data, path = load_workflow_yaml(workflow_id, output_dir=tmp_path)

    assert data["name"] == "demo"
    assert path == target


def test_falls_back_to_export_filename(tmp_path: Path):
    from src.optimizer.yaml_loader import load_workflow_yaml

    workflow_id = "wf-002"
    target_dir = tmp_path / workflow_id
    target_dir.mkdir(parents=True)
    fallback = target_dir / f"app_{workflow_id}_export.yml"
    fallback.write_text("id: wf-002\n", encoding="utf-8")

    data, path = load_workflow_yaml(workflow_id, output_dir=tmp_path)

    assert data["id"] == "wf-002"
    assert path == fallback


def test_raises_when_yaml_missing(tmp_path: Path):
    from src.optimizer.yaml_loader import load_workflow_yaml

    with pytest.raises(FileNotFoundError) as excinfo:
        load_workflow_yaml("wf-404", output_dir=tmp_path)
    assert "wf-404" in str(excinfo.value)


def test_handles_export_without_id_when_sanitized(tmp_path: Path):
    from src.optimizer.yaml_loader import load_workflow_yaml

    workflow_id = "wf/003"
    safe_dir = tmp_path / "wf_003"
    safe_dir.mkdir(parents=True)
    target = safe_dir / "app_export.yml"
    target.write_text("id: wf/003\n", encoding="utf-8")

    data, path = load_workflow_yaml(workflow_id, output_dir=tmp_path)

    assert data["id"] == "wf/003"
    assert path == target
