import io
import json
from pathlib import Path

import pytest


def test_list_all_apps_pagination(monkeypatch):
    from src.workflow.apps import list_all_apps

    calls = {"n": 0}

    class DummyResp:
        def __init__(self, data, status=200):
            self._data = data
            self.status_code = status
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

        def json(self):
            return self._data

    def fake_get(url, headers=None, params=None, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return DummyResp({"data": {"items": [{"id": 1}, {"id": 2}], "total": 3}})
        else:
            return DummyResp({"data": {"items": [{"id": 3}], "total": 3}})

    monkeypatch.setattr("requests.get", fake_get)

    apps = list_all_apps(base_url="http://x", token="tkn", limit=2)
    assert [a["id"] for a in apps] == [1, 2, 3]


def test_export_app_dsl_json_to_yaml_from_data_string(monkeypatch, tmp_path):
    from src.workflow.export import export_app_dsl
    import yaml

    payload = {"result": "success", "data": json.dumps({"a": 1, "b": {"c": 2}})}

    class DummyResp:
        def __init__(self, data_bytes):
            self._data = data_bytes
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}
            self.encoding = "utf-8"

        def raise_for_status(self):
            pass

        @property
        def content(self):
            return self._data

        def iter_content(self, chunk_size=8192):
            yield self._data

        def json(self):
            return json.loads(self._data.decode("utf-8"))

    def fake_get(url, headers=None, params=None, timeout=None, stream=False):
        return DummyResp(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("requests.get", fake_get)

    out = export_app_dsl("my/app", base_url="http://x", token="tkn", output_dir=tmp_path)
    # Expect subdir created with sanitized id
    assert out.parent.name == "my_app"
    assert out.suffix == ".yml"
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert data == {"a": 1, "b": {"c": 2}}


def test_import_app_yaml_payload(monkeypatch, tmp_path):
    from src.workflow.imports import import_app_yaml

    captured = {}

    class DummyResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_post(url, headers=None, json=None, timeout=None, files=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return DummyResp({"result": "success", "data": {"id": "new_app"}})

    monkeypatch.setattr("requests.post", fake_post)

    yaml_text = "a: 1\nb: 2\n"
    res = import_app_yaml(yaml_content=yaml_text, base_url="http://x", token="tkn")
    assert captured["url"].endswith("/console/api/apps/imports")
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    assert captured["json"]["mode"] == "yaml-content"
    assert captured["json"]["yaml_content"] == yaml_text
    assert res["result"] == "success"


def test_publish_workflow(monkeypatch):
    from src.workflow.publish import publish_workflow

    class DummyResp:
        def __init__(self):
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            pass

        def json(self):
            return {"result": "success", "data": {"status": "ok"}}

    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return DummyResp()

    monkeypatch.setattr("requests.post", fake_post)

    res = publish_workflow("abc-123", base_url="http://x", token="tkn")
    assert captured["url"].endswith("/console/api/apps/abc-123/workflows/publish")
    assert captured["headers"]["Authorization"].startswith("Bearer ")
    assert res["result"] == "success"


def test_execute_workflow_broadcast_and_file_upload_v1(monkeypatch, tmp_path):
    from src.workflow.execute import execute_workflow_v1

    # Prepare a temp file to simulate file input
    f = tmp_path / "input.txt"
    f.write_text("hello", encoding="utf-8")

    calls = {"uploads": 0, "runs": 0, "last_payloads": []}

    class DummyResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_post(url, headers=None, json=None, timeout=None, files=None, data=None):
        if url.endswith("/files/upload"):
            calls["uploads"] += 1
            return DummyResp({"data": {"id": f"fid_{calls['uploads']}"}})
        else:
            calls["runs"] += 1
            calls["last_payloads"].append(json)
            return DummyResp({"result": "success", "data": {"run": calls["runs"]}})

    monkeypatch.setattr("requests.post", fake_post)

    # Inputs: file list and scalar number + string broadcast
    inputs = {
        "file": [str(f), str(f)],
        "top_k": 3,
        "note": "n",
    }
    res = execute_workflow_v1("app1", inputs, base_url="http://api", api_key="KEY")
    assert len(res) == 2
    assert calls["uploads"] == 2
    # Ensure each run payload contains a Dify document object with upload_file_id
    for p in calls["last_payloads"]:
        assert "inputs" in p
        doc = p["inputs"]["file"]
        assert isinstance(doc, dict)
        assert doc.get("type") == "document"
        assert doc.get("transfer_method") == "local_file"
        assert str(doc.get("upload_file_id")).startswith("fid_")


def test_execute_workflow_v1_grouped_inputs_list(monkeypatch, tmp_path):
    from src.workflow.execute import execute_workflow_v1

    # Prepare a temp file to simulate file input
    f = tmp_path / "g.txt"
    f.write_text("group", encoding="utf-8")

    calls = {"uploads": 0, "runs": 0, "payloads": []}

    class DummyResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_post(url, headers=None, json=None, timeout=None, files=None, data=None):
        if url.endswith("/files/upload"):
            calls["uploads"] += 1
            return DummyResp({"data": {"id": f"gid_{calls['uploads']}"}})
        else:
            calls["runs"] += 1
            calls["payloads"].append(json)
            return DummyResp({"result": "ok", "data": {"run": calls["runs"]}})

    monkeypatch.setattr("requests.post", fake_post)

    grouped = [
        {"file": str(f), "note": "a"},
        {"file": str(f), "note": "b"},
    ]
    res = execute_workflow_v1("appG", grouped, base_url="http://api", api_key="KEY")
    assert len(res) == 2
    assert calls["uploads"] == 2 and calls["runs"] == 2
    for p in calls["payloads"]:
        doc = p["inputs"]["file"]
        assert isinstance(doc, dict) and str(doc.get("upload_file_id")).startswith("gid_")


def test_execute_workflow_v1_uses_base_url_and_api_key(monkeypatch, tmp_path):
    from src.workflow.execute import execute_workflow_v1

    # temp file
    f = tmp_path / "a.txt"
    f.write_text("a", encoding="utf-8")

    calls = {"uploads": 0, "runs": 0, "last_headers": None, "urls": []}

    class DummyResp:
        def __init__(self, data):
            self._data = data
            self.status_code = 200
            self.headers = {"Content-Type": "application/json"}

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_post(url, headers=None, json=None, timeout=None, files=None, data=None):
        calls["urls"].append(url)
        if url.endswith("/files/upload"):
            calls["uploads"] += 1
            calls["last_headers"] = headers
            return DummyResp({"data": {"id": "fid1"}})
        else:
            calls["runs"] += 1
            calls["last_headers"] = headers
            return DummyResp({"result": "success"})

    monkeypatch.setattr("requests.post", fake_post)

    res = execute_workflow_v1(
        "wf1",
        {"file": str(f), "n": 1},
        base_url="http://api",
        api_key="KEY",
    )
    assert len(res) == 1
    assert calls["uploads"] == 1 and calls["runs"] == 1
    assert calls["urls"][0].endswith("/files/upload")
    assert calls["urls"][1].endswith("/workflows/run")
    assert calls["last_headers"]["Authorization"].startswith("Bearer ")


def test_execute_workflow_from_config_expands_rows(monkeypatch):
    from types import SimpleNamespace

    from src.workflow.execute import execute_workflow_from_config

    # Build config-like objects
    wf_inputs = {
        "file_field": {"type": "file", "value": ["a.pdf", "b.pdf"]},
        "note": {"type": "string", "value": "hello"},
        "FilelD": {"type": "string", "value": "alias"},
    }
    workflow_entry = SimpleNamespace(id="wf-x", inputs=wf_inputs, api_key="KEY")
    runtime = SimpleNamespace(app=SimpleNamespace(workflows=[workflow_entry], dify={"base_url": "http://api"}))

    calls = {}

    def fake_execute(app_id, rows, **kwargs):
        calls["app_id"] = app_id
        calls["rows"] = rows
        calls["input_types"] = kwargs.get("input_types")
        return rows

    monkeypatch.setattr("src.config.bootstrap.get_runtime", lambda: runtime)
    monkeypatch.setattr("src.workflow.execute.execute_workflow_v1", fake_execute)

    res = execute_workflow_from_config("wf-x")
    assert res == [
        {"file_field": "a.pdf", "note": "hello", "FileID": "alias"},
        {"file_field": "b.pdf", "note": "hello", "FileID": "alias"},
    ]
    assert calls["app_id"] == "wf-x"
    assert calls["input_types"]["file_field"] == "file"
    assert calls["input_types"]["FileID"] == "string"
