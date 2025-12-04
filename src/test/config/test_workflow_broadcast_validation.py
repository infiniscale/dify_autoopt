import textwrap
from pathlib import Path
import pytest


def write_cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def test_workflow_inputs_broadcast_valid(tmp_path):
    from src.config.loaders.unified_config import UnifiedConfigLoader

    cfg = write_cfg(
        tmp_path,
        """
        meta: {version: "1.0.0", environment: development}
        dify: {base_url: "http://x"}
        auth: {username: u, password: p}
        variables: {}
        execution: {}
        optimization: {}
        io_paths: {}
        logging: {level: INFO}
        workflows:
          - id: wf1
            inputs:
              a: [1, 2]
              b: x
              c: [3, 4]
            reference: [r1, r2]
        """,
    )

    loader = UnifiedConfigLoader()
    app = loader.load(cfg)
    assert len(app.workflows) == 1


def test_workflow_inputs_broadcast_mismatch_raises(tmp_path):
    from src.config.loaders.unified_config import UnifiedConfigLoader

    cfg = write_cfg(
        tmp_path,
        """
        meta: {version: "1.0.0", environment: development}
        dify: {base_url: "http://x"}
        auth: {username: u, password: p}
        variables: {}
        execution: {}
        optimization: {}
        io_paths: {}
        logging: {level: INFO}
        workflows:
          - id: wf_bad
            inputs:
              a: [1, 2, 3]
              b: [10, 20]
            reference: [r1, r2]
        """,
    )

    loader = UnifiedConfigLoader()
    with pytest.raises(ValueError, match="list inputs must have equal length"):
        loader.load(cfg)

