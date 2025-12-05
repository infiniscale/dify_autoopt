"""
Tests that logger can read simplified logging config from a unified config.yaml.
"""

import sys
import asyncio
from pathlib import Path
import tempfile


# Ensure project root is in sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_logger_reads_simplified_config_yaml():
    from src.utils.logger.logger import setup_logging, get_logger, _log_manager
    import yaml

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "logging": {
                "level": "INFO",
                "format": "simple",
                "console_enabled": True,
                "file_enabled": False,
            }
        }
        cfg_path = Path(tmp) / "config.yaml"
        (Path(tmp) / "config").mkdir(parents=True, exist_ok=True)
        # allow both root and config/config.yaml paths; use root/config.yaml
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        async def _run():
            _log_manager._configured = False
            # pass explicit path
            await setup_logging(str(cfg_path))

        asyncio.run(_run())

        lg = get_logger("cfg_yaml_test")
        lg.info("cfg ok")

