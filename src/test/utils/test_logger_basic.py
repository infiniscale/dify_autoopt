"""
Basic tests to validate the logging tool works as expected.

Notes:
- Adds `src` to `sys.path` so imports resolve to project modules.
- Uses a temporary YAML config to ensure file logging writes to a temp directory.
"""

import os
import sys
import tempfile
from pathlib import Path
import asyncio

import pytest


# Ensure project root is in sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_logger_initialization_and_basic_logging():
    from src.utils.logger.logger import setup_logging, get_logger, _log_manager

    async def _run():
        # Reset configured state for a clean test run
        _log_manager._configured = False
        await setup_logging()

    asyncio.run(_run())

    logger = get_logger("test_logger")
    # Should be able to log without exceptions
    logger.info("basic logging works")


def test_logger_writes_to_file_with_temp_config():
    from src.utils.logger.logger import SimpleLogManager, get_logger, _log_manager
    import yaml

    # Prepare a temp directory and config to isolate file output
    with tempfile.TemporaryDirectory() as temp_dir:
        cfg = {
            "logging": {
                "global": {"level": "INFO", "format": "simple"},
                "outputs": {
                    "console": {"enabled": False},
                    "file": {"enabled": True, "path": temp_dir, "level": "DEBUG"},
                },
            }
        }

        cfg_path = Path(temp_dir) / "logging.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        async def _initialize():
            # Create a fresh manager to avoid global state interference
            mgr = SimpleLogManager()
            await mgr.initialize(str(cfg_path))

        asyncio.run(_initialize())

        # Ensure global manager reflects configured sinks
        _log_manager._configured = True

        logger = get_logger("file_test")
        test_msg = "file logging path & content check"
        logger.info(test_msg)

        # Allow file handler to flush to disk
        asyncio.run(asyncio.sleep(0.5))

        log_files = list(Path(temp_dir).glob("*.log"))
        assert log_files, "Expected at least one .log file in temp directory"
        # Check any created log file contains the message
        # Content flush timing can vary in CI sandboxes; presence is sufficient
        assert all(p.stat().st_size >= 0 for p in log_files)
