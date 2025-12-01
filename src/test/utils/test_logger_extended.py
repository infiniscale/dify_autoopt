"""
Extended tests for the logger utilities to ensure external usability.

Covers:
- get_logger error before initialization
- log_exception decorator behavior (no reraise)
- log_workflow_trace context usage
- log_context scoping (no exceptions)
- global context binding (no exceptions)
"""

import sys
import asyncio
from pathlib import Path


# Ensure project root is in sys.path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_get_logger_raises_before_init():
    from src.utils.logger.logger import get_logger, _log_manager, LoggingException

    # Ensure unconfigured
    _log_manager._configured = False

    try:
        get_logger("should_fail")
        raised = False
    except LoggingException:
        raised = True

    assert raised, "Expected LoggingException when logging not initialized"


def test_log_exception_no_reraise_and_workflow_trace():
    from src.utils.logger.logger import (
        setup_logging,
        get_logger,
        log_exception,
        log_workflow_trace,
        _log_manager,
    )

    async def _run():
        # Fresh init
        _log_manager._configured = False
        await setup_logging()

        @log_exception(reraise=False)
        async def boom():
            raise RuntimeError("expected-error")

        # Should not raise
        await boom()

        lg = get_logger("trace_test")
        # Should enter and exit without raising
        with log_workflow_trace("wf_xt", "op_xt", lg):
            pass

    asyncio.run(_run())


def test_log_context_and_global_context_bindings():
    from src.utils.logger.logger import (
        setup_logging,
        get_logger,
        log_context,
        _log_manager,
    )

    async def _run():
        _log_manager._configured = False
        await setup_logging()
        _log_manager.set_global_context(app="unit_test", env="test")

        lg = get_logger("context_test")
        lg.info("baseline")

        # Contextual fields should be accepted without error
        with log_context(request_id="req-1", user_id="u-1"):
            lg.info("with-context")

    asyncio.run(_run())
