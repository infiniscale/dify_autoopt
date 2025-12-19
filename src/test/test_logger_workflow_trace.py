import asyncio

from loguru import logger as loguru_logger


def test_log_workflow_trace_emits_start_end_and_injects_context():
    from src.utils.logger import setup_logging, get_logger, log_workflow_trace

    asyncio.run(setup_logging(force_reinit=True))

    records = []
    sink_id = loguru_logger.add(lambda m: records.append(m.record), level="INFO")
    try:
        wf_logger = get_logger("workflow.test")

        with log_workflow_trace("wf-1", "full_execution", logger=wf_logger):
            inner = get_logger("inner")
            inner.info("inner message")

        messages = [r.get("message") for r in records]
        assert "工作流阶段开始" in messages
        assert "工作流阶段完成" in messages

        inner_record = next(r for r in records if r.get("message") == "inner message")
        assert inner_record["extra"].get("workflow_id") == "wf-1"
        assert inner_record["extra"].get("workflow_operation") == "full_execution"
    finally:
        loguru_logger.remove(sink_id)


def test_log_exception_decorator_swallows_when_reraise_false():
    from src.utils.logger import setup_logging, log_exception

    asyncio.run(setup_logging(force_reinit=True))

    @log_exception(reraise=False)
    def _boom():
        raise ValueError("x")

    assert _boom() is None

