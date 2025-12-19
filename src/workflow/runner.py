"""
Workflow Runner - Minimal Async Stub

Runs a stubbed workflow execution with structured logging and returns a minimal
result payload for reporting.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

from src.config.models import WorkflowEntry
from src.utils.logger import get_logger, log_workflow_trace, log_performance


@dataclass
class RunResult:
    workflow_id: str
    label: str
    status: str
    started_at: str
    ended_at: str
    metrics: Dict[str, Any]


@log_performance("workflow_run")
async def run_workflow(workflow: WorkflowEntry) -> RunResult:
    """Run a single workflow in a stubbed manner.

    Simulates 3 phases and returns simple metrics.
    """
    logger = get_logger("workflow.runner")
    start = datetime.now()

    with log_workflow_trace(workflow.id, "full_execution", logger):
        # Input phase
        with log_workflow_trace(workflow.id, "input", logger):
            await asyncio.sleep(0.02)
            logger.info(
                "输入准备完成",
                extra={"workflow_id": workflow.id, "records": 10},
            )

        # Processing phase
        with log_workflow_trace(workflow.id, "processing", logger):
            await asyncio.sleep(0.05)
            logger.info(
                "处理完成",
                extra={"workflow_id": workflow.id, "nodes": len(workflow.nodes)},
            )

        # Output phase
        with log_workflow_trace(workflow.id, "output", logger):
            await asyncio.sleep(0.02)
            logger.info(
                "输出完成",
                extra={"workflow_id": workflow.id, "files": 1},
            )

    end = datetime.now()
    return RunResult(
        workflow_id=workflow.id,
        label=workflow.label,
        status="success",
        started_at=start.isoformat(),
        ended_at=end.isoformat(),
        metrics={
            "input_records": 10,
            "nodes": len(workflow.nodes),
            "outputs": 1,
            "duration_seconds": (end - start).total_seconds(),
        },
    )


@log_performance("workflow_run_inline")
async def run_inline_workflow(workflow_id: str, label: str) -> RunResult:
    """Run a workflow defined inline in unified config (no DSL/nodes)."""
    logger = get_logger("workflow.runner")
    start = datetime.now()
    with log_workflow_trace(workflow_id, "full_execution", logger):
        with log_workflow_trace(workflow_id, "input", logger):
            await asyncio.sleep(0.02)
            logger.info("输入准备完成", extra={"workflow_id": workflow_id, "records": 5})
        with log_workflow_trace(workflow_id, "processing", logger):
            await asyncio.sleep(0.04)
            logger.info("处理完成", extra={"workflow_id": workflow_id, "nodes": 0})
        with log_workflow_trace(workflow_id, "output", logger):
            await asyncio.sleep(0.02)
            logger.info("输出完成", extra={"workflow_id": workflow_id, "files": 1})
    end = datetime.now()
    return RunResult(
        workflow_id=workflow_id,
        label=label,
        status="success",
        started_at=start.isoformat(),
        ended_at=end.isoformat(),
        metrics={
            "input_records": 5,
            "nodes": 0,
            "outputs": 1,
            "duration_seconds": (end - start).total_seconds(),
        },
    )
