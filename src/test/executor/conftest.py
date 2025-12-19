"""
Test Fixtures for Executor Module

Date: 2025-11-14
Author: qa-engineer
Description: Shared pytest fixtures for executor module tests
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from datetime import datetime
from typing import Any, Dict

from src.config.models.run_manifest import RunManifest, TestCase
from src.config.models.test_plan import (
    ExecutionPolicy,
    RetryPolicy,
    ConversationFlow,
    ConversationStep
)
from src.config.models.common import RateLimit, ModelEvaluator
from src.executor.models import Task, TaskStatus
from src.collector.models import TestResult, TestStatus


@pytest.fixture
def fixed_time():
    """Fixed timestamp for deterministic testing"""
    return datetime(2025, 1, 1, 12, 0, 0)


@pytest.fixture
def fixed_id():
    """Fixed ID for deterministic testing"""
    return "test_id_001"


@pytest.fixture
def sample_retry_policy():
    """Sample retry policy configuration"""
    return RetryPolicy(
        max_attempts=3,
        backoff_seconds=2.0,
        backoff_multiplier=1.5
    )


@pytest.fixture
def sample_rate_limit():
    """Sample rate limit configuration"""
    return RateLimit(
        per_minute=60,
        burst=10
    )


@pytest.fixture
def sample_execution_policy(sample_rate_limit, sample_retry_policy):
    """Sample execution policy configuration"""
    return ExecutionPolicy(
        concurrency=5,
        batch_size=10,
        rate_control=sample_rate_limit,
        backoff_seconds=2.0,
        retry_policy=sample_retry_policy,
        stop_conditions={
            "timeout_per_task": 30.0,
            "max_failures": 10
        }
    )


@pytest.fixture
def sample_model_evaluator():
    """Sample model evaluator configuration"""
    return ModelEvaluator(
        provider="openai",
        model_name="gpt-4",
        temperature=0.2,
        max_tokens=512
    )


@pytest.fixture
def sample_conversation_flow():
    """Sample conversation flow for chatflow testing"""
    return ConversationFlow(
        title="Basic Conversation",
        steps=[
            ConversationStep(
                role="user",
                message="Hello, how are you?",
                wait_for_response=True
            ),
            ConversationStep(
                role="assistant",
                message="I'm doing well, thank you!",
                wait_for_response=True
            )
        ],
        expected_outcome="Successful greeting exchange"
    )


@pytest.fixture
def sample_test_case():
    """Sample test case"""
    return TestCase(
        workflow_id="wf_001",
        dataset="test_dataset",
        scenario="normal",
        parameters={"query": "test query", "max_results": 10},
        conversation_flow=None,
        prompt_variant=None,
        seed=42
    )


@pytest.fixture
def sample_test_case_with_conversation(sample_conversation_flow):
    """Sample test case with conversation flow"""
    return TestCase(
        workflow_id="wf_001",
        dataset="chatflow_dataset",
        scenario="normal",
        parameters={"context": "customer support"},
        conversation_flow=sample_conversation_flow,
        prompt_variant="variant_a",
        seed=42
    )


@pytest.fixture
def sample_manifest(
        sample_execution_policy,
        sample_rate_limit,
        sample_model_evaluator,
        sample_test_case
):
    """Sample RunManifest for testing"""
    return RunManifest(
        workflow_id="wf_001",
        workflow_version="1.0.0",
        prompt_variant="baseline",
        dsl_payload="workflow: test\nsteps: []",
        cases=[sample_test_case],
        execution_policy=sample_execution_policy,
        rate_limits=sample_rate_limit,
        evaluator=sample_model_evaluator,
        metadata={
            "plan_id": "plan_001",
            "owner": "qa_team",
            "description": "Test manifest"
        }
    )


@pytest.fixture
def sample_manifest_with_multiple_cases(
        sample_execution_policy,
        sample_rate_limit,
        sample_model_evaluator
):
    """Sample RunManifest with multiple test cases"""
    cases = [
        TestCase(
            workflow_id="wf_001",
            dataset="dataset_1",
            scenario="normal",
            parameters={"query": f"test query {i}"},
            conversation_flow=None,
            prompt_variant=None,
            seed=i
        )
        for i in range(5)
    ]

    return RunManifest(
        workflow_id="wf_001",
        workflow_version="1.0.0",
        prompt_variant="baseline",
        dsl_payload="workflow: test\nsteps: []",
        cases=cases,
        execution_policy=sample_execution_policy,
        rate_limits=sample_rate_limit,
        evaluator=sample_model_evaluator,
        metadata={}
    )


@pytest.fixture
def sample_task(sample_test_case, sample_execution_policy, fixed_time, fixed_id):
    """Sample Task object"""
    return Task.from_manifest_case(
        test_case=sample_test_case,
        execution_policy=sample_execution_policy,
        workflow_id="wf_001",
        id_fn=lambda: fixed_id,
        now_fn=lambda: fixed_time
    )


@pytest.fixture
def sample_test_result():
    """Sample TestResult object"""
    return TestResult(
        workflow_id="wf_001",
        execution_id="exec_001",
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        status=TestStatus.SUCCESS,
        execution_time=2.5,
        tokens_used=150,
        cost=0.002,
        inputs={"query": "test"},
        outputs={"answer": "response"},
        error_message=None,
        prompt_variant="baseline"
    )


@pytest.fixture
def id_generator():
    """ID generator fixture for sequential IDs"""
    counter = [0]

    def generate_id():
        counter[0] += 1
        return f"task_{counter[0]:03d}"

    return generate_id


@pytest.fixture
def time_generator(fixed_time):
    """Time generator fixture for sequential timestamps"""
    counter = [0]

    def generate_time():
        counter[0] += 1
        return datetime(
            fixed_time.year,
            fixed_time.month,
            fixed_time.day,
            fixed_time.hour,
            fixed_time.minute,
            fixed_time.second + counter[0]
        )

    return generate_time


# ============================================================================
# Phase 2 Fixtures for ConcurrentExecutor and StubExecutor
# ============================================================================


@pytest.fixture
def mock_now_fn(fixed_time):
    """Mock time function that returns fixed time"""
    return lambda: fixed_time


@pytest.fixture
def mock_sleep_fn():
    """Mock sleep function that does nothing (for fast tests)"""
    return lambda seconds: None


@pytest.fixture
def mock_id_fn():
    """Mock ID generator with predictable sequential IDs"""
    counter = [0]

    def generate_id():
        counter[0] += 1
        return f"task_{counter[0]:03d}"

    return generate_id


@pytest.fixture
def sample_run_manifest(sample_manifest):
    """Alias for sample_manifest for clarity in Phase 2 tests"""
    return sample_manifest


@pytest.fixture
def sample_task_result(sample_task):
    """Sample TaskResult from a Task"""
    from src.executor.models import TaskResult
    return TaskResult.from_task(sample_task)


@pytest.fixture
def cancellation_token():
    """CancellationToken instance for testing cancellation"""
    from src.executor.models import CancellationToken
    return CancellationToken()


@pytest.fixture
def concurrent_executor(mock_now_fn, mock_sleep_fn, mock_id_fn):
    """ConcurrentExecutor instance with dependency injection"""
    from src.executor.concurrent_executor import ConcurrentExecutor
    return ConcurrentExecutor(
        now_fn=mock_now_fn,
        sleep_fn=mock_sleep_fn,
        id_fn=mock_id_fn
    )


@pytest.fixture
def stub_executor(mock_now_fn, mock_sleep_fn, mock_id_fn):
    """StubExecutor instance with dependency injection"""
    from src.executor.stub_executor import StubExecutor
    return StubExecutor(
        simulated_delay=0.0,
        failure_rate=0.0,
        now_fn=mock_now_fn,
        sleep_fn=mock_sleep_fn,
        id_fn=mock_id_fn
    )


# ============================================================================
# Phase 3 Fixtures for RateLimiter and TaskScheduler
# ============================================================================


@pytest.fixture
def rate_limit_config():
    """RateLimit configuration for testing"""
    from src.config.models import RateLimit
    return RateLimit(per_minute=60, burst=10)


@pytest.fixture
def rate_limiter(rate_limit_config, mock_now_fn, mock_sleep_fn):
    """RateLimiter instance with mocked time"""
    from src.executor.rate_limiter import RateLimiter
    return RateLimiter(
        rate_limit=rate_limit_config,
        now_fn=mock_now_fn,
        sleep_fn=mock_sleep_fn
    )


@pytest.fixture
def task_scheduler(mock_now_fn, mock_sleep_fn, mock_id_fn):
    """TaskScheduler instance with dependency injection"""
    from src.executor.task_scheduler import TaskScheduler
    return TaskScheduler(
        now_fn=mock_now_fn,
        sleep_fn=mock_sleep_fn,
        id_fn=mock_id_fn
    )


@pytest.fixture
def execution_policy_with_rate_limit(rate_limit_config):
    """ExecutionPolicy with rate control"""
    from src.config.models import ExecutionPolicy, RetryPolicy
    return ExecutionPolicy(
        concurrency=5,
        batch_size=5,
        rate_control=rate_limit_config,
        backoff_seconds=0.1,
        retry_policy=RetryPolicy(),
        stop_conditions={}
    )
