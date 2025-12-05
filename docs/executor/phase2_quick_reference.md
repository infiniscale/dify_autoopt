# ConcurrentExecutor & StubExecutor - Quick Reference

## Basic Usage

### 1. Using ConcurrentExecutor with Custom Function

```python
from src.executor import ConcurrentExecutor
from src.config.models import RunManifest

# Define custom execution function
def dify_api_caller(task):
    """Execute task by calling real Dify API"""
    response = dify_client.run_workflow(
        workflow_id=task.workflow_id,
        inputs=task.parameters
    )
    return {
        "workflow_run_id": response.run_id,
        "status": response.status,
        "outputs": response.outputs,
        "elapsed_time": response.elapsed_time
    }

# Create executor with custom function
executor = ConcurrentExecutor(task_execution_func=dify_api_caller)

# Execute manifest
manifest = RunManifest(...)
result = executor.run_manifest(manifest)

# Access results
print(f"Success rate: {result.statistics.success_rate}")
print(f"Total duration: {result.total_duration}s")
print(f"Tasks succeeded: {result.statistics.succeeded_tasks}")
print(f"Tasks failed: {result.statistics.failed_tasks}")
```

### 2. Using StubExecutor for Testing

```python
from src.executor import StubExecutor
from src.config.models import RunManifest

# Simple stub with delay
stub = StubExecutor(simulated_delay=0.1)
result = stub.run_manifest(manifest)

# Stub with 30% failure rate
stub = StubExecutor(failure_rate=0.3)
result = stub.run_manifest(manifest)

# Stub with specific task behaviors
stub = StubExecutor(task_behaviors={
    "task-1": "success",
    "task-2": "failure",
    "task-3": "timeout",
    "task-4": "error"
})
result = stub.run_manifest(manifest)
```

### 3. Using Cancellation Token

```python
from src.executor import StubExecutor, CancellationToken
import threading
import time

# Create cancellation token
token = CancellationToken()

# Create executor
stub = StubExecutor(simulated_delay=5.0)

# Function to cancel after timeout
def cancel_after_timeout(seconds):
    time.sleep(seconds)
    token.cancel()
    print("Cancelled!")

# Start cancellation timer
threading.Thread(target=cancel_after_timeout, args=(3.0,)).start()

# Execute with cancellation support
result = stub.run_manifest(manifest, cancellation_token=token)

print(f"Cancelled tasks: {result.statistics.cancelled_tasks}")
```

## Configuration Examples

### ExecutionPolicy Configuration

```yaml
# In test_plan.yaml
execution:
  concurrency: 5              # Max 5 concurrent tasks
  batch_size: 10              # Process 10 tasks per batch
  retry_policy:
    max_attempts: 3           # Try up to 3 times
    backoff_seconds: 2.0      # Initial backoff 2s
    backoff_multiplier: 1.5   # Exponential multiplier
  stop_conditions:
    timeout_per_task: 300.0   # 5 minutes per task
    max_failures: 10          # Stop after 10 failures
```

### Dynamic StubExecutor Configuration

```python
stub = StubExecutor()

# Configure delay
stub.set_simulated_delay(0.5)  # 500ms delay

# Configure failure rate
stub.set_failure_rate(0.2)  # 20% failure rate

# Configure specific task behaviors
stub.set_task_behavior("task-abc-123", "timeout")
stub.set_task_behavior("task-xyz-456", "success")

# Clear all behaviors
stub.clear_task_behaviors()

# Execute with current configuration
result = stub.run_manifest(manifest)
```

## Retry Behavior

### Retry Flow

```
Attempt 1: Execute → Fail
  ↓ Wait 2.0s (backoff_seconds * 1.5^0)
Attempt 2: Execute → Fail
  ↓ Wait 3.0s (backoff_seconds * 1.5^1)
Attempt 3: Execute → Success
  ✓ Return TaskResult with attempt_count=3
```

### What Gets Retried

- TaskStatus.FAILED (business logic error)
- TaskStatus.TIMEOUT (execution timeout)
- TaskStatus.ERROR (system exception)

### What Doesn't Get Retried

- TaskStatus.SUCCEEDED (success)
- TaskStatus.CANCELLED (cancelled)

## Task Status Flow

```
PENDING
  ↓ (mark_started)
RUNNING
  ↓
  ├─→ SUCCEEDED (execution successful)
  ├─→ FAILED (business error) ──┐
  ├─→ TIMEOUT (time exceeded) ───┤→ Retry? → Back to PENDING
  ├─→ ERROR (system exception) ──┘
  └─→ CANCELLED (user cancelled)
```

## Error Handling

### Exception to Status Mapping

| Exception Type | Task Status | Retryable |
|----------------|-------------|-----------|
| TaskExecutionException | FAILED | Yes |
| TaskTimeoutException | TIMEOUT | Yes |
| Generic Exception | ERROR | Yes |
| (none) | SUCCEEDED | No |
| Cancellation | CANCELLED | No |

### Custom Exception Raising

```python
from src.utils.exceptions import TaskExecutionException, TaskTimeoutException

def my_execution_func(task):
    if not validate_input(task.parameters):
        # Business logic error → FAILED status
        raise TaskExecutionException("Invalid input parameters")

    if check_timeout(task):
        # Timeout → TIMEOUT status
        raise TaskTimeoutException("Operation exceeded deadline")

    # Any other exception → ERROR status
    # Success → return dict
    return {"status": "succeeded", "outputs": {...}}
```

## Testing Patterns

### Pattern 1: Test Retry Logic

```python
def test_retry_mechanism():
    attempts = []

    def flaky_execution(task):
        attempts.append(task.attempt_count)
        if task.attempt_count < 3:
            raise TaskExecutionException("Simulated failure")
        return {"status": "succeeded"}

    executor = ConcurrentExecutor(task_execution_func=flaky_execution)
    result = executor.run_manifest(manifest)

    assert len(attempts) == 3  # Executed 3 times
    assert result.statistics.succeeded_tasks == len(manifest.cases)
```

### Pattern 2: Test Timeout Handling

```python
def test_timeout():
    def slow_execution(task):
        time.sleep(100)  # Longer than timeout
        return {"status": "succeeded"}

    # Manifest with 1 second timeout
    manifest.execution_policy.stop_conditions["timeout_per_task"] = 1.0

    executor = ConcurrentExecutor(task_execution_func=slow_execution)
    result = executor.run_manifest(manifest)

    assert result.statistics.timeout_tasks > 0
```

### Pattern 3: Test Cancellation

```python
def test_cancellation():
    token = CancellationToken()
    stub = StubExecutor(simulated_delay=10.0)

    # Cancel immediately
    token.cancel()

    result = stub.run_manifest(manifest, cancellation_token=token)

    assert result.statistics.cancelled_tasks == len(manifest.cases)
```

### Pattern 4: Test Concurrency

```python
def test_concurrent_execution():
    execution_times = []

    def timed_execution(task):
        start = time.time()
        time.sleep(1.0)  # Simulate 1s work
        execution_times.append((task.task_id, time.time() - start))
        return {"status": "succeeded"}

    # 10 tasks with concurrency=5
    manifest = create_manifest_with_10_tasks()
    manifest.execution_policy.concurrency = 5

    executor = ConcurrentExecutor(task_execution_func=timed_execution)
    start = time.time()
    result = executor.run_manifest(manifest)
    total_time = time.time() - start

    # With concurrency=5, 10 tasks of 1s each should take ~2s (not 10s)
    assert total_time < 3.0
    assert len(execution_times) == 10
```

## Performance Tips

### 1. Optimize Concurrency

```python
# Too low: underutilized resources
manifest.execution_policy.concurrency = 1

# Too high: thread overhead, potential rate limiting
manifest.execution_policy.concurrency = 100

# Good balance for I/O-bound tasks
manifest.execution_policy.concurrency = 5-10
```

### 2. Set Appropriate Timeouts

```python
# Too short: premature timeouts
stop_conditions["timeout_per_task"] = 10.0  # 10s

# Too long: slow failure detection
stop_conditions["timeout_per_task"] = 3600.0  # 1 hour

# Reasonable for API calls
stop_conditions["timeout_per_task"] = 300.0  # 5 minutes
```

### 3. Configure Retry Wisely

```python
# No retries (fail fast)
retry_policy.max_attempts = 1

# Aggressive retries (may delay results)
retry_policy.max_attempts = 10
retry_policy.backoff_seconds = 1.0
retry_policy.backoff_multiplier = 2.0

# Balanced approach
retry_policy.max_attempts = 3
retry_policy.backoff_seconds = 2.0
retry_policy.backoff_multiplier = 1.5
```

## Common Pitfalls

### Pitfall 1: Not Checking Cancellation

```python
# Bad: Long-running function ignores cancellation
def bad_execution(task):
    for i in range(1000000):
        heavy_computation()  # No cancellation check
    return {"status": "succeeded"}

# Good: Check cancellation in custom code
def good_execution(task, cancellation_token=None):
    for i in range(1000000):
        if cancellation_token and cancellation_token.is_cancelled():
            raise TaskExecutionException("Cancelled")
        heavy_computation()
    return {"status": "succeeded"}
```

### Pitfall 2: Not Handling Exceptions

```python
# Bad: Unhandled exceptions become ERROR status
def bad_execution(task):
    result = api_call(task.parameters)  # May raise
    return result  # Not wrapped in try/except

# Good: Explicit exception handling
def good_execution(task):
    try:
        result = api_call(task.parameters)
        return result
    except ValueError as e:
        raise TaskExecutionException(f"Invalid input: {e}")
    except TimeoutError as e:
        raise TaskTimeoutException(f"API timeout: {e}")
```

### Pitfall 3: Blocking in Execution Function

```python
# Bad: Synchronous I/O blocks thread pool
def bad_execution(task):
    time.sleep(60)  # Blocks thread for 1 minute
    return {"status": "succeeded"}

# Better: Use timeout and keep work short
def good_execution(task):
    # Work should complete within timeout_per_task
    result = quick_api_call(task.parameters, timeout=10.0)
    return result
```

## Debugging Tips

### Enable Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("src.executor")

# Add debug logging in custom execution function
def debug_execution(task):
    logger.debug(f"Executing task {task.task_id}")
    logger.debug(f"Parameters: {task.parameters}")

    result = api_call(task.parameters)

    logger.debug(f"Result: {result}")
    return result
```

### Inspect Task Results

```python
result = executor.run_manifest(manifest)

# Print summary statistics
print(f"Total tasks: {result.statistics.total_tasks}")
print(f"Succeeded: {result.statistics.succeeded_tasks}")
print(f"Failed: {result.statistics.failed_tasks}")
print(f"Timeout: {result.statistics.timeout_tasks}")
print(f"Cancelled: {result.statistics.cancelled_tasks}")
print(f"Error: {result.statistics.error_tasks}")

# Inspect individual task results
for task_result in result.task_results:
    if task_result.task_status != TaskStatus.SUCCEEDED:
        print(f"Task {task_result.task_id} failed:")
        print(f"  Status: {task_result.task_status.value}")
        print(f"  Error: {task_result.error_message}")
        print(f"  Attempts: {task_result.attempt_count}")
```

### Use StubExecutor for Isolation

```python
# Test your workflow without external dependencies
stub = StubExecutor(simulated_delay=0.01)
result = stub.run_manifest(manifest)

# Verify manifest structure and execution policy
assert result.statistics.total_tasks == len(manifest.cases)
assert result.statistics.success_rate == 1.0  # All stubs succeed by default
```

## API Reference Summary

### ConcurrentExecutor

**Constructor:**
```python
ConcurrentExecutor(
    task_execution_func: Optional[Callable[[Task], Dict[str, Any]]] = None,
    now_fn: Callable[[], datetime] = datetime.now,
    sleep_fn: Callable[[float], None] = time.sleep,
    id_fn: Callable[[], str] = lambda: str(uuid4())
)
```

**Methods:**
- `run_manifest(manifest, cancellation_token=None) -> RunExecutionResult`
- Inherits all methods from `ExecutorBase`

### StubExecutor

**Constructor:**
```python
StubExecutor(
    simulated_delay: float = 0.0,
    failure_rate: float = 0.0,
    task_behaviors: Optional[Dict[str, str]] = None,
    now_fn: Callable[[], datetime] = datetime.now,
    sleep_fn: Callable[[float], None] = time.sleep,
    id_fn: Callable[[], str] = lambda: str(uuid4())
)
```

**Methods:**
- `set_task_behavior(task_id: str, behavior: str) -> None`
- `set_failure_rate(failure_rate: float) -> None`
- `set_simulated_delay(delay: float) -> None`
- `clear_task_behaviors() -> None`
- Inherits all methods from `ConcurrentExecutor` and `ExecutorBase`

**Behavior Values:**
- `"success"`: Task succeeds
- `"failure"`: Task fails with TaskExecutionException
- `"timeout"`: Task fails with TaskTimeoutException
- `"error"`: Task fails with RuntimeError

### CancellationToken

**Constructor:**
```python
CancellationToken()
```

**Methods:**
- `cancel() -> None`: Set cancellation flag
- `is_cancelled() -> bool`: Check if cancelled
- `reset() -> None`: Reset flag to False

## Next Steps

After Phase 2, the following integrations are possible:

1. **Real Dify API Integration**: Implement actual API caller function
2. **Result Collection**: Convert execution outputs to TestResult objects
3. **Batch Processing**: Implement batch execution respecting batch_size
4. **Rate Limiting**: Integrate with RateLimit configuration
5. **Progress Tracking**: Add callbacks for real-time progress updates

For questions or issues, refer to:
- Phase 1 documentation: `docs/executor/phase1_models.md`
- Phase 2 documentation: `docs/executor/phase2_implementation.md`
- System architecture: Review with system-architect role
