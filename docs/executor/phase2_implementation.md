# Executor Module - Phase 2 Implementation Documentation

## Overview

Phase 2 of the executor module implements **ConcurrentExecutor** and **StubExecutor**, providing concurrent task execution capabilities with retry mechanisms, timeout handling, and cancellation support.

## Implementation Summary

### Files Delivered

1. **src/executor/concurrent_executor.py** (~370 lines)
   - `ConcurrentExecutor` class with ThreadPoolExecutor-based concurrent execution
   - Retry mechanism with exponential backoff
   - Timeout handling per task
   - Cancellation token support
   - Dependency injection for testability

2. **src/executor/stub_executor.py** (~180 lines)
   - `StubExecutor` class for testing without real Dify API calls
   - Configurable simulated delay, failure rate, and task behaviors
   - Helper methods for dynamic configuration
   - Extends ConcurrentExecutor with custom execution function

3. **src/executor/__init__.py** (updated)
   - Added exports: `ConcurrentExecutor`, `TaskExecutionFunc`, `StubExecutor`

## Architecture Design

### ConcurrentExecutor

**Key Features:**
- Uses `concurrent.futures.ThreadPoolExecutor` for concurrent execution
- Dynamic concurrency control from `ExecutionPolicy.concurrency`
- Per-task timeout handling
- Retry mechanism with exponential backoff
- Graceful cancellation support via `CancellationToken`
- Dependency injection: `task_execution_func`, `now_fn`, `sleep_fn`, `id_fn`

**Core Methods:**

1. **`__init__(task_execution_func, now_fn, sleep_fn, id_fn)`**
   - Initializes executor with optional custom task execution function
   - Defaults to `_default_stub_execution` if no function provided

2. **`_execute_tasks(tasks, manifest, cancellation_token)`** (implements abstract method)
   - Reads concurrency from `manifest.execution_policy.concurrency`
   - Creates ThreadPoolExecutor with specified max_workers
   - Submits all tasks to thread pool
   - Uses `as_completed()` to collect results as they finish
   - Checks cancellation token at multiple points
   - Handles thread pool exceptions and task cancellation

3. **`_execute_single_task(task, retry_policy, cancellation_token)`**
   - Implements retry loop (max_attempts from retry_policy)
   - Uses nested ThreadPoolExecutor for timeout control
   - Handles three exception types:
     - `TaskTimeoutException` → TaskStatus.TIMEOUT
     - `TaskExecutionException` → TaskStatus.FAILED
     - Generic Exception → TaskStatus.ERROR
   - Implements exponential backoff: `backoff_seconds * (backoff_multiplier ** attempt)`
   - Checks cancellation token before each attempt and during backoff

4. **`_default_stub_execution(task)`**
   - Simple stub returning mock success data
   - Used when no custom execution function provided

### StubExecutor

**Key Features:**
- Extends `ConcurrentExecutor` with test-specific execution function
- No real Dify API calls
- Configurable behavior per task or globally
- Random failure simulation based on failure_rate

**Configuration Options:**

1. **Constructor Parameters:**
   - `simulated_delay`: Execution delay in seconds (default: 0.0)
   - `failure_rate`: Global failure probability 0.0-1.0 (default: 0.0)
   - `task_behaviors`: Dict mapping task_id to behavior string

2. **Behavior Types:**
   - `"success"`: Returns mock success data
   - `"failure"`: Raises `TaskExecutionException`
   - `"timeout"`: Raises `TaskTimeoutException`
   - `"error"`: Raises generic `RuntimeError`

3. **Helper Methods:**
   - `set_task_behavior(task_id, behavior)`: Configure specific task
   - `set_failure_rate(rate)`: Update global failure rate
   - `set_simulated_delay(delay)`: Update execution delay
   - `clear_task_behaviors()`: Remove all specific behaviors

**Core Methods:**

1. **`_stub_execution(task)`**
   - Checks `task_behaviors` for specific configuration
   - Falls back to random failure based on `failure_rate`
   - Sleeps for `simulated_delay`
   - Returns mock data or raises appropriate exception

2. **`_stub_success(task)`**
   - Generates realistic mock output data
   - Includes workflow_run_id, status, outputs, timestamps
   - Matches expected Dify API response structure

## Integration with Phase 1

### Dependency on ExecutorBase

Both executors inherit from `ExecutorBase` and follow the template method pattern:

```
ExecutorBase.run_manifest():
  1. _validate_manifest() ✓ (from base)
  2. _build_tasks()        ✓ (from base)
  3. _execute_tasks()      ✓ (implemented by ConcurrentExecutor)
  4. Aggregate results     ✓ (from base)
```

### Data Model Usage

- **Task**: Created via `Task.from_manifest_case()`, updated during execution
- **TaskStatus**: All 8 states used (PENDING, QUEUED, RUNNING, SUCCEEDED, FAILED, TIMEOUT, CANCELLED, ERROR)
- **TaskResult**: Built via `TaskResult.from_task()` after execution
- **CancellationToken**: Checked at critical points for graceful shutdown

### Exception Handling

- `TaskExecutionException`: Business logic failures
- `TaskTimeoutException`: Timeout exceeded
- `SchedulerException`: Thread pool or scheduling errors
- Generic exceptions → TaskStatus.ERROR

## Retry Mechanism Details

**Exponential Backoff Formula:**
```
backoff_time = retry_policy.backoff_seconds * (retry_policy.backoff_multiplier ** attempt)
```

**Example:** (backoff_seconds=2.0, backoff_multiplier=1.5)
- Attempt 0: Execute immediately
- Attempt 1: Wait 2.0s (2.0 * 1.5^0)
- Attempt 2: Wait 3.0s (2.0 * 1.5^1)
- Attempt 3: Wait 4.5s (2.0 * 1.5^2)

**Retry Conditions:**
- Only retry on FAILED, TIMEOUT, or ERROR status
- Stop after `max_attempts` reached
- Check cancellation token before each retry
- Reset task status to PENDING between retries

## Timeout Handling

**Implementation:**
- Uses nested `ThreadPoolExecutor` with single worker
- Submits task_execution_func to nested executor
- Calls `future.result(timeout=task.timeout_seconds)`
- On `TimeoutError`: cancels future, marks task as TIMEOUT
- Allows retry if not the last attempt

## Cancellation Support

**Cancellation Points:**
1. Before submitting tasks to thread pool
2. During task submission loop
3. After each task completion
4. Before each retry attempt
5. During retry backoff wait

**Behavior:**
- Remaining futures are cancelled via `future.cancel()`
- In-flight tasks complete normally
- Unstarted tasks marked as CANCELLED
- Cancellation is cooperative, not forceful

## Code Quality Compliance

### PEP 8 Standards
- 4-space indentation
- Max line length: ~88 characters (Black style)
- Snake_case for functions/variables
- PascalCase for classes
- Import order: stdlib → third-party → project

### Type Hints
- All function parameters annotated
- All return types specified
- Used typing module: `Callable`, `List`, `Optional`, `Dict`, `Any`

### Documentation
- Google-style docstrings for all classes and methods
- Args, Returns, Raises sections
- Inline comments for complex logic
- Module-level docstrings with date, author, description

### Dependency Injection
- `now_fn`: Callable[[], datetime] - for time mocking
- `sleep_fn`: Callable[[float], None] - for delay mocking
- `id_fn`: Callable[[], str] - for ID generation mocking
- `task_execution_func`: Optional custom execution function

## Testing Strategy

### Unit Testing with StubExecutor

```python
from src.executor import StubExecutor, CancellationToken
from src.config.models import RunManifest

# Create stub executor with 100% failure rate
stub = StubExecutor(simulated_delay=0.01, failure_rate=1.0)

# Or configure specific task behaviors
stub = StubExecutor(task_behaviors={
    "task-1": "success",
    "task-2": "timeout",
    "task-3": "failure"
})

# Execute manifest
result = stub.run_manifest(manifest)

# Verify results
assert result.statistics.failed_tasks == expected_failures
```

### Integration Testing with Custom Execution Function

```python
from src.executor import ConcurrentExecutor

def my_execution_func(task):
    # Real Dify API call logic here
    response = dify_client.run_workflow(...)
    return response.data

executor = ConcurrentExecutor(task_execution_func=my_execution_func)
result = executor.run_manifest(manifest)
```

### Cancellation Testing

```python
from src.executor import StubExecutor, CancellationToken
import threading

token = CancellationToken()
stub = StubExecutor(simulated_delay=10.0)  # Long delay

# Cancel after 1 second
def cancel_later():
    time.sleep(1.0)
    token.cancel()

threading.Thread(target=cancel_later).start()
result = stub.run_manifest(manifest, cancellation_token=token)

# Verify tasks were cancelled
assert result.statistics.cancelled_tasks > 0
```

## Performance Characteristics

### Concurrency
- Thread pool size controlled by `ExecutionPolicy.concurrency`
- Typical values: 1-10 threads (I/O-bound tasks)
- No GIL contention for network I/O operations

### Memory Usage
- O(n) where n = number of tasks
- Each task holds test case data + parameters
- Task results accumulated in list

### Scalability Limits
- Python threading: ~100-500 concurrent threads practical limit
- For higher concurrency, consider asyncio-based executor in future phases

## Error Recovery

### Transient Failures
- Network timeouts → Retry with backoff
- Rate limit errors → Retry with backoff
- Temporary API errors → Retry with backoff

### Permanent Failures
- Invalid parameters → Fail immediately, no retry
- Authentication errors → Fail immediately
- Resource not found → Fail immediately

## Success Criteria Verification

- ✓ ConcurrentExecutor inherits ExecutorBase and implements _execute_tasks()
- ✓ Supports concurrent execution using ThreadPoolExecutor
- ✓ Supports retry mechanism with exponential backoff
- ✓ Supports timeout handling per task
- ✓ Supports cancellation token for graceful shutdown
- ✓ StubExecutor configurable with simulated behaviors
- ✓ Code follows PEP 8 and Google-style docstrings
- ✓ All exports updated in __init__.py

## Future Enhancements (Out of Scope for Phase 2)

1. **AsyncIO-based executor** for higher concurrency
2. **Progress callbacks** for real-time status updates
3. **Task prioritization** based on dataset weights
4. **Batch execution** respecting batch_size from ExecutionPolicy
5. **Rate limiting integration** with RateLimit configuration
6. **Result caching** to avoid redundant executions
7. **TestResult conversion** from execution output (currently returns None)

## Files Changed

1. **New:** `src/executor/concurrent_executor.py`
2. **New:** `src/executor/stub_executor.py`
3. **Modified:** `src/executor/__init__.py`

## Dependencies

### Standard Library
- `concurrent.futures`: ThreadPoolExecutor, Future, as_completed
- `datetime`: datetime
- `time`: sleep, time
- `typing`: Callable, List, Optional, Dict, Any
- `uuid`: uuid4
- `random`: random (StubExecutor only)

### Project Modules
- `src.config.models`: RunManifest, ExecutionPolicy, RetryPolicy
- `src.utils.exceptions`: TaskExecutionException, TaskTimeoutException, SchedulerException
- `src.executor.executor_base`: ExecutorBase
- `src.executor.models`: Task, TaskResult, TaskStatus, CancellationToken

## Summary

Phase 2 successfully implements production-ready concurrent execution infrastructure with:
- Thread-safe concurrent execution
- Robust retry and timeout mechanisms
- Graceful cancellation support
- Comprehensive test stub for unit testing
- Full compliance with project code quality standards

The implementation is ready for integration with the collector module and real Dify API calls.
