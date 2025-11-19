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
# Executor Module - Phase 3 Implementation

## Overview

Phase 3 of the executor module implements **TaskScheduler** and **RateLimiter** to provide advanced scheduling capabilities including batch management, rate limiting, and stop conditions.

**Date:** 2025-11-14
**Author:** backend-developer
**Status:** Completed

---

## Implementation Summary

### 1. RateLimiter (`src/executor/rate_limiter.py`)

**Purpose:** Implements token bucket algorithm for rate limiting

**Features:**
- Token bucket algorithm with configurable rate and burst
- Thread-safe operations using `threading.Lock`
- Blocking (`acquire`) and non-blocking (`try_acquire`) token acquisition
- Dependency injection for `now_fn` and `sleep_fn` (testability)

**Algorithm:**
- Bucket capacity = `burst`
- Refill rate = `per_minute / 60.0` tokens/second
- Each request consumes tokens
- Waits when tokens insufficient

**Key Methods:**
- `acquire(tokens)` - Blocks until tokens available
- `try_acquire(tokens)` - Non-blocking attempt to acquire
- `available_tokens` - Property to check current token count

**Thread Safety:**
- All state modifications protected by `threading.Lock`
- Safe for concurrent access from multiple threads

---

### 2. TaskScheduler (`src/executor/task_scheduler.py`)

**Purpose:** Enhanced concurrent executor with batch management and rate limiting

**Inheritance:**
- Extends `ConcurrentExecutor`
- Overrides `_execute_tasks()` to add scheduling features

**Features:**
- **Batch Management:** Splits tasks into batches based on `batch_size`
- **Rate Limiting:** Uses `RateLimiter` to control execution rate
- **Batch Backoff:** Configurable delay between batches (`backoff_seconds`)
- **Stop Conditions:** Early termination based on:
  - `max_failures` - Maximum allowed failures
  - `timeout` - Total execution timeout

**Execution Flow:**
1. Split tasks into batches
2. For each batch:
   - Check stop conditions
   - Check cancellation token
   - Acquire rate limit tokens
   - Execute batch (via parent class)
   - Apply inter-batch backoff
3. Aggregate results from all batches

**Key Methods:**
- `_execute_tasks()` - Main scheduling logic (overrides parent)
- `_split_into_batches()` - Batch creation
- `_should_stop()` - Stop condition evaluation

**Integration:**
- Fully compatible with `ExecutionPolicy` from config module
- Reuses retry and timeout logic from `ConcurrentExecutor`
- Uses existing exception classes from `src/utils/exceptions.py`

---

## Code Quality Standards

### PEP 8 Compliance
- 4-space indentation
- Maximum line length: 100 characters
- Proper naming conventions (snake_case for functions/variables)
- No trailing whitespace

### Documentation
- Google-style docstrings for all classes and methods
- Complete parameter descriptions
- Return type documentation
- Raises section for exceptions

### Type Hints
- All function parameters typed
- All return types specified
- Optional types properly annotated

### Dependency Injection
- `now_fn: Callable[[], datetime]` - Time function
- `sleep_fn: Callable[[float], None]` - Sleep function
- `id_fn: Callable[[], str]` - ID generation function
- Enables comprehensive unit testing

---

## Integration Points

### With Config Module
```python
from src.config.models import RateLimit, ExecutionPolicy

# RateLimiter uses RateLimit config
rate_limit = RateLimit(per_minute=60, burst=5)
limiter = RateLimiter(rate_limit=rate_limit)

# TaskScheduler uses ExecutionPolicy
scheduler = TaskScheduler()
results = scheduler.execute(tasks, manifest)  # manifest contains ExecutionPolicy
```

### With Phase 1 & 2 Modules
```python
from src.executor import (
    Task,                  # Phase 1: Task model
    TaskResult,            # Phase 1: Result model
    CancellationToken,     # Phase 1: Cancellation
    ConcurrentExecutor,    # Phase 2: Base executor
    TaskScheduler,         # Phase 3: Enhanced scheduler
    RateLimiter           # Phase 3: Rate limiting
)
```

### Exception Handling
Uses existing exception classes:
- `TaskExecutionException` - Business logic failures
- `TaskTimeoutException` - Task timeouts
- `SchedulerException` - Scheduling failures

---

## File Structure

```
src/executor/
├── models.py                 # Phase 1: Core models
├── executor_base.py          # Phase 1: Abstract base class
├── concurrent_executor.py    # Phase 2: Concurrent execution
├── stub_executor.py          # Phase 2: Test stub
├── rate_limiter.py          # Phase 3: Rate limiting (NEW)
├── task_scheduler.py        # Phase 3: Advanced scheduling (NEW)
└── __init__.py              # Updated exports
```

---

## Exports Update

### Updated `src/executor/__init__.py`

Added Phase 3 exports:
```python
from .rate_limiter import RateLimiter
from .task_scheduler import TaskScheduler

__all__ = [
    # ... existing exports ...
    'RateLimiter',
    'TaskScheduler',
]
```

---

## Success Criteria Verification

- ✅ RateLimiter implements token bucket algorithm
- ✅ RateLimiter is thread-safe (uses `threading.Lock`)
- ✅ TaskScheduler inherits from `ConcurrentExecutor`
- ✅ TaskScheduler supports batch management (`_split_into_batches`)
- ✅ TaskScheduler supports rate limiting (via `RateLimiter`)
- ✅ TaskScheduler supports stop conditions (`_should_stop`)
- ✅ TaskScheduler supports batch backoff (`backoff_seconds`)
- ✅ Code follows PEP 8 conventions
- ✅ Google-style docstrings throughout
- ✅ All exports updated in `__init__.py`

---

## Testing

### Basic Functionality Tests

**RateLimiter:**
```python
from src.config.models import RateLimit
from src.executor import RateLimiter

rate_limit = RateLimit(per_minute=60, burst=5)
limiter = RateLimiter(rate_limit=rate_limit)

# Check initial state
assert limiter.available_tokens == 5.0

# Test acquire
result = limiter.try_acquire(1)
assert result == True
assert limiter.available_tokens < 5.0
```

**TaskScheduler:**
```python
from src.executor import TaskScheduler

scheduler = TaskScheduler()

# Test batch splitting
tasks = ['task_1', 'task_2', ..., 'task_12']
batches = scheduler._split_into_batches(tasks, batch_size=5)

assert len(batches) == 3
assert len(batches[0]) == 5
assert len(batches[1]) == 5
assert len(batches[2]) == 2
```

---

## Implementation Notes

### RateLimiter Design Decisions

1. **Initial Token State:** Bucket starts full (`burst` tokens) for immediate availability
2. **Refill Calculation:** Continuous refill based on elapsed time (not periodic)
3. **Blocking Behavior:** `acquire()` uses `sleep_fn` in loop until tokens available
4. **Thread Safety:** Single lock protects all state modifications

### TaskScheduler Design Decisions

1. **Parent Class Reuse:** Calls `super()._execute_tasks()` for individual batch execution
2. **Stop Condition Timing:** Checked before each batch (not during batch execution)
3. **Cancelled Task Handling:** Remaining tasks marked as `CANCELLED` when stop conditions met
4. **Rate Limit Granularity:** Tokens acquired per batch (not per task)

### Performance Considerations

1. **Lock Contention:** RateLimiter uses single lock (acceptable for current use case)
2. **Batch Size:** Configurable via `ExecutionPolicy.batch_size`
3. **Memory:** All results accumulated in memory (fine for expected task counts)

---

## Dependencies

**Standard Library:**
- `time` - Sleep functionality
- `threading` - Thread safety (Lock)
- `datetime` - Time operations
- `typing` - Type hints
- `uuid` - ID generation

**Project Modules:**
- `src.config.models` - Configuration models (RateLimit, ExecutionPolicy)
- `src.utils.exceptions` - Exception classes
- `src.executor.concurrent_executor` - Base executor
- `src.executor.models` - Task models

---

## Future Enhancements

Potential improvements for future phases:
1. **Adaptive Rate Limiting:** Adjust rate based on error rates
2. **Priority-Based Scheduling:** Support task priorities
3. **Resource-Based Throttling:** Limit by memory/CPU usage
4. **Distributed Rate Limiting:** Share rate limit across multiple instances
5. **Metrics Collection:** Track batch performance, rate limit hit rates

---

## Summary

Phase 3 successfully implements advanced task scheduling capabilities:

- **RateLimiter:** 150 lines, thread-safe token bucket algorithm
- **TaskScheduler:** 200 lines, batch management with rate limiting and stop conditions
- **Integration:** Seamlessly integrates with Phase 1/2 and config module
- **Quality:** Full PEP 8 compliance, comprehensive docstrings, type hints
- **Testing:** Dependency injection enables easy unit testing

The implementation is production-ready and follows all project standards.
# Executor Module - Phase 4 Usage Guide

## Overview

Phase 4 introduces **ResultConverter** and **ExecutorService**, completing the integration between the executor module and the collector module.

## Components

### 1. ResultConverter

A static utility class that converts `TaskResult` (executor) to `TestResult` (collector).

#### Status Mapping

| TaskStatus (8 states) | TestStatus (4 states) |
|----------------------|----------------------|
| SUCCEEDED            | SUCCESS              |
| FAILED               | FAILED               |
| TIMEOUT              | TIMEOUT              |
| ERROR                | ERROR                |
| CANCELLED            | ERROR                |
| PENDING              | ERROR                |
| QUEUED               | ERROR                |
| RUNNING              | ERROR                |

#### Usage Example

```python
from src.executor import ResultConverter
from src.executor.models import TaskResult, TaskStatus

# Single conversion
task_result = TaskResult(...)
test_result = ResultConverter.convert(task_result)

# Batch conversion
task_results = [...]
test_results = ResultConverter.convert_batch(task_results)
```

#### Fields Extracted

- **workflow_id**: From TaskResult.workflow_id
- **execution_id**: From TaskResult.task_id
- **timestamp**: From TaskResult.created_at
- **status**: Mapped from TaskResult.task_status
- **execution_time**: From TaskResult.execution_time
- **tokens_used**: From embedded test_result or 0
- **cost**: From embedded test_result or 0.0
- **inputs**: From embedded test_result or {}
- **outputs**: From embedded test_result or {}
- **error_message**: From TaskResult.error_message
- **dataset**: From TaskResult.dataset
- **metadata**: Includes dataset, scenario, attempt_count

### 2. ExecutorService

A high-level service that integrates TaskScheduler and ResultConverter.

#### Purpose

Provides a unified entry point for test execution:
1. Accepts RunManifest
2. Executes tasks via TaskScheduler
3. Converts results via ResultConverter
4. Returns List[TestResult]

#### Usage Example

```python
from src.executor import ExecutorService
from src.config.models import RunManifest

# Initialize service
service = ExecutorService()

# Execute test plan
manifest = RunManifest(...)
test_results = service.execute_test_plan(manifest)

# Process results
for result in test_results:
    print(f"Test {result.execution_id}: {result.status}")
    print(f"  Time: {result.execution_time}s")
    print(f"  Tokens: {result.tokens_used}")
    print(f"  Cost: ${result.cost}")
```

#### With Cancellation Token

```python
from src.executor.models import CancellationToken

token = CancellationToken()
test_results = service.execute_test_plan(manifest, token)

# Cancel from another thread
token.cancel()
```

#### With Custom Execution Function

```python
from src.collector.models import TestResult, TestStatus

def my_executor(task):
    # Custom execution logic
    return TestResult(
        workflow_id=task.workflow_id,
        execution_id=task.task_id,
        timestamp=datetime.now(),
        status=TestStatus.SUCCESS,
        execution_time=1.0,
        tokens_used=100,
        cost=0.002,
        inputs=task.parameters,
        outputs={"result": "success"}
    )

service = ExecutorService(task_execution_func=my_executor)
test_results = service.execute_test_plan(manifest)
```

## Integration Flow

```
RunManifest
    ↓
ExecutorService.execute_test_plan()
    ↓
TaskScheduler.run_manifest()
    ↓
RunExecutionResult (contains List[TaskResult])
    ↓
ResultConverter.convert_batch()
    ↓
List[TestResult] (ready for Collector module)
```

## Testing

### ResultConverter Tests

- Status mapping (all 8 → 4 mappings)
- Field extraction (tokens, cost, error_message)
- Batch conversion
- Error handling (None inputs)
- Metadata preservation

### ExecutorService Tests

- Initialization (default and custom functions)
- Test plan execution
- Cancellation support
- Empty manifest handling
- Integration with ResultConverter
- Large batch execution (100+ tests)

## Error Handling

### ResultConverter

```python
# Raises ValueError if task_result is None
try:
    result = ResultConverter.convert(None)
except ValueError as e:
    print(f"Error: {e}")  # "task_result cannot be None"

# Raises ValueError if batch is None
try:
    results = ResultConverter.convert_batch(None)
except ValueError as e:
    print(f"Error: {e}")  # "task_results cannot be None"
```

### ExecutorService

```python
# Raises ValueError if manifest is None
try:
    service.execute_test_plan(None)
except ValueError as e:
    print(f"Error: {e}")  # "manifest cannot be None"

# Raises ExecutorException if manifest.cases is empty
from src.utils.exceptions import ExecutorException

try:
    empty_manifest = RunManifest(workflow_id="test", cases=[], ...)
    service.execute_test_plan(empty_manifest)
except ExecutorException as e:
    print(f"Error: {e}")  # "RunManifest.cases cannot be empty"
```

## Best Practices

1. **Use ExecutorService as the main entry point**
   - Don't call TaskScheduler directly unless you need advanced control
   - ExecutorService handles both execution and conversion

2. **Handle all TestStatus states**
   - SUCCESS: Test passed
   - FAILED: Business logic failure
   - TIMEOUT: Execution timeout
   - ERROR: System error, cancellation, or non-terminal states

3. **Check execution_time for performance analysis**
   - Calculated from task start/finish timestamps
   - May be 0.0 if task was never started

4. **Use metadata for debugging**
   - Contains dataset, scenario, attempt_count
   - Useful for tracing test execution

5. **Leverage dependency injection for testing**
   ```python
   # Mock time for deterministic tests
   mock_now = lambda: datetime(2025, 11, 14, 10, 0, 0)
   mock_sleep = lambda x: None
   mock_id = lambda: "test_id_001"

   service = ExecutorService(
       now_fn=mock_now,
       sleep_fn=mock_sleep,
       id_fn=mock_id
   )
   ```

## File Locations

- **Implementation**:
  - `src/executor/result_converter.py`
  - `src/executor/executor_service.py`
  - `src/executor/__init__.py` (exports)

- **Tests**:
  - `src/test/executor/test_result_converter.py` (12 tests)
  - `src/test/executor/test_executor_service.py` (11 tests)

- **Documentation**:
  - `docs/executor/phase4_usage.md` (this file)

## Next Steps

Phase 4 completes the executor module. The module is now ready for integration with:

1. **Collector Module**: Use TestResult objects for reporting
2. **Evaluator Module**: Analyze test results and performance
3. **API Layer**: Expose execution capabilities via REST API

For production use, configure:
- Rate limiting (via ExecutionPolicy.rate_control)
- Retry policies (via ExecutionPolicy.retry_policy)
- Batch sizes (via ExecutionPolicy.batch_size)
- Concurrency limits (via ExecutionPolicy.concurrency)
