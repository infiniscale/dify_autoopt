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
