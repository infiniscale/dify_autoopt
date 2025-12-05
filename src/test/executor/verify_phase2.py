"""
Quick verification test for Phase 2 implementation

This script verifies that ConcurrentExecutor and StubExecutor can be imported
and instantiated without errors.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.executor import (
    ConcurrentExecutor,
    StubExecutor,
    TaskExecutionFunc,
    Task,
    TaskStatus,
    CancellationToken
)

def test_concurrent_executor_instantiation():
    """Test that ConcurrentExecutor can be instantiated"""
    print("Testing ConcurrentExecutor instantiation...")
    executor = ConcurrentExecutor()
    assert executor is not None
    print("✓ ConcurrentExecutor instantiated successfully")

def test_stub_executor_instantiation():
    """Test that StubExecutor can be instantiated"""
    print("\nTesting StubExecutor instantiation...")
    stub_executor = StubExecutor()
    assert stub_executor is not None
    print("✓ StubExecutor instantiated successfully")

def test_stub_executor_configuration():
    """Test StubExecutor configuration methods"""
    print("\nTesting StubExecutor configuration...")

    stub_executor = StubExecutor(
        simulated_delay=0.1,
        failure_rate=0.2,
        task_behaviors={"task-1": "success", "task-2": "failure"}
    )

    # Test setter methods
    stub_executor.set_failure_rate(0.5)
    stub_executor.set_simulated_delay(0.05)
    stub_executor.set_task_behavior("task-3", "timeout")
    stub_executor.clear_task_behaviors()

    print("✓ StubExecutor configuration methods work correctly")

def test_imports():
    """Test that all expected exports are available"""
    print("\nTesting module imports...")

    from src.executor import (
        ConcurrentExecutor,
        StubExecutor,
        TaskExecutionFunc,
        ExecutorBase,
        Task,
        TaskResult,
        TaskStatus,
        RunStatistics,
        RunExecutionResult,
        CancellationToken
    )

    print("✓ All expected exports are available")

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 Implementation Verification")
    print("=" * 60)

    try:
        test_imports()
        test_concurrent_executor_instantiation()
        test_stub_executor_instantiation()
        test_stub_executor_configuration()

        print("\n" + "=" * 60)
        print("✓ All verification tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
