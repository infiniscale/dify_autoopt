"""
DataCollector 性能测试

Date: 2025-11-13
Author: backend-developer
Description: 测试 DataCollector 的性能指标
"""

import sys
import time
from datetime import datetime

sys.path.insert(0, '.')

from src.collector import DataCollector, TestResult, TestStatus


def test_large_dataset_performance():
    """测试大数据集性能 (10,000 条结果)"""
    print("=" * 60)
    print("Performance Test: 10,000 Results")
    print("=" * 60)

    collector = DataCollector()

    # 测试 1: collect_result() 性能
    print("\n1. Testing collect_result() performance...")
    start_time = time.time()

    for i in range(10000):
        result = TestResult(
            workflow_id=f"wf_{i % 10:03d}",  # 10 个不同工作流
            execution_id=f"exec_{i:05d}",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS if i % 3 != 0 else TestStatus.FAILED,
            execution_time=1.0 + (i % 100) * 0.1,
            tokens_used=100 + (i % 50) * 10,
            cost=0.01 + (i % 20) * 0.001,
            inputs={"query": f"test_{i}"},
            outputs={"answer": f"result_{i}"},
            prompt_variant=f"v{i % 5}",
            dataset=f"dataset_{i % 3}"
        )
        collector.collect_result(result)

    collect_time = time.time() - start_time
    print(f"   - Collected 10,000 results in {collect_time:.3f}s")
    print(f"   - Average time per collect: {collect_time / 10000 * 1000:.3f}ms")
    print(f"   - Throughput: {10000 / collect_time:.0f} results/second")

    # 测试 2: get_statistics() 性能
    print("\n2. Testing get_statistics() performance...")
    start_time = time.time()

    metrics = collector.get_statistics()

    stats_time = time.time() - start_time
    print(f"   - Calculated statistics in {stats_time:.3f}s")
    print(f"   - Total executions: {metrics.total_executions}")
    print(f"   - Success rate: {metrics.success_rate:.2%}")
    print(f"   - Avg execution time: {metrics.avg_execution_time:.2f}s")
    print(f"   - P50: {metrics.p50_execution_time:.2f}s")
    print(f"   - P95: {metrics.p95_execution_time:.2f}s")
    print(f"   - P99: {metrics.p99_execution_time:.2f}s")

    # 测试 3: 工作流查询性能
    print("\n3. Testing get_results_by_workflow() performance...")
    start_time = time.time()

    wf_results = collector.get_results_by_workflow("wf_005")

    query_time = time.time() - start_time
    print(f"   - Retrieved {len(wf_results)} results in {query_time:.3f}s")
    print(f"   - Query time: {query_time * 1000:.3f}ms")

    # 测试 4: 变体查询性能
    print("\n4. Testing get_results_by_variant() performance...")
    start_time = time.time()

    variant_results = collector.get_results_by_variant("wf_005", "v2")

    query_time = time.time() - start_time
    print(f"   - Retrieved {len(variant_results)} results in {query_time:.3f}s")
    print(f"   - Query time: {query_time * 1000:.3f}ms")

    # 测试 5: 数据集查询性能
    print("\n5. Testing get_results_by_dataset() performance...")
    start_time = time.time()

    dataset_results = collector.get_results_by_dataset("dataset_1")

    query_time = time.time() - start_time
    print(f"   - Retrieved {len(dataset_results)} results in {query_time:.3f}s")
    print(f"   - Query time: {query_time * 1000:.3f}ms")

    # 测试 6: 内存占用估算
    print("\n6. Memory usage estimation...")
    import sys

    result_size = sys.getsizeof(collector._results)
    index_size = sys.getsizeof(collector._results_by_workflow)
    total_size = result_size + index_size

    print(f"   - Results list size: {result_size / 1024:.2f} KB")
    print(f"   - Workflow index size: {index_size / 1024:.2f} KB")
    print(f"   - Total estimated size: {total_size / 1024:.2f} KB")
    print(f"   - Avg size per result: {total_size / 10000:.2f} bytes")

    print("\n" + "=" * 60)
    print("Performance Test Complete!")
    print("=" * 60)

    # 性能要求验证
    print("\nPerformance Requirements Validation:")
    print(f"[OK] Supports 10,000+ results: {collector.get_result_count() >= 10000}")
    print(f"[OK] collect_result() < 1ms: {collect_time / 10000 * 1000 < 1.0}")
    print(f"[OK] get_statistics() < 1s: {stats_time < 1.0}")


if __name__ == "__main__":
    test_large_dataset_performance()
