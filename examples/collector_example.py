"""
DataCollector 快速上手示例

Date: 2025-11-13
Author: backend-developer
Description: DataCollector 类的快速使用示例
"""

import sys
sys.path.insert(0, '.')

from datetime import datetime
from src.collector import DataCollector, TestResult, TestStatus


def example_basic_usage():
    """示例 1: 基础使用"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # 创建收集器
    collector = DataCollector()

    # 创建测试结果
    result = TestResult(
        workflow_id="wf_001",
        execution_id="exec_001",
        timestamp=datetime.now(),
        status=TestStatus.SUCCESS,
        execution_time=1.5,
        tokens_used=150,
        cost=0.01,
        inputs={"query": "What is AI?"},
        outputs={"answer": "AI is artificial intelligence."}
    )

    # 收集结果
    collector.collect_result(result)

    # 获取统计
    metrics = collector.get_statistics()
    print(f"\nTotal executions: {metrics.total_executions}")
    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Average execution time: {metrics.avg_execution_time:.2f}s")
    print()


def example_batch_collection():
    """示例 2: 批量收集和统计"""
    print("=" * 60)
    print("Example 2: Batch Collection")
    print("=" * 60)

    collector = DataCollector()

    # 批量收集 50 个结果
    for i in range(50):
        result = TestResult(
            workflow_id="wf_001",
            execution_id=f"exec_{i:03d}",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS if i % 3 != 0 else TestStatus.FAILED,
            execution_time=1.0 + (i % 10) * 0.2,
            tokens_used=100 + (i % 20) * 5,
            cost=0.01 + (i % 10) * 0.002,
            inputs={"query": f"Question {i}"},
            outputs={"answer": f"Answer {i}"}
        )
        collector.collect_result(result)

    # 计算统计
    metrics = collector.get_statistics()
    print(f"\nTotal executions: {metrics.total_executions}")
    print(f"Success rate: {metrics.success_rate:.2%}")
    print(f"Average execution time: {metrics.avg_execution_time:.2f}s")
    print(f"P50 execution time: {metrics.p50_execution_time:.2f}s")
    print(f"P95 execution time: {metrics.p95_execution_time:.2f}s")
    print(f"P99 execution time: {metrics.p99_execution_time:.2f}s")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Total cost: ${metrics.total_cost:.2f}")
    print()


def example_multi_workflow():
    """示例 3: 多工作流统计"""
    print("=" * 60)
    print("Example 3: Multi-Workflow Statistics")
    print("=" * 60)

    collector = DataCollector()

    # 添加不同工作流的数据
    workflows = ["wf_001", "wf_002", "wf_003"]
    for wf_id in workflows:
        for i in range(20):
            result = TestResult(
                workflow_id=wf_id,
                execution_id=f"{wf_id}_exec_{i:03d}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS if i % 2 == 0 else TestStatus.FAILED,
                execution_time=1.0 + i * 0.1,
                tokens_used=100 + i * 10,
                cost=0.01 + i * 0.001,
                inputs={"query": f"test {i}"},
                outputs={"answer": f"result {i}"}
            )
            collector.collect_result(result)

    # 全局统计
    all_metrics = collector.get_statistics()
    print(f"\n[All Workflows]")
    print(f"  Total executions: {all_metrics.total_executions}")
    print(f"  Success rate: {all_metrics.success_rate:.2%}")

    # 各工作流统计
    for wf_id in workflows:
        metrics = collector.get_statistics(workflow_id=wf_id)
        print(f"\n[{wf_id}]")
        print(f"  Executions: {metrics.total_executions}")
        print(f"  Success rate: {metrics.success_rate:.2%}")
        print(f"  Avg time: {metrics.avg_execution_time:.2f}s")
    print()


def example_query_operations():
    """示例 4: 数据查询操作"""
    print("=" * 60)
    print("Example 4: Query Operations")
    print("=" * 60)

    collector = DataCollector()

    # 添加带有变体和数据集的结果
    for i in range(30):
        result = TestResult(
            workflow_id="wf_001",
            execution_id=f"exec_{i:03d}",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0 + i * 0.05,
            tokens_used=100,
            cost=0.01,
            inputs={"query": f"test {i}"},
            outputs={"answer": f"result {i}"},
            prompt_variant=f"v{i % 3}",  # v0, v1, v2
            dataset=f"dataset_{i % 2}"  # dataset_0, dataset_1
        )
        collector.collect_result(result)

    # 查询所有结果
    all_results = collector.get_all_results()
    print(f"\nTotal results: {len(all_results)}")

    # 按工作流查询
    wf_results = collector.get_results_by_workflow("wf_001")
    print(f"Results for wf_001: {len(wf_results)}")

    # 按变体查询
    v0_results = collector.get_results_by_variant("wf_001", "v0")
    v1_results = collector.get_results_by_variant("wf_001", "v1")
    v2_results = collector.get_results_by_variant("wf_001", "v2")
    print(f"Results for v0: {len(v0_results)}")
    print(f"Results for v1: {len(v1_results)}")
    print(f"Results for v2: {len(v2_results)}")

    # 按数据集查询
    ds0_results = collector.get_results_by_dataset("dataset_0")
    ds1_results = collector.get_results_by_dataset("dataset_1")
    print(f"Results for dataset_0: {len(ds0_results)}")
    print(f"Results for dataset_1: {len(ds1_results)}")
    print()


def example_error_handling():
    """示例 5: 错误处理"""
    print("=" * 60)
    print("Example 5: Error Handling")
    print("=" * 60)

    from src.utils.exceptions import DataValidationException

    collector = DataCollector()

    # 测试 1: 无效类型
    print("\n1. Testing invalid type...")
    try:
        collector.collect_result("not a TestResult")
    except DataValidationException as e:
        print(f"   Caught error: {e}")

    # 测试 2: 空 workflow_id
    print("\n2. Testing empty workflow_id...")
    try:
        result = TestResult(
            workflow_id="",  # 空字符串
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={}
        )
        collector.collect_result(result)
    except DataValidationException as e:
        print(f"   Caught error: {e}")

    # 测试 3: 负执行时间
    print("\n3. Testing negative execution_time...")
    try:
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=-1.0,  # 负数
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={}
        )
        collector.collect_result(result)
    except DataValidationException as e:
        print(f"   Caught error: {e}")

    # 测试 4: 无数据统计
    print("\n4. Testing statistics with no data...")
    empty_collector = DataCollector()
    try:
        empty_collector.get_statistics()
    except DataValidationException as e:
        print(f"   Caught error: {e}")

    print("\nAll error handling examples completed!")
    print()


if __name__ == "__main__":
    # 运行所有示例
    example_basic_usage()
    example_batch_collection()
    example_multi_workflow()
    example_query_operations()
    example_error_handling()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
