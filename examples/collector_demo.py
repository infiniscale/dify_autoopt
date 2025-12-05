"""
DataCollector 模块演示示例

Date: 2025-11-13
Author: backend-developer
Description: 展示 DataCollector 的基本用法和功能
"""

import sys
from datetime import datetime

sys.path.insert(0, '.')

from src.collector import DataCollector, TestResult, TestStatus


def main():
    """主演示函数"""
    print("=" * 70)
    print(" DataCollector 模块演示")
    print("=" * 70)

    # 1. 创建收集器
    print("\n[Step 1] 创建 DataCollector 实例...")
    collector = DataCollector()
    print(f"[OK] DataCollector initialized, current count: {collector.get_result_count()}")

    # 2. 收集测试结果
    print("\n[Step 2] 收集测试结果...")
    test_data = [
        {
            "workflow_id": "wf_chatbot",
            "execution_id": "exec_001",
            "status": TestStatus.SUCCESS,
            "execution_time": 1.2,
            "tokens_used": 120,
            "cost": 0.012,
        },
        {
            "workflow_id": "wf_chatbot",
            "execution_id": "exec_002",
            "status": TestStatus.SUCCESS,
            "execution_time": 1.5,
            "tokens_used": 150,
            "cost": 0.015,
        },
        {
            "workflow_id": "wf_chatbot",
            "execution_id": "exec_003",
            "status": TestStatus.FAILED,
            "execution_time": 3.0,
            "tokens_used": 200,
            "cost": 0.020,
            "error_message": "Timeout error",
        },
        {
            "workflow_id": "wf_translation",
            "execution_id": "exec_004",
            "status": TestStatus.SUCCESS,
            "execution_time": 0.8,
            "tokens_used": 80,
            "cost": 0.008,
        },
        {
            "workflow_id": "wf_translation",
            "execution_id": "exec_005",
            "status": TestStatus.SUCCESS,
            "execution_time": 1.0,
            "tokens_used": 100,
            "cost": 0.010,
        },
    ]

    for data in test_data:
        result = TestResult(
            workflow_id=data["workflow_id"],
            execution_id=data["execution_id"],
            timestamp=datetime.now(),
            status=data["status"],
            execution_time=data["execution_time"],
            tokens_used=data["tokens_used"],
            cost=data["cost"],
            inputs={"query": f"test query for {data['execution_id']}"},
            outputs={"answer": f"test answer for {data['execution_id']}"},
            error_message=data.get("error_message"),
        )
        collector.collect_result(result)
        print(f"  [OK] Collected: {data['workflow_id']} - {data['execution_id']} ({data['status'].value})")

    print(f"\n[OK] Total results collected: {collector.get_result_count()}")

    # 3. 计算全局统计
    print("\n[Step 3] 计算全局统计...")
    metrics = collector.get_statistics()
    print(f"\n全局统计结果:")
    print(f"  Total executions: {metrics.total_executions}")
    print(f"  Successful: {metrics.successful_count}")
    print(f"  Failed: {metrics.failed_count}")
    print(f"  Success rate: {metrics.success_rate:.2%}")
    print(f"  Avg execution time: {metrics.avg_execution_time:.3f}s")
    print(f"  P50 execution time: {metrics.p50_execution_time:.3f}s")
    print(f"  P95 execution time: {metrics.p95_execution_time:.3f}s")
    print(f"  P99 execution time: {metrics.p99_execution_time:.3f}s")
    print(f"  Total tokens used: {metrics.total_tokens}")
    print(f"  Total cost: ${metrics.total_cost:.3f}")
    print(f"  Avg tokens per request: {metrics.avg_tokens_per_request:.1f}")

    # 4. 按工作流统计
    print("\n[Step 4] 按工作流统计...")
    for workflow_id in ["wf_chatbot", "wf_translation"]:
        metrics = collector.get_statistics(workflow_id=workflow_id)
        print(f"\n{workflow_id} 统计:")
        print(f"  Total: {metrics.total_executions}")
        print(f"  Success rate: {metrics.success_rate:.2%}")
        print(f"  Avg time: {metrics.avg_execution_time:.3f}s")
        print(f"  Total cost: ${metrics.total_cost:.3f}")

    # 5. 查询功能演示
    print("\n[Step 5] 数据查询演示...")

    # 按工作流查询
    chatbot_results = collector.get_results_by_workflow("wf_chatbot")
    print(f"\nwf_chatbot 结果数量: {len(chatbot_results)}")
    for r in chatbot_results:
        print(f"  - {r.execution_id}: {r.status.value}, {r.execution_time}s")

    # 获取所有结果
    all_results = collector.get_all_results()
    print(f"\n所有结果数量: {len(all_results)}")

    # 6. 数据验证演示
    print("\n[Step 6] 数据验证演示...")
    try:
        invalid_result = TestResult(
            workflow_id="",  # 空 workflow_id (无效)
            execution_id="exec_invalid",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={},
        )
        collector.collect_result(invalid_result)
    except Exception as e:
        print(f"[OK] Validation error caught: {e}")

    # 7. 清空数据
    print("\n[Step 7] 清空数据...")
    current_count = collector.get_result_count()
    print(f"Before clear: {current_count} results")
    collector.clear()
    print(f"After clear: {collector.get_result_count()} results")

    print("\n" + "=" * 70)
    print(" 演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    main()
