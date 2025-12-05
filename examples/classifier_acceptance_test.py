"""
ResultClassifier 验收测试脚本

Date: 2025-11-13
Author: backend-developer
Description: 演示 ResultClassifier 的核心功能
"""

from datetime import datetime
from src.collector import (
    ResultClassifier,
    TestResult,
    TestStatus,
    PerformanceGrade,
)


def main():
    print("=" * 60)
    print("ResultClassifier 验收测试")
    print("=" * 60)

    # 创建分类器
    classifier = ResultClassifier()
    print("\n1. 初始化分类器 (使用默认阈值)")
    print(f"   默认阈值: {classifier.get_thresholds()}")

    # 测试 1: 优秀结果
    print("\n2. 测试优秀结果分类")
    excellent_result = TestResult(
        workflow_id="wf_001",
        execution_id="exec_001",
        timestamp=datetime.now(),
        status=TestStatus.SUCCESS,
        execution_time=1.0,  # < 2s
        tokens_used=100,
        cost=0.01,
        inputs={"query": "test"},
        outputs={"answer": "x" * 500},  # 高效输出
    )

    grade = classifier.classify_result(excellent_result)
    print(f"   执行时间: {excellent_result.execution_time}s")
    print(f"   Tokens: {excellent_result.tokens_used}")
    print(f"   输出长度: {len(str(excellent_result.outputs))}")
    print(f"   [OK] 分级结果: {grade.value.upper()}")
    assert grade == PerformanceGrade.EXCELLENT

    # 测试 2: 批量分类
    print("\n3. 测试批量分类")
    results = [
        # EXCELLENT
        TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 400},
        ),
        # GOOD
        TestResult(
            workflow_id="wf_001",
            execution_id="exec_002",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=3.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 300},
        ),
        # FAIR
        TestResult(
            workflow_id="wf_001",
            execution_id="exec_003",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=8.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 200},
        ),
        # POOR
        TestResult(
            workflow_id="wf_001",
            execution_id="exec_004",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=15.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 50},
        ),
    ]

    stats = classifier.classify_batch(results)
    print(f"   总结果数: {len(results)}")
    print(f"   EXCELLENT: {stats.excellent_count} ({stats.grade_distribution[PerformanceGrade.EXCELLENT]:.1f}%)")
    print(f"   GOOD: {stats.good_count} ({stats.grade_distribution[PerformanceGrade.GOOD]:.1f}%)")
    print(f"   FAIR: {stats.fair_count} ({stats.grade_distribution[PerformanceGrade.FAIR]:.1f}%)")
    print(f"   POOR: {stats.poor_count} ({stats.grade_distribution[PerformanceGrade.POOR]:.1f}%)")

    assert stats.excellent_count == 1
    assert stats.good_count == 1
    assert stats.fair_count == 1
    assert stats.poor_count == 1

    # 测试 3: 自定义阈值
    print("\n4. 测试自定义阈值")
    custom_thresholds = {
        "excellent": {"execution_time": 1.5, "token_efficiency": 0.85},
        "good": {"execution_time": 4.0, "token_efficiency": 0.65},
        "fair": {"execution_time": 9.0, "token_efficiency": 0.45},
    }
    classifier.set_thresholds(custom_thresholds)
    print(f"   新阈值: {custom_thresholds}")

    # 使用相同结果重新分类
    grade = classifier.classify_result(excellent_result)
    print(f"   [OK] 重新分级结果: {grade.value.upper()}")

    print("\n" + "=" * 60)
    print("[SUCCESS] 所有验收测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
