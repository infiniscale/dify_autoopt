"""
测试 DataCollector 类

Date: 2025-11-13
Author: backend-developer
Description: DataCollector 完整功能测试
"""

import pytest
from datetime import datetime

from src.collector import DataCollector, TestResult, TestStatus, PerformanceMetrics
from src.utils.exceptions import DataValidationException


class TestDataCollectorBasics:
    """测试基础功能"""

    def test_initialization(self):
        """测试初始化"""
        collector = DataCollector()
        assert collector.get_result_count() == 0
        assert collector.get_all_results() == []

    def test_collect_single_result(self):
        """测试收集单个结果"""
        collector = DataCollector()

        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.5,
            tokens_used=150,
            cost=0.01,
            inputs={"query": "test1"},
            outputs={"answer": "result1"}
        )

        collector.collect_result(result)
        assert collector.get_result_count() == 1

        all_results = collector.get_all_results()
        assert len(all_results) == 1
        assert all_results[0].workflow_id == "wf_001"

    def test_collect_multiple_results(self):
        """测试收集多个结果"""
        collector = DataCollector()

        for i in range(5):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i:03d}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=1.0 + i * 0.5,
                tokens_used=100 + i * 10,
                cost=0.01 + i * 0.001,
                inputs={"query": f"test{i}"},
                outputs={"answer": f"result{i}"}
            )
            collector.collect_result(result)

        assert collector.get_result_count() == 5


class TestDataCollectorValidation:
    """测试数据验证"""

    def test_invalid_type(self):
        """测试无效类型"""
        collector = DataCollector()

        with pytest.raises(DataValidationException) as exc_info:
            collector.collect_result("not_a_result")

        assert "Expected TestResult" in str(exc_info.value)

    def test_empty_workflow_id(self):
        """测试空 workflow_id"""
        collector = DataCollector()

        result = TestResult(
            workflow_id="",  # 空字符串
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.5,
            tokens_used=150,
            cost=0.01,
            inputs={},
            outputs={}
        )

        with pytest.raises(DataValidationException) as exc_info:
            collector.collect_result(result)

        assert "workflow_id is required" in str(exc_info.value)

    def test_empty_execution_id(self):
        """测试空 execution_id"""
        collector = DataCollector()

        result = TestResult(
            workflow_id="wf_001",
            execution_id="",  # 空字符串
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.5,
            tokens_used=150,
            cost=0.01,
            inputs={},
            outputs={}
        )

        with pytest.raises(DataValidationException) as exc_info:
            collector.collect_result(result)

        assert "execution_id is required" in str(exc_info.value)

    def test_negative_execution_time(self):
        """测试负执行时间"""
        collector = DataCollector()

        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=-1.0,  # 负数
            tokens_used=150,
            cost=0.01,
            inputs={},
            outputs={}
        )

        with pytest.raises(DataValidationException) as exc_info:
            collector.collect_result(result)

        assert "execution_time must be non-negative" in str(exc_info.value)

    def test_negative_tokens(self):
        """测试负 Token 数"""
        collector = DataCollector()

        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.5,
            tokens_used=-100,  # 负数
            cost=0.01,
            inputs={},
            outputs={}
        )

        with pytest.raises(DataValidationException) as exc_info:
            collector.collect_result(result)

        assert "tokens_used must be non-negative" in str(exc_info.value)

    def test_negative_cost(self):
        """测试负成本"""
        collector = DataCollector()

        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.5,
            tokens_used=150,
            cost=-0.01,  # 负数
            inputs={},
            outputs={}
        )

        with pytest.raises(DataValidationException) as exc_info:
            collector.collect_result(result)

        assert "cost must be non-negative" in str(exc_info.value)


class TestDataCollectorStatistics:
    """测试统计计算"""

    def test_statistics_basic(self):
        """测试基础统计"""
        collector = DataCollector()

        # 添加测试数据
        result1 = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.5,
            tokens_used=150,
            cost=0.01,
            inputs={"query": "test1"},
            outputs={"answer": "result1"}
        )

        result2 = TestResult(
            workflow_id="wf_001",
            execution_id="exec_002",
            timestamp=datetime.now(),
            status=TestStatus.FAILED,
            execution_time=3.0,
            tokens_used=200,
            cost=0.02,
            inputs={"query": "test2"},
            outputs={},
            error_message="Timeout"
        )

        collector.collect_result(result1)
        collector.collect_result(result2)

        # 计算统计
        metrics = collector.get_statistics()

        assert metrics.total_executions == 2
        assert metrics.successful_count == 1
        assert metrics.failed_count == 1
        assert metrics.success_rate == 0.5
        assert metrics.avg_execution_time == 2.25  # (1.5 + 3.0) / 2
        assert metrics.total_tokens == 350  # 150 + 200
        assert metrics.total_cost == 0.03  # 0.01 + 0.02
        assert metrics.avg_tokens_per_request == 175.0  # 350 / 2

    def test_statistics_no_data(self):
        """测试无数据时的统计"""
        collector = DataCollector()

        with pytest.raises(DataValidationException) as exc_info:
            collector.get_statistics()

        assert "No results to calculate statistics" in str(exc_info.value)

    def test_statistics_by_workflow(self):
        """测试按工作流统计"""
        collector = DataCollector()

        # 添加不同工作流的数据
        for i in range(3):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_001_{i}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=1.0,
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={}
            )
            collector.collect_result(result)

        for i in range(2):
            result = TestResult(
                workflow_id="wf_002",
                execution_id=f"exec_002_{i}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=2.0,
                tokens_used=200,
                cost=0.02,
                inputs={},
                outputs={}
            )
            collector.collect_result(result)

        # 全部统计
        all_metrics = collector.get_statistics()
        assert all_metrics.total_executions == 5

        # wf_001 统计
        wf1_metrics = collector.get_statistics(workflow_id="wf_001")
        assert wf1_metrics.total_executions == 3
        assert wf1_metrics.avg_execution_time == 1.0

        # wf_002 统计
        wf2_metrics = collector.get_statistics(workflow_id="wf_002")
        assert wf2_metrics.total_executions == 2
        assert wf2_metrics.avg_execution_time == 2.0

    def test_percentiles_calculation(self):
        """测试百分位数计算"""
        collector = DataCollector()

        # 添加 10 个结果，执行时间从 1.0 到 10.0
        for i in range(1, 11):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i:03d}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=float(i),
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={}
            )
            collector.collect_result(result)

        metrics = collector.get_statistics()

        # P50 应该在 5.5 附近
        assert 5.0 <= metrics.p50_execution_time <= 6.0

        # P95 应该在 9.5 附近
        assert 9.0 <= metrics.p95_execution_time <= 10.0

        # P99 应该在 9.9 附近
        assert 9.5 <= metrics.p99_execution_time <= 10.0


class TestDataCollectorQueries:
    """测试数据查询"""

    def test_get_all_results(self):
        """测试获取所有结果"""
        collector = DataCollector()

        for i in range(3):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=1.0,
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={}
            )
            collector.collect_result(result)

        all_results = collector.get_all_results()
        assert len(all_results) == 3

        # 验证返回的是副本
        all_results.clear()
        assert collector.get_result_count() == 3

    def test_get_results_by_workflow(self):
        """测试按工作流查询"""
        collector = DataCollector()

        # 添加不同工作流的数据
        for i in range(2):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_001_{i}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=1.0,
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={}
            )
            collector.collect_result(result)

        for i in range(3):
            result = TestResult(
                workflow_id="wf_002",
                execution_id=f"exec_002_{i}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=2.0,
                tokens_used=200,
                cost=0.02,
                inputs={},
                outputs={}
            )
            collector.collect_result(result)

        wf1_results = collector.get_results_by_workflow("wf_001")
        assert len(wf1_results) == 2

        wf2_results = collector.get_results_by_workflow("wf_002")
        assert len(wf2_results) == 3

        # 查询不存在的工作流
        wf3_results = collector.get_results_by_workflow("wf_003")
        assert len(wf3_results) == 0

    def test_get_results_by_variant(self):
        """测试按变体查询"""
        collector = DataCollector()

        # 添加不同变体的数据
        for i in range(2):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=1.0,
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={},
                prompt_variant="v1"
            )
            collector.collect_result(result)

        for i in range(3):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i + 2}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=2.0,
                tokens_used=200,
                cost=0.02,
                inputs={},
                outputs={},
                prompt_variant="v2"
            )
            collector.collect_result(result)

        v1_results = collector.get_results_by_variant("wf_001", "v1")
        assert len(v1_results) == 2

        v2_results = collector.get_results_by_variant("wf_001", "v2")
        assert len(v2_results) == 3

        # 查询不存在的变体
        v3_results = collector.get_results_by_variant("wf_001", "v3")
        assert len(v3_results) == 0

    def test_get_results_by_dataset(self):
        """测试按数据集查询"""
        collector = DataCollector()

        # 添加不同数据集的数据
        for i in range(2):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=1.0,
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={},
                dataset="dataset_a"
            )
            collector.collect_result(result)

        for i in range(3):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i + 2}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=2.0,
                tokens_used=200,
                cost=0.02,
                inputs={},
                outputs={},
                dataset="dataset_b"
            )
            collector.collect_result(result)

        ds_a_results = collector.get_results_by_dataset("dataset_a")
        assert len(ds_a_results) == 2

        ds_b_results = collector.get_results_by_dataset("dataset_b")
        assert len(ds_b_results) == 3

        # 查询不存在的数据集
        ds_c_results = collector.get_results_by_dataset("dataset_c")
        assert len(ds_c_results) == 0


class TestDataCollectorClear:
    """测试清空功能"""

    def test_clear(self):
        """测试清空数据"""
        collector = DataCollector()

        # 添加数据
        for i in range(5):
            result = TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=1.0,
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={}
            )
            collector.collect_result(result)

        assert collector.get_result_count() == 5

        # 清空
        collector.clear()

        assert collector.get_result_count() == 0
        assert collector.get_all_results() == []
        assert collector.get_results_by_workflow("wf_001") == []


class TestPercentileEdgeCases:
    """测试百分位数边界情况"""

    def test_empty_list(self):
        """测试空列表"""
        collector = DataCollector()
        percentiles = collector._calculate_percentiles([])

        assert percentiles["p50"] == 0.0
        assert percentiles["p95"] == 0.0
        assert percentiles["p99"] == 0.0

    def test_single_value(self):
        """测试单个值"""
        collector = DataCollector()
        percentiles = collector._calculate_percentiles([5.0])

        assert percentiles["p50"] == 5.0
        assert percentiles["p95"] == 5.0
        assert percentiles["p99"] == 5.0

    def test_two_values(self):
        """测试两个值"""
        collector = DataCollector()
        percentiles = collector._calculate_percentiles([1.0, 10.0])

        # P50 应该在中间
        assert 4.0 <= percentiles["p50"] <= 6.0

        # P95 应该接近 10.0
        assert percentiles["p95"] >= 9.0


if __name__ == "__main__":
    # 快速验证测试
    print("Running basic validation tests...")

    from datetime import datetime
    from src.collector import DataCollector, TestResult, TestStatus

    # 测试 1: 基础功能
    collector = DataCollector()

    result1 = TestResult(
        workflow_id="wf_001",
        execution_id="exec_001",
        timestamp=datetime.now(),
        status=TestStatus.SUCCESS,
        execution_time=1.5,
        tokens_used=150,
        cost=0.01,
        inputs={"query": "test1"},
        outputs={"answer": "result1"}
    )

    result2 = TestResult(
        workflow_id="wf_001",
        execution_id="exec_002",
        timestamp=datetime.now(),
        status=TestStatus.FAILED,
        execution_time=3.0,
        tokens_used=200,
        cost=0.02,
        inputs={"query": "test2"},
        outputs={},
        error_message="Timeout"
    )

    # 收集结果
    collector.collect_result(result1)
    collector.collect_result(result2)

    # 测试 2: 统计计算
    metrics = collector.get_statistics()
    assert metrics.total_executions == 2
    assert metrics.successful_count == 1
    assert metrics.failed_count == 1
    assert metrics.success_rate == 0.5

    # 测试 3: 数据查询
    all_results = collector.get_all_results()
    assert len(all_results) == 2

    wf_results = collector.get_results_by_workflow("wf_001")
    assert len(wf_results) == 2

    # 测试 4: 清空数据
    collector.clear()
    assert collector.get_result_count() == 0

    print("All validation tests passed!")
