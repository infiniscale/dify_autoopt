"""
Unit tests for ResultClassifier

Date: 2025-11-13
Author: backend-developer
Description: 测试 ResultClassifier 类的所有功能，包括分级逻辑、批量统计、阈值管理等
"""

# 标准库
from datetime import datetime

# 第三方库
import pytest

# 项目内部
from src.collector import (
    ClassificationResult,
    PerformanceGrade,
    ResultClassifier,
    TestResult,
    TestStatus,
)
from src.utils.exceptions import ClassificationException


class TestResultClassifierInit:
    """测试 ResultClassifier 初始化"""

    def test_init_with_default_thresholds(self):
        """测试使用默认阈值初始化"""
        classifier = ResultClassifier()

        assert classifier._thresholds == ResultClassifier.DEFAULT_THRESHOLDS
        assert "excellent" in classifier._thresholds
        assert "good" in classifier._thresholds
        assert "fair" in classifier._thresholds

    def test_init_with_custom_thresholds(self):
        """测试使用自定义阈值初始化"""
        custom_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 0.9},
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        classifier = ResultClassifier(thresholds=custom_thresholds)

        assert classifier._thresholds == custom_thresholds
        assert classifier._thresholds["excellent"]["execution_time"] == 1.0

    def test_init_with_invalid_thresholds(self):
        """测试使用无效阈值初始化"""
        # 缺少 excellent
        invalid_thresholds = {
            "good": {"execution_time": 5.0, "token_efficiency": 0.6},
            "fair": {"execution_time": 10.0, "token_efficiency": 0.4},
        }

        with pytest.raises(ClassificationException) as exc_info:
            ResultClassifier(thresholds=invalid_thresholds)

        assert "Missing threshold for grade: excellent" in str(exc_info.value)


class TestCalculateTokenEfficiency:
    """测试 Token 效率计算"""

    def test_efficiency_with_zero_tokens(self):
        """测试 tokens_used 为 0 的情况"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=0,  # 零 tokens
            cost=0.0,
            inputs={},
            outputs={"answer": "test"},
        )

        efficiency = classifier._calculate_token_efficiency(result)
        assert efficiency == 0.0

    def test_efficiency_perfect_ratio(self):
        """测试完美比例 (1:4)"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 400},  # 400 字符 / 100 tokens = 4:1
        )

        efficiency = classifier._calculate_token_efficiency(result)
        assert efficiency == 1.0

    def test_efficiency_below_perfect(self):
        """测试低于完美比例"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 200},  # 200 字符 / 100 tokens = 2:1
        )

        efficiency = classifier._calculate_token_efficiency(result)
        # str({"answer": "x" * 200}) 会有额外的字符 (引号、冒号等)
        # 实际字符数 > 200，所以效率会稍高于 0.5
        assert 0.5 <= efficiency < 0.6

    def test_efficiency_above_perfect(self):
        """测试超过完美比例 (上限为 1.0)"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 600},  # 600 字符 / 100 tokens = 6:1
        )

        efficiency = classifier._calculate_token_efficiency(result)
        assert efficiency == 1.0  # 上限为 1.0

    def test_efficiency_with_empty_outputs(self):
        """测试空输出"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={},  # 空输出
        )

        efficiency = classifier._calculate_token_efficiency(result)
        # str({}) = '{}' 有 2 个字符
        expected = 2.0 / (100 * 4.0)
        assert efficiency == pytest.approx(expected, rel=1e-6)


class TestClassifyResult:
    """测试单个结果分级"""

    def test_classify_excellent_result(self):
        """测试优秀等级"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,  # < 2s
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 400},  # 效率 = 1.0 > 0.8
        )

        grade = classifier.classify_result(result)
        assert grade == PerformanceGrade.EXCELLENT

    def test_classify_good_result(self):
        """测试良好等级"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=3.0,  # < 5s
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 300},  # 效率 = 0.75 > 0.6
        )

        grade = classifier.classify_result(result)
        assert grade == PerformanceGrade.GOOD

    def test_classify_fair_result(self):
        """测试一般等级"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=8.0,  # < 10s
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 200},  # 效率 = 0.5 > 0.4
        )

        grade = classifier.classify_result(result)
        assert grade == PerformanceGrade.FAIR

    def test_classify_poor_result_slow_execution(self):
        """测试较差等级 - 执行时间慢"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=15.0,  # > 10s
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 400},  # 效率高但时间慢
        )

        grade = classifier.classify_result(result)
        assert grade == PerformanceGrade.POOR

    def test_classify_poor_result_low_efficiency(self):
        """测试较差等级 - Token 效率低"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,  # 时间快
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 50},  # 效率 = 0.125 < 0.4
        )

        grade = classifier.classify_result(result)
        assert grade == PerformanceGrade.POOR

    def test_classify_boundary_excellent_exact(self):
        """测试边界条件 - EXCELLENT 阈值精确匹配"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.999,  # < 2s
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 320},  # 效率 = 0.8 (精确)
        )

        grade = classifier.classify_result(result)
        assert grade == PerformanceGrade.EXCELLENT

    def test_classify_boundary_good_just_below(self):
        """测试边界条件 - GOOD 阈值刚好低于 EXCELLENT"""
        classifier = ResultClassifier()
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=2.0,  # >= 2s (不是 EXCELLENT)
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 320},  # 效率 = 0.8
        )

        grade = classifier.classify_result(result)
        assert grade == PerformanceGrade.GOOD

    def test_classify_invalid_result_type(self):
        """测试无效的结果类型"""
        classifier = ResultClassifier()

        with pytest.raises(ClassificationException) as exc_info:
            classifier.classify_result("not_a_result")

        assert "Expected TestResult" in str(exc_info.value)

    def test_classify_result_exception_handling(self):
        """测试分类过程中的异常处理"""
        classifier = ResultClassifier()

        # 创建一个会导致异常的结果对象
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "test"},
        )

        # Mock _calculate_token_efficiency 抛出异常
        def mock_error(self, r):
            raise ValueError("Mock error")

        original_method = classifier._calculate_token_efficiency
        classifier._calculate_token_efficiency = lambda r: mock_error(classifier, r)

        with pytest.raises(ClassificationException) as exc_info:
            classifier.classify_result(result)

        assert "Failed to classify result" in str(exc_info.value)

        # 恢复原方法
        classifier._calculate_token_efficiency = original_method


class TestClassifyBatch:
    """测试批量分类"""

    def test_classify_batch_mixed_results(self):
        """测试混合等级的批量分类"""
        classifier = ResultClassifier()

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

        assert isinstance(stats, ClassificationResult)
        assert stats.excellent_count == 1
        assert stats.good_count == 1
        assert stats.fair_count == 1
        assert stats.poor_count == 1

        # 验证分布 (每个 25%)
        assert stats.grade_distribution[PerformanceGrade.EXCELLENT] == 25.0
        assert stats.grade_distribution[PerformanceGrade.GOOD] == 25.0
        assert stats.grade_distribution[PerformanceGrade.FAIR] == 25.0
        assert stats.grade_distribution[PerformanceGrade.POOR] == 25.0

    def test_classify_batch_all_excellent(self):
        """测试全部优秀的情况"""
        classifier = ResultClassifier()

        results = [
            TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i:03d}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=1.0,
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={"answer": "x" * 400},
            )
            for i in range(5)
        ]

        stats = classifier.classify_batch(results)

        assert stats.excellent_count == 5
        assert stats.good_count == 0
        assert stats.fair_count == 0
        assert stats.poor_count == 0
        assert stats.grade_distribution[PerformanceGrade.EXCELLENT] == 100.0

    def test_classify_batch_empty_list(self):
        """测试空列表"""
        classifier = ResultClassifier()
        results = []

        stats = classifier.classify_batch(results)

        assert stats.excellent_count == 0
        assert stats.good_count == 0
        assert stats.fair_count == 0
        assert stats.poor_count == 0
        # 分布应该全为 0
        for grade in PerformanceGrade:
            assert stats.grade_distribution[grade] == 0.0

    def test_classify_batch_invalid_type(self):
        """测试无效的输入类型"""
        classifier = ResultClassifier()

        with pytest.raises(ClassificationException) as exc_info:
            classifier.classify_batch("not_a_list")

        assert "Expected list" in str(exc_info.value)

    def test_classify_batch_with_invalid_result(self):
        """测试列表中包含无效结果"""
        classifier = ResultClassifier()

        results = [
            TestResult(
                workflow_id="wf_001",
                execution_id="exec_001",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=1.0,
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={"answer": "test"},
            ),
            "invalid_result",  # 无效结果
        ]

        with pytest.raises(ClassificationException):
            classifier.classify_batch(results)

    def test_classify_batch_exception_in_classify_result(self):
        """测试批量分类中classify_result抛出非ClassificationException异常"""
        classifier = ResultClassifier()

        # 创建一个正常结果
        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "test"},
        )

        # Mock classify_result 抛出非 ClassificationException 异常
        def mock_error(r):
            raise RuntimeError("Unexpected error")

        original_method = classifier.classify_result
        classifier.classify_result = mock_error

        with pytest.raises(ClassificationException) as exc_info:
            classifier.classify_batch([result])

        assert "Failed to classify batch" in str(exc_info.value)

        # 恢复原方法
        classifier.classify_result = original_method


class TestSetThresholds:
    """测试阈值设置"""

    def test_set_valid_thresholds(self):
        """测试设置有效阈值"""
        classifier = ResultClassifier()

        new_thresholds = {
            "excellent": {"execution_time": 1.5, "token_efficiency": 0.85},
            "good": {"execution_time": 4.0, "token_efficiency": 0.65},
            "fair": {"execution_time": 9.0, "token_efficiency": 0.45},
        }

        classifier.set_thresholds(new_thresholds)

        assert classifier._thresholds == new_thresholds
        assert classifier._thresholds["excellent"]["execution_time"] == 1.5

    def test_set_thresholds_missing_grade(self):
        """测试缺少必需等级"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 0.9},
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            # 缺少 fair
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "Missing threshold for grade: fair" in str(exc_info.value)

    def test_set_thresholds_missing_field(self):
        """测试缺少必需字段"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"execution_time": 1.0},  # 缺少 token_efficiency
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "Missing token_efficiency for excellent" in str(exc_info.value)

    def test_set_thresholds_missing_execution_time(self):
        """测试缺少 execution_time 字段"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"token_efficiency": 0.9},  # 缺少 execution_time
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "Missing execution_time for excellent" in str(exc_info.value)

    def test_set_thresholds_invalid_type(self):
        """测试无效的阈值类型"""
        classifier = ResultClassifier()

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds("not_a_dict")

        assert "Thresholds must be a dictionary" in str(exc_info.value)

    def test_set_thresholds_negative_value(self):
        """测试负数阈值"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"execution_time": -1.0, "token_efficiency": 0.9},
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "must be positive" in str(exc_info.value)

    def test_set_thresholds_invalid_efficiency_range(self):
        """测试 token_efficiency 超出范围"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 1.5},  # > 1
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "must be in (0, 1]" in str(exc_info.value)

    def test_set_thresholds_wrong_order_execution_time(self):
        """测试 execution_time 顺序错误"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"execution_time": 5.0, "token_efficiency": 0.9},  # 应该最小
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "ascending order" in str(exc_info.value)

    def test_set_thresholds_wrong_order_efficiency(self):
        """测试 token_efficiency 顺序错误"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 0.5},  # 应该最大
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.9},
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "descending order" in str(exc_info.value)

    def test_set_thresholds_invalid_value_type(self):
        """测试值类型错误"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"execution_time": "1.0", "token_efficiency": 0.9},  # 字符串
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "must be a number" in str(exc_info.value)

    def test_set_thresholds_invalid_efficiency_type(self):
        """测试效率值类型错误"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": "0.9"},  # 字符串
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "must be a number" in str(exc_info.value)

    def test_set_thresholds_zero_efficiency(self):
        """测试效率值为0"""
        classifier = ResultClassifier()

        invalid_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 0.0},  # 0
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        with pytest.raises(ClassificationException) as exc_info:
            classifier.set_thresholds(invalid_thresholds)

        assert "must be in (0, 1]" in str(exc_info.value)


class TestGetThresholds:
    """测试获取阈值配置"""

    def test_get_thresholds_default(self):
        """测试获取默认阈值"""
        classifier = ResultClassifier()
        thresholds = classifier.get_thresholds()

        assert thresholds == ResultClassifier.DEFAULT_THRESHOLDS
        assert "excellent" in thresholds
        assert "good" in thresholds
        assert "fair" in thresholds

    def test_get_thresholds_custom(self):
        """测试获取自定义阈值"""
        custom_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 0.9},
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        classifier = ResultClassifier(thresholds=custom_thresholds)
        thresholds = classifier.get_thresholds()

        assert thresholds == custom_thresholds

    def test_get_thresholds_returns_copy(self):
        """测试返回的是副本，修改不影响原配置"""
        classifier = ResultClassifier()
        thresholds = classifier.get_thresholds()

        # 修改返回的字典
        thresholds["excellent"]["execution_time"] = 999.0

        # 原配置不应改变
        assert classifier._thresholds["excellent"]["execution_time"] == 2.0


class TestValidateThresholds:
    """测试阈值验证方法"""

    def test_validate_valid_thresholds(self):
        """测试验证有效阈值"""
        classifier = ResultClassifier()

        valid_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 0.9},
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        # 不应抛出异常
        classifier._validate_thresholds(valid_thresholds)

    def test_validate_with_int_values(self):
        """测试整数值也应该被接受"""
        classifier = ResultClassifier()

        valid_thresholds = {
            "excellent": {"execution_time": 1, "token_efficiency": 1.0},  # int
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }

        # 不应抛出异常
        classifier._validate_thresholds(valid_thresholds)


class TestIntegrationScenarios:
    """集成场景测试"""

    def test_complete_workflow_with_custom_thresholds(self):
        """测试完整工作流 - 自定义阈值"""
        # 1. 创建严格的分类器
        strict_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 0.9},
            "good": {"execution_time": 2.0, "token_efficiency": 0.8},
            "fair": {"execution_time": 4.0, "token_efficiency": 0.7},
        }
        classifier = ResultClassifier(thresholds=strict_thresholds)

        # 2. 创建测试结果
        results = [
            TestResult(
                workflow_id="wf_001",
                execution_id=f"exec_{i:03d}",
                timestamp=datetime.now(),
                status=TestStatus.SUCCESS,
                execution_time=0.5 + i * 0.5,
                tokens_used=100,
                cost=0.01,
                inputs={},
                outputs={"answer": "x" * (400 - i * 50)},
            )
            for i in range(6)
        ]

        # 3. 批量分类
        stats = classifier.classify_batch(results)

        # 4. 验证结果
        assert stats.excellent_count > 0
        assert stats.poor_count > 0

        # 5. 验证总数
        total = (
                stats.excellent_count
                + stats.good_count
                + stats.fair_count
                + stats.poor_count
        )
        assert total == len(results)

    def test_dynamic_threshold_adjustment(self):
        """测试动态调整阈值"""
        classifier = ResultClassifier()

        result = TestResult(
            workflow_id="wf_001",
            execution_id="exec_001",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS,
            execution_time=1.5,
            tokens_used=100,
            cost=0.01,
            inputs={},
            outputs={"answer": "x" * 350},  # 效率 0.875
        )

        # 使用默认阈值，应该是 EXCELLENT
        grade1 = classifier.classify_result(result)
        assert grade1 == PerformanceGrade.EXCELLENT

        # 调整到更严格的阈值
        strict_thresholds = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 0.95},
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.5},
        }
        classifier.set_thresholds(strict_thresholds)

        # 现在应该是 GOOD
        grade2 = classifier.classify_result(result)
        assert grade2 == PerformanceGrade.GOOD

    def test_acceptance_test_from_requirements(self):
        """验收测试 - 来自需求文档"""
        classifier = ResultClassifier()

        # 测试 1: 优秀结果
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
        assert grade == PerformanceGrade.EXCELLENT

        # 测试 2: 批量分类
        results = [excellent_result]
        stats = classifier.classify_batch(results)
        assert stats.excellent_count == 1
        assert stats.grade_distribution[PerformanceGrade.EXCELLENT] == 100.0
