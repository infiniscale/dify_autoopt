"""
Collector Module - Result Classifier

Date: 2025-11-13
Author: backend-developer
Description: 负责测试结果的性能分级和批量统计分析
"""

# 标准库
import copy
from typing import Dict, List, Optional

# 第三方库
from loguru import logger as loguru_logger

# 项目内部
from ..utils.exceptions import ClassificationException
from .models import ClassificationResult, PerformanceGrade, TestResult


# 延迟初始化 logger，避免未初始化异常
def _get_safe_logger():
    """获取安全的 logger 实例"""
    try:
        from ..utils.logger import get_logger

        return get_logger(__name__)
    except Exception:
        # 如果 logger 未初始化，使用 loguru 默认 logger
        return loguru_logger.bind(name=__name__)


logger = _get_safe_logger()


class ResultClassifier:
    """
    测试结果性能分类器

    根据执行时间和Token效率对测试结果进行性能分级。
    支持自定义阈值配置。

    分级标准 (默认):
    - EXCELLENT: execution_time < 2s AND token_efficiency > 0.8
    - GOOD: execution_time < 5s AND token_efficiency > 0.6
    - FAIR: execution_time < 10s AND token_efficiency > 0.4
    - POOR: 其他情况

    Token效率 = (有效输出长度 / tokens_used)，范围 0-1
    假设理想比例是 1:4 (每个token产生4字符)

    Attributes:
        _thresholds: 分级阈值配置字典

    Example:
        >>> classifier = ResultClassifier()
        >>> grade = classifier.classify_result(test_result)
        >>> print(f"Performance grade: {grade.value}")

        >>> # 自定义阈值
        >>> custom_thresholds = {
        ...     "excellent": {"execution_time": 1.0, "token_efficiency": 0.9},
        ...     "good": {"execution_time": 3.0, "token_efficiency": 0.7},
        ...     "fair": {"execution_time": 8.0, "token_efficiency": 0.5}
        ... }
        >>> classifier = ResultClassifier(thresholds=custom_thresholds)
    """

    DEFAULT_THRESHOLDS = {
        "excellent": {"execution_time": 2.0, "token_efficiency": 0.8},
        "good": {"execution_time": 5.0, "token_efficiency": 0.6},
        "fair": {"execution_time": 10.0, "token_efficiency": 0.4},
    }

    def __init__(self, thresholds: Optional[Dict[str, Dict[str, float]]] = None):
        """
        初始化分类器

        Args:
            thresholds: 可选，自定义阈值配置
                格式: {
                    "excellent": {"execution_time": 2.0, "token_efficiency": 0.8},
                    "good": {"execution_time": 5.0, "token_efficiency": 0.6},
                    "fair": {"execution_time": 10.0, "token_efficiency": 0.4}
                }

        Raises:
            ClassificationException: 阈值配置无效时

        Example:
            >>> classifier = ResultClassifier()
            >>> # 或使用自定义阈值
            >>> classifier = ResultClassifier(thresholds=custom_config)
        """
        if thresholds is None:
            self._thresholds = self.DEFAULT_THRESHOLDS.copy()
            logger.info("ResultClassifier initialized with default thresholds")
        else:
            self._validate_thresholds(thresholds)
            self._thresholds = thresholds
            logger.info(f"ResultClassifier initialized with custom thresholds: {thresholds}")

    def classify_result(self, result: TestResult) -> PerformanceGrade:
        """
        对单个测试结果进行性能分级

        分级采用级联检查，从高到低依次匹配:
        1. EXCELLENT: 执行时间和效率同时满足最高标准
        2. GOOD: 执行时间和效率同时满足良好标准
        3. FAIR: 执行时间和效率同时满足一般标准
        4. POOR: 其他情况

        Args:
            result: 测试结果对象

        Returns:
            PerformanceGrade 枚举值

        Raises:
            ClassificationException: 分类失败时

        Example:
            >>> result = TestResult(...)
            >>> grade = classifier.classify_result(result)
            >>> assert grade == PerformanceGrade.EXCELLENT
        """
        try:
            # 1. 验证输入类型
            if not isinstance(result, TestResult):
                error_msg = f"Expected TestResult, got {type(result).__name__}"
                logger.error(f"Invalid result type: {error_msg}")
                raise ClassificationException(error_msg)

            # 2. 计算性能指标
            exec_time = result.execution_time
            efficiency = self._calculate_token_efficiency(result)

            # 3. 级联检查分级
            if (
                    exec_time < self._thresholds["excellent"]["execution_time"]
                    and efficiency >= self._thresholds["excellent"]["token_efficiency"]
            ):
                grade = PerformanceGrade.EXCELLENT

            elif (
                    exec_time < self._thresholds["good"]["execution_time"]
                    and efficiency >= self._thresholds["good"]["token_efficiency"]
            ):
                grade = PerformanceGrade.GOOD

            elif (
                    exec_time < self._thresholds["fair"]["execution_time"]
                    and efficiency >= self._thresholds["fair"]["token_efficiency"]
            ):
                grade = PerformanceGrade.FAIR

            else:
                grade = PerformanceGrade.POOR

            # 4. 记录日志
            logger.debug(
                f"Classified result '{result.execution_id}': "
                f"grade={grade.value}, exec_time={exec_time:.2f}s, "
                f"efficiency={efficiency:.2f}"
            )

            return grade

        except ClassificationException:
            raise
        except Exception as e:
            error_msg = f"Failed to classify result: {str(e)}"
            logger.error(error_msg)
            raise ClassificationException(error_msg) from e

    def classify_batch(self, results: List[TestResult]) -> ClassificationResult:
        """
        批量分类���统计

        对测试结果列表进行批量分级，并计算各等级的数量和占比。

        Args:
            results: 测试结果列表

        Returns:
            ClassificationResult 统计对象

        Raises:
            ClassificationException: 批量分类失败时

        Example:
            >>> results = [result1, result2, result3]
            >>> stats = classifier.classify_batch(results)
            >>> print(f"Excellent: {stats.excellent_count}")
            >>> print(f"Good rate: {stats.grade_distribution[PerformanceGrade.GOOD]:.1f}%")
        """
        try:
            # 1. 验证输入
            if not isinstance(results, list):
                error_msg = f"Expected list, got {type(results).__name__}"
                logger.error(f"Invalid results type: {error_msg}")
                raise ClassificationException(error_msg)

            # 2. 初始化计数器
            counts = {grade: 0 for grade in PerformanceGrade}

            # 3. 逐个分类并统计
            for result in results:
                grade = self.classify_result(result)
                counts[grade] += 1

            # 4. 计算分布比例 (百分比, 0-100)
            total = len(results)
            distribution = {
                grade: (count / total * 100.0) if total > 0 else 0.0
                for grade, count in counts.items()
            }

            # 5. 记录日志
            logger.info(
                f"Classified {total} results: "
                f"EXCELLENT={counts[PerformanceGrade.EXCELLENT]}, "
                f"GOOD={counts[PerformanceGrade.GOOD]}, "
                f"FAIR={counts[PerformanceGrade.FAIR]}, "
                f"POOR={counts[PerformanceGrade.POOR]}"
            )

            # 6. 返回统计结果
            return ClassificationResult(
                excellent_count=counts[PerformanceGrade.EXCELLENT],
                good_count=counts[PerformanceGrade.GOOD],
                fair_count=counts[PerformanceGrade.FAIR],
                poor_count=counts[PerformanceGrade.POOR],
                grade_distribution=distribution,
            )

        except ClassificationException:
            raise
        except Exception as e:
            error_msg = f"Failed to classify batch: {str(e)}"
            logger.error(error_msg)
            raise ClassificationException(error_msg) from e

    def set_thresholds(self, thresholds: Dict[str, Dict[str, float]]) -> None:
        """
        动态调整分类阈值

        允许运行时修改分级标准，便于实验和优化。

        Args:
            thresholds: 阈值配置字典
                格式: {
                    "excellent": {"execution_time": float, "token_efficiency": float},
                    "good": {"execution_time": float, "token_efficiency": float},
                    "fair": {"execution_time": float, "token_efficiency": float}
                }

        Raises:
            ClassificationException: 阈值配置无效时

        Example:
            >>> new_thresholds = {
            ...     "excellent": {"execution_time": 1.5, "token_efficiency": 0.85},
            ...     "good": {"execution_time": 4.0, "token_efficiency": 0.65},
            ...     "fair": {"execution_time": 9.0, "token_efficiency": 0.45}
            ... }
            >>> classifier.set_thresholds(new_thresholds)
        """
        self._validate_thresholds(thresholds)
        self._thresholds = thresholds
        logger.info(f"Thresholds updated: {thresholds}")

    def get_thresholds(self) -> Dict[str, Dict[str, float]]:
        """
        获取当前阈值配置

        Returns:
            当前使用的阈值配置字典 (深拷贝)

        Example:
            >>> current = classifier.get_thresholds()
            >>> print(current["excellent"]["execution_time"])
        """
        return copy.deepcopy(self._thresholds)

    def _calculate_token_efficiency(self, result: TestResult) -> float:
        """
        计算Token效率

        算法说明:
        1. 如果 tokens_used 为 0，效率为 0
        2. 计算输出内容字符长度 (将输出字典转为字符串)
        3. 效率 = 输出长度 / (tokens * 4.0)
        4. 归一化到 0-1 范围 (假设理想比例是 1:4，即每个token产生4字符)
        5. 上限设为 1.0

        Args:
            result: 测试结果对象

        Returns:
            Token效率值 (0.0 - 1.0)

        Example:
            >>> result = TestResult(..., tokens_used=100, outputs={"answer": "x" * 400})
            >>> efficiency = classifier._calculate_token_efficiency(result)
            >>> assert efficiency == 1.0  # 400 / (100 * 4) = 1.0
        """
        # 1. tokens 为 0 时，效率为 0
        if result.tokens_used == 0:
            return 0.0

        # 2. 计算输出内容长度
        output_str = str(result.outputs)
        output_length = len(output_str)

        # 3. 计算效率 (假设理想比例是 1:4)
        # 每个 token 理想情况下产生 4 个字符
        ideal_chars_per_token = 4.0
        efficiency = output_length / (result.tokens_used * ideal_chars_per_token)

        # 4. 归一化到 0-1 范围
        return min(efficiency, 1.0)

    def _validate_thresholds(self, thresholds: Dict[str, Dict[str, float]]) -> None:
        """
        验证阈值配置的合法性

        检查项:
        1. 必须包含 excellent, good, fair 三个等级
        2. 每个等级必须包含 execution_time 和 token_efficiency
        3. 所有值必须为正数
        4. 阈值应该递减 (excellent < good < fair)

        Args:
            thresholds: 待验证的阈值配置

        Raises:
            ClassificationException: 配置无效时
        """
        # 1. 验证类型
        if not isinstance(thresholds, dict):
            raise ClassificationException("Thresholds must be a dictionary")

        # 2. 验证必需等级
        required_grades = ["excellent", "good", "fair"]
        for grade in required_grades:
            if grade not in thresholds:
                raise ClassificationException(f"Missing threshold for grade: {grade}")

            # 3. 验证必需字段
            if "execution_time" not in thresholds[grade]:
                raise ClassificationException(f"Missing execution_time for {grade}")

            if "token_efficiency" not in thresholds[grade]:
                raise ClassificationException(f"Missing token_efficiency for {grade}")

            # 4. 验证值的类型和范围
            exec_time = thresholds[grade]["execution_time"]
            efficiency = thresholds[grade]["token_efficiency"]

            if not isinstance(exec_time, (int, float)):
                raise ClassificationException(
                    f"execution_time for {grade} must be a number, got {type(exec_time).__name__}"
                )

            if not isinstance(efficiency, (int, float)):
                raise ClassificationException(
                    f"token_efficiency for {grade} must be a number, got {type(efficiency).__name__}"
                )

            if exec_time <= 0:
                raise ClassificationException(
                    f"execution_time for {grade} must be positive, got {exec_time}"
                )

            if not (0 < efficiency <= 1):
                raise ClassificationException(
                    f"token_efficiency for {grade} must be in (0, 1], got {efficiency}"
                )

        # 5. 验证阈值递增顺序 (excellent < good < fair)
        if not (
                thresholds["excellent"]["execution_time"]
                < thresholds["good"]["execution_time"]
                < thresholds["fair"]["execution_time"]
        ):
            raise ClassificationException(
                "execution_time thresholds must be in ascending order: excellent < good < fair"
            )

        if not (
                thresholds["excellent"]["token_efficiency"]
                > thresholds["good"]["token_efficiency"]
                > thresholds["fair"]["token_efficiency"]
        ):
            raise ClassificationException(
                "token_efficiency thresholds must be in descending order: excellent > good > fair"
            )
