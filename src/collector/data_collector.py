"""
Collector Module - Data Collector

Date: 2025-11-13
Author: backend-developer
Description: 负责测试结果的收集和统计分析
"""

# 标准库
from typing import List, Optional, Dict

# 第三方库
from loguru import logger as loguru_logger

# 项目内部
from ..utils.exceptions import DataValidationException
from .models import TestResult, TestStatus, PerformanceMetrics


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


def calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """
    计算百分位数（P50, P95, P99）

    使用线性插值法计算百分位数，这是统计学上的标准做法。

    Args:
        values: 数值列表

    Returns:
        包含 p50, p95, p99 的字典

    Algorithm:
        1. 对值列表排序
        2. 使用线性插值法计算百分位数
        3. 百分位计算公式: index = (p / 100.0) * (n - 1)
        4. 在 index 的整数部分和上界之间线性插值

    Edge Cases:
        - 空列表: 返回全 0
        - 单个值: 返回该值作为所有百分位
        - 两个值: 返回插值结果

    Example:
        >>> calculate_percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        {'p50': 3.0, 'p95': 4.8, 'p99': 4.96}
    """
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

    sorted_values = sorted(values)
    n = len(sorted_values)

    def percentile(p: float) -> float:
        """计算第 p 百分位 (0-100)"""
        if n == 1:
            return sorted_values[0]

        # 使用线性插值
        index = (p / 100.0) * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        weight = index - lower

        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    return {
        "p50": percentile(50),
        "p95": percentile(95),
        "p99": percentile(99),
    }


class DataCollector:
    """
    测试结果数据收集器

    负责收集测试执行结果，提供统计分析和数据查询功能。
    支持按工作流、变体、数据集等维度过滤数据。

    Thread Safety:
        在 CPython 环境下，``collect_result()`` 支持多线程并发写入
        （包括同一 workflow_id），不会丢失数据。但统计相关方法
        （``get_statistics()``, ``get_results_by_workflow()`` 等）
        返回的是近实时快照，不保证跨线程的强一致性。如需强一致视图，
        请在外层进行同步控制。

    Attributes:
        _results: 存储所有测试结果的列表
        _results_by_workflow: 按工作流ID索引的结果字典

    Example:
        >>> collector = DataCollector()
        >>> collector.collect_result(test_result)
        >>> metrics = collector.get_statistics()
        >>> print(f"Success rate: {metrics.success_rate:.2%}")
    """

    def __init__(self):
        """初始化数据收集器"""
        self._results: List[TestResult] = []
        self._results_by_workflow: Dict[str, List[TestResult]] = {}
        logger.info("DataCollector initialized")

    def collect_result(self, result: TestResult) -> None:
        """
        收集单个测试结果

        Args:
            result: TestResult 对象

        Raises:
            DataValidationException: 当结果数据无效时

        Example:
            >>> result = TestResult(...)
            >>> collector.collect_result(result)
        """
        # 1. 验证类型
        if not isinstance(result, TestResult):
            error_msg = f"Expected TestResult, got {type(result).__name__}"
            logger.error(f"Invalid result type: {type(result)}")
            raise DataValidationException(error_msg)

        # 2. 验证必需字段非空
        if not result.workflow_id:
            logger.error("workflow_id cannot be empty")
            raise DataValidationException("workflow_id is required")

        if not result.execution_id:
            logger.error("execution_id cannot be empty")
            raise DataValidationException("execution_id is required")

        # 3. 验证数值合法性
        if result.execution_time < 0:
            logger.error(f"Invalid execution_time: {result.execution_time}")
            raise DataValidationException("execution_time must be non-negative")

        if result.tokens_used < 0:
            logger.error(f"Invalid tokens_used: {result.tokens_used}")
            raise DataValidationException("tokens_used must be non-negative")

        if result.cost < 0:
            logger.error(f"Invalid cost: {result.cost}")
            raise DataValidationException("cost must be non-negative")

        # 4. 添加到结果列表
        self._results.append(result)

        # 5. 更新工作流索引（使用 setdefault 避免并发时列表覆盖）
        workflow_list = self._results_by_workflow.setdefault(result.workflow_id, [])
        workflow_list.append(result)

        # 6. 记录日志
        logger.debug(
            f"Collected result for workflow '{result.workflow_id}', "
            f"execution '{result.execution_id}', status: {result.status.value}"
        )

    def get_statistics(self, workflow_id: Optional[str] = None) -> PerformanceMetrics:
        """
        计算性能统计指标

        Args:
            workflow_id: 可选，仅计算指定工作流的统计

        Returns:
            PerformanceMetrics 对象

        Raises:
            DataValidationException: 当没有结果数据时

        Example:
            >>> metrics = collector.get_statistics()
            >>> metrics = collector.get_statistics(workflow_id="wf_001")
        """
        # 1. 获取要统计的结果集
        if workflow_id:
            results = self.get_results_by_workflow(workflow_id)
        else:
            results = self._results

        # 2. 检查是否有数据
        if not results:
            error_msg = "No results to calculate statistics" if not workflow_id else f"No results found for workflow '{workflow_id}'"
            logger.error(error_msg)
            raise DataValidationException(error_msg)

        # 3. 计算基础统计
        total_executions = len(results)
        successful_count = sum(1 for r in results if r.status == TestStatus.SUCCESS)
        failed_count = total_executions - successful_count
        success_rate = successful_count / total_executions if total_executions > 0 else 0.0

        # 4. 计算执行时间统计
        execution_times = [r.execution_time for r in results]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0

        # 计算百分位数
        percentiles = self._calculate_percentiles(execution_times)
        p50_execution_time = percentiles["p50"]
        p95_execution_time = percentiles["p95"]
        p99_execution_time = percentiles["p99"]

        # 5. 计算 Token 和成本统计
        total_tokens = sum(r.tokens_used for r in results)
        total_cost = sum(r.cost for r in results)
        avg_tokens_per_request = total_tokens / total_executions if total_executions > 0 else 0.0

        # 6. 记录日志
        logger.info(
            f"Calculated statistics: {total_executions} executions, "
            f"success rate: {success_rate:.2%}"
        )

        # 7. 返回 PerformanceMetrics 对象
        return PerformanceMetrics(
            total_executions=total_executions,
            successful_count=successful_count,
            failed_count=failed_count,
            success_rate=success_rate,
            avg_execution_time=avg_execution_time,
            p50_execution_time=p50_execution_time,
            p95_execution_time=p95_execution_time,
            p99_execution_time=p99_execution_time,
            total_tokens=total_tokens,
            total_cost=total_cost,
            avg_tokens_per_request=avg_tokens_per_request,
        )

    def _calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """
        计算百分位数（内部方法，建议使用模块级 calculate_percentiles 函数）

        Deprecated:
            This method wraps the module-level calculate_percentiles() function.
            For new code, consider using calculate_percentiles() directly.

        Args:
            values: 数值列表

        Returns:
            包含 p50, p95, p99 的字典
        """
        return calculate_percentiles(values)

    def get_all_results(self) -> List[TestResult]:
        """
        获取所有测试结果

        Returns:
            TestResult 列表
        """
        # 返回副本，避免外部修改
        return list(self._results)

    def get_results_by_workflow(self, workflow_id: str) -> List[TestResult]:
        """
        获取指定工作流的所有测试结果

        Args:
            workflow_id: 工作流ID

        Returns:
            TestResult 列表
        """
        # 返回副本，避免外部修改
        return list(self._results_by_workflow.get(workflow_id, []))

    def get_results_by_variant(self, workflow_id: str, variant_id: str) -> List[TestResult]:
        """
        获取指定工作流和变体的测试���果

        Args:
            workflow_id: 工作流ID
            variant_id: 提示词变体ID

        Returns:
            TestResult 列表
        """
        workflow_results = self.get_results_by_workflow(workflow_id)
        return [r for r in workflow_results if r.prompt_variant == variant_id]

    def get_results_by_dataset(self, dataset: str) -> List[TestResult]:
        """
        获取指定数据集的测试结果

        Args:
            dataset: 数据集名称

        Returns:
            TestResult 列表
        """
        return [r for r in self._results if r.dataset == dataset]

    def get_result_count(self) -> int:
        """
        获取已收集的结果总数

        Returns:
            结果数量
        """
        return len(self._results)

    def clear(self) -> None:
        """
        清空所有已收集的数据

        记录警告日志并清空内部存储
        """
        count = len(self._results)
        logger.warning(f"Clearing all collected data ({count} results)")
        self._results.clear()
        self._results_by_workflow.clear()
