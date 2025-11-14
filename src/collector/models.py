"""
结果采集模块 - 数据模型定义

Date: 2025-11-13
Author: backend-developer
Description: 定义测试结果、性能指标和分类统计的核心数据结构
"""

# 标准库
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class TestStatus(Enum):
    """测试执行状态"""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    ERROR = "error"


class PerformanceGrade(Enum):
    """性能分级"""

    EXCELLENT = "excellent"  # 优秀: 执行时间<2s, Token效率>0.8
    GOOD = "good"  # 良好: 执行时间<5s, Token效率>0.6
    FAIR = "fair"  # 一般: 执行时间<10s, Token效率>0.4
    POOR = "poor"  # 较差: 其他情况


@dataclass
class TestResult:
    """
    单次测试执行结果

    Attributes:
        workflow_id: 工作流唯一标识
        execution_id: 本次执行的唯一ID
        timestamp: 执行时间戳
        status: 执行状态 (SUCCESS/FAILED/TIMEOUT/ERROR)
        execution_time: 执行耗时(秒)
        tokens_used: 消耗的Token数量
        cost: 本次执行成本(美元)
        inputs: 输入参数字典
        outputs: 输出结果字典
        error_message: 错误信息(失败时填写)
        prompt_variant: 提示词变体ID(可选)
        dataset: 数据集名称(可选)
        metadata: 额外元数据
    """

    workflow_id: str
    execution_id: str
    timestamp: datetime
    status: TestStatus
    execution_time: float
    tokens_used: int
    cost: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    error_message: Optional[str] = None
    prompt_variant: Optional[str] = None
    dataset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """
    性能统计指标

    Attributes:
        total_executions: 总执行次数
        successful_count: 成功次数
        failed_count: 失败次数
        success_rate: 成功率 (0-1)
        avg_execution_time: 平均执行时间(秒)
        p50_execution_time: 50分位执行时间
        p95_execution_time: 95分位执行时间
        p99_execution_time: 99分位执行时间
        total_tokens: 总Token消耗
        total_cost: 总成本(美元)
        avg_tokens_per_request: 平均每次请求Token数
    """

    total_executions: int
    successful_count: int
    failed_count: int
    success_rate: float
    avg_execution_time: float
    p50_execution_time: float
    p95_execution_time: float
    p99_execution_time: float
    total_tokens: int
    total_cost: float
    avg_tokens_per_request: float


@dataclass
class ClassificationResult:
    """
    性能分类统计结果

    Attributes:
        excellent_count: 优秀等级数量
        good_count: 良好等级数量
        fair_count: 一般等级数量
        poor_count: 较差等级数量
        grade_distribution: 各等级占比(百分比, 0-100)
    """

    excellent_count: int
    good_count: int
    fair_count: int
    poor_count: int
    grade_distribution: Dict[PerformanceGrade, float]
