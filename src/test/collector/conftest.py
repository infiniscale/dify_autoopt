"""
共享测试 Fixtures for Collector 集成测试

Date: 2025-11-14
Author: qa-engineer
Description: 提供可复用的测试数据和辅助函数
"""

import pytest
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from src.collector import TestResult, TestStatus


@pytest.fixture
def sample_results() -> List[TestResult]:
    """
    生成 20 条示例结果

    包含不同的工作流、状态和性能特征
    """
    results = []
    base_time = datetime.now()

    for i in range(20):
        # 生成不同状态：75% 成功，15% 失败，5% 超时，5% 错误
        if i % 20 < 15:
            status = TestStatus.SUCCESS
            error_msg = None
        elif i % 20 < 18:
            status = TestStatus.FAILED
            error_msg = f"Validation error at step {i}"
        elif i % 20 < 19:
            status = TestStatus.TIMEOUT
            error_msg = "Request timeout after 30s"
        else:
            status = TestStatus.ERROR
            error_msg = "Internal server error"

        result = TestResult(
            workflow_id=f"wf_{i % 3}",  # 3个不同的工作流
            execution_id=f"exec_{i:03d}",
            timestamp=base_time - timedelta(seconds=i * 10),
            status=status,
            execution_time=0.5 + i * 0.2,  # 0.5s 到 4.3s
            tokens_used=100 + i * 20,  # 100 到 480
            cost=0.01 + i * 0.002,  # $0.01 到 $0.048
            inputs={"query": f"test_query_{i}", "context": f"context_{i}"},
            outputs={"answer": f"result_{i}" * 50} if status == TestStatus.SUCCESS else {},
            error_message=error_msg,
            prompt_variant=f"variant_{i % 2}" if i % 2 == 0 else None,
            dataset=f"dataset_{i % 3}" if i % 3 == 0 else None
        )
        results.append(result)

    return results


@pytest.fixture
def large_dataset() -> List[TestResult]:
    """
    生成 5,000 条结果用于性能测试

    模拟真实场景的数据分布
    """
    results = []
    base_time = datetime.now()
    workflows = [f"wf_{i}" for i in range(10)]

    for i in range(5000):
        # 状态分布：80% 成功，15% 失败，3% 超时，2% 错误
        rand_val = random.random()
        if rand_val < 0.80:
            status = TestStatus.SUCCESS
            error_msg = None
        elif rand_val < 0.95:
            status = TestStatus.FAILED
            error_msg = f"Validation failed: reason_{random.randint(1, 5)}"
        elif rand_val < 0.98:
            status = TestStatus.TIMEOUT
            error_msg = "Timeout"
        else:
            status = TestStatus.ERROR
            error_msg = "System error"

        result = TestResult(
            workflow_id=random.choice(workflows),
            execution_id=f"exec_{i:05d}",
            timestamp=base_time - timedelta(seconds=i),
            status=status,
            execution_time=random.uniform(0.1, 10.0),
            tokens_used=random.randint(50, 500),
            cost=random.uniform(0.005, 0.05),
            inputs={"query": f"query_{i}"},
            outputs={"answer": "x" * random.randint(100, 1000)} if status == TestStatus.SUCCESS else {},
            error_message=error_msg,
            prompt_variant=f"v{random.randint(1, 3)}" if random.random() < 0.5 else None,
            dataset=f"ds_{random.randint(1, 5)}" if random.random() < 0.3 else None
        )
        results.append(result)

    return results


@pytest.fixture
def mixed_workflow_results() -> List[TestResult]:
    """
    生成多工作流混合数据，用于多工作流集成测试

    - workflow_1: 50条，80%成功率，快速执行
    - workflow_2: 30条，90%成功率，中等执行时间
    - workflow_3: 20条，60%成功率，慢速执行
    """
    results = []
    base_time = datetime.now()

    # Workflow 1: 50条，80%成功率
    for i in range(50):
        status = TestStatus.SUCCESS if i % 5 != 0 else TestStatus.FAILED
        results.append(TestResult(
            workflow_id="workflow_1",
            execution_id=f"wf1_exec_{i:03d}",
            timestamp=base_time - timedelta(seconds=i),
            status=status,
            execution_time=random.uniform(0.5, 2.0),  # 快速
            tokens_used=random.randint(80, 150),
            cost=random.uniform(0.008, 0.015),
            inputs={"query": f"wf1_query_{i}"},
            outputs={"answer": f"wf1_result_{i}" * 30} if status == TestStatus.SUCCESS else {},
            error_message="Workflow 1 error" if status == TestStatus.FAILED else None
        ))

    # Workflow 2: 30条，90%成功率
    for i in range(30):
        status = TestStatus.SUCCESS if i % 10 != 0 else TestStatus.FAILED
        results.append(TestResult(
            workflow_id="workflow_2",
            execution_id=f"wf2_exec_{i:03d}",
            timestamp=base_time - timedelta(seconds=i),
            status=status,
            execution_time=random.uniform(2.0, 5.0),  # 中等
            tokens_used=random.randint(150, 300),
            cost=random.uniform(0.015, 0.030),
            inputs={"query": f"wf2_query_{i}"},
            outputs={"answer": f"wf2_result_{i}" * 40} if status == TestStatus.SUCCESS else {},
            error_message="Workflow 2 error" if status == TestStatus.FAILED else None
        ))

    # Workflow 3: 20条，60%成功率
    for i in range(20):
        status = TestStatus.SUCCESS if i % 5 < 3 else TestStatus.FAILED
        results.append(TestResult(
            workflow_id="workflow_3",
            execution_id=f"wf3_exec_{i:03d}",
            timestamp=base_time - timedelta(seconds=i),
            status=status,
            execution_time=random.uniform(5.0, 15.0),  # 慢速
            tokens_used=random.randint(300, 600),
            cost=random.uniform(0.030, 0.060),
            inputs={"query": f"wf3_query_{i}"},
            outputs={"answer": f"wf3_result_{i}" * 50} if status == TestStatus.SUCCESS else {},
            error_message="Workflow 3 error" if status == TestStatus.FAILED else None
        ))

    return results


@pytest.fixture
def edge_case_results() -> List[TestResult]:
    """
    生成边界和异常情况的测试数据

    包含：
    - 极端执行时间（极快和极慢）
    - 零 Token 消耗
    - 空输出
    - 特殊字符
    """
    base_time = datetime.now()

    return [
        # 极快执行
        TestResult(
            workflow_id="edge_fast",
            execution_id="exec_fast_001",
            timestamp=base_time,
            status=TestStatus.SUCCESS,
            execution_time=0.01,
            tokens_used=10,
            cost=0.001,
            inputs={"query": "fast"},
            outputs={"answer": "ok"}
        ),
        # 极慢执行
        TestResult(
            workflow_id="edge_slow",
            execution_id="exec_slow_001",
            timestamp=base_time,
            status=TestStatus.SUCCESS,
            execution_time=100.0,
            tokens_used=1000,
            cost=0.1,
            inputs={"query": "slow"},
            outputs={"answer": "x" * 10000}
        ),
        # 零 Token
        TestResult(
            workflow_id="edge_zero_token",
            execution_id="exec_zero_001",
            timestamp=base_time,
            status=TestStatus.SUCCESS,
            execution_time=1.0,
            tokens_used=0,
            cost=0.0,
            inputs={"query": "zero"},
            outputs={"answer": ""}
        ),
        # 失败但有输出
        TestResult(
            workflow_id="edge_failed_output",
            execution_id="exec_fail_001",
            timestamp=base_time,
            status=TestStatus.FAILED,
            execution_time=2.0,
            tokens_used=100,
            cost=0.01,
            inputs={"query": "fail"},
            outputs={"partial": "data"},
            error_message="Validation failed after processing"
        ),
        # 特殊字符
        TestResult(
            workflow_id="edge_special_chars",
            execution_id="exec_special_001",
            timestamp=base_time,
            status=TestStatus.SUCCESS,
            execution_time=1.5,
            tokens_used=150,
            cost=0.015,
            inputs={"query": "中文\n换行\t制表符"},
            outputs={"answer": "🎉 emoji, quotes: \"test\", backslash: \\"}
        )
    ]


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """
    创建临时输出目录

    用于测试文件导出功能
    """
    output_dir = tmp_path / "test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def sample_config_yaml(tmp_path: Path) -> Path:
    """
    创建示例配置 YAML 文件

    用于测试配置集成
    """
    config_content = """
classification:
  thresholds:
    excellent:
      execution_time: 2.0
      token_efficiency: 0.8
    good:
      execution_time: 5.0
      token_efficiency: 0.6
    fair:
      execution_time: 10.0
      token_efficiency: 0.4

export:
  default_filename: "test_results.xlsx"
  include_charts: true
  max_detail_rows: 10000
"""

    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file


def create_test_result(
    workflow_id: str = "test_wf",
    execution_id: str = "test_exec",
    status: TestStatus = TestStatus.SUCCESS,
    execution_time: float = 1.0,
    tokens_used: int = 100,
    cost: float = 0.01
) -> TestResult:
    """
    辅助函数：快速创建测试结果

    Args:
        workflow_id: 工作流ID
        execution_id: 执行ID
        status: 执行状态
        execution_time: 执行时间
        tokens_used: Token使用量
        cost: 成本

    Returns:
        TestResult: 测试结果对象
    """
    return TestResult(
        workflow_id=workflow_id,
        execution_id=execution_id,
        timestamp=datetime.now(),
        status=status,
        execution_time=execution_time,
        tokens_used=tokens_used,
        cost=cost,
        inputs={"query": "test"},
        outputs={"answer": "result" * 20} if status == TestStatus.SUCCESS else {},
        error_message=f"{status.value} error" if status != TestStatus.SUCCESS else None
    )
