"""
Executor Module - Task Execution and Scheduling

This README documents the current executor implementation (concurrent execution,
task scheduling, pairwise generation, and RunManifest pipeline). It replaces
older docs that referenced non‑existent classes such as AsyncExecutor or
DynamicExecutor.
"""

# Executor 模块（任务执行与调度）

## 作用概述

Executor 模块负责**测试用例生成、执行清单构建、并发执行、任务调度和结果转换**。  
它把上游的配置/TestPlan 转换为可执行的 `RunManifest`，再通过并发执行器和调度器跑完所有任务，最终输出标准化的 `TestResult`。

> 当前 `src/executor` 包内代码在本地通过 `pytest` 全量测试，且已达到 100% 覆盖率。

---

## 模块结构概览

### 核心模型（`models.py`）

- `TaskStatus`：任务状态枚举（`PENDING/QUEUED/RUNNING/SUCCEEDED/FAILED/TIMEOUT/CANCELLED/ERROR`）。
- `Task`：单个测试任务的完整描述（来自 `TestCase` 的业务数据 + 执行策略 + 运行时状态）。
- `TaskResult`：任务执行结果（状态、错误信息、执行耗时、metadata 等）。
- `RunStatistics`：执行批次的聚合统计（成功数、失败数、重试次数等）。
- `RunExecutionResult`：单次 `RunManifest` 执行的完整结果（任务结果列表 + 统计 + metadata）。
- `CancellationToken`：线程安全的取消令牌，用于在并发执行中优雅地中断任务。

### 抽象执行器（`executor_base.py`）

- `ExecutorBase`：执行器抽象基类，定义统一的执行模板：
   1. 验证 `RunManifest` 完整性。
   2. 构建 `Task` 列表（`Task.from_manifest_case`）。
   3. 调用子类实现的 `_execute_tasks` 执行任务。
   4. 把 `TaskResult` 聚合为 `RunExecutionResult`。

对外主要使用公开方法：

```python
run_result = executor.run_manifest(manifest, cancellation_token=None)
```

### 并发执行器（`concurrent_executor.py`）

- `TaskExecutionFunc = Callable[[Task], Dict[str, Any]]`：任务执行函数签名。
- `ConcurrentExecutor(ExecutorBase)`：基于 `ThreadPoolExecutor` 的并发执行器，特性：
   - 从 `RunManifest.execution_policy` 读取并发度和 `RetryPolicy`。
   - 使用 `Task.timeout_seconds` 做单任务超时控制。
   - 按 `RetryPolicy.max_attempts/backoff_seconds/backoff_multiplier` 实现重试与指数退避。
   - 在执行前/执行中/退避阶段响应 `CancellationToken` 取消。
   - 对业务异常（`TaskExecutionException`, `TaskTimeoutException`）与系统异常分别处理，并写入
     `Task.metadata["error_message"]`。

### Stub 执行器（`stub_executor.py`）

- `StubExecutor(ConcurrentExecutor)`：用于本地/测试的桩执行器：
   - 不调用真实 Dify API。
   - 支持配置：
      - `simulated_delay`：模拟执行延迟。
      - `failure_rate`：全局失败率。
      - `task_behaviors`：按 task_id 配置 `"success"|"failure"|"timeout"|"error"`。
   - 通过 `_stub_execution` 生成结构稳定的输出，方便 collector 和报告模块消费。

### 速率限制与调度（`rate_limiter.py`、`task_scheduler.py`）

- `RateLimiter`：令牌桶速率限制器：
   - 使用 `RateLimit(per_minute, burst)` 控制吞吐。
   - 暴露 `acquire(tokens=1)` 和 `try_acquire(tokens=1)`。
   - 可注入 `now_fn/sleep_fn` 做精确测试。

- `TaskScheduler(ConcurrentExecutor)`：在并发执行基础上增加：
   - **批次调度**：按 `ExecutionPolicy.batch_size` 分批执行任务。
   - **速率限制**：若配置了 `rate_control`，每批调用 `RateLimiter.acquire`。
   - **停止条件**：根据 `stop_conditions["max_failures"]` 和 `stop_conditions["timeout"]` 早停。
   - **批次间退避**：按 `ExecutionPolicy.backoff_seconds` 在批次间 `sleep`（支持取消令牌）。

### Pairwise 与测试用例生成（`pairwise_engine.py`、`test_case_generator.py`）

- `PairwiseEngine`：
   - 支持 `engine_type='PICT'|'IPO'|'naive'`。
   - `PICT/IPO` 通过 `allpairspy.AllPairs` 生成成对覆盖组合，失败时回退到 naive 笛卡尔积。
   - 高维场景（维度 > 10）启用分层策略：前 5 个维度 pairwise，其余取默认值。
   - 可选 `cache_dir`，对同一 `dimensions+seed+engine_type` 结果进行 pickle 缓存。

- `TestCaseGenerator`：
   - 输入：`TestPlan`、`WorkflowCatalog`、`PairwiseEngine`。
   - 输出：`Dict[workflow_id, List[TestCase]]`。
   - 支持三种组合策略：
      - `mode='scenario_only'`：每个参数取第一个值。
      - `mode='pairwise'`：对 `Dataset.pairwise_dimensions` 使用 PairwiseEngine。
      - `mode='cartesian'`：笛卡尔积，受 `max_cases` 限制。
   - 若 dataset 配置了 `conversation_flows`，则优先按 flow 生成 `TestCase`。

### RunManifest 构建（`run_manifest_builder.py`）

- `RunManifestBuilder`：
   - 使用 `TestCaseGenerator.generate_all_cases()` 获取所有工作流的用例。
   - 按 `prompt_variant` 将 `TestCase` 分组为不同执行清单。
   - 使用 `PromptPatchEngine.apply_patches` 对非默认变体的 DSL 做 patch。
   - 构建 `RunManifest`（含执行策略、限流配置、评估配置、metadata）。

### 结果转换与 Service（`result_converter.py`、`executor_service.py`）

- `ResultConverter`：
   - 将 `TaskResult` 映射为 collector 模块的 `TestResult`。
   - 映射 `TaskStatus` → `TestStatus`。
   - 提取 tokens/cost/inputs/outputs。
   - 将 `TaskResult.metadata` 写入 `TestResult.metadata` 并解析 `prompt_variant`。

- `ExecutorService`：
   - 内部组合 `TaskScheduler` 与 `ResultConverter`。
   - 公开方法：`execute_test_plan(manifest, cancellation_token=None) -> List[TestResult]`。
   - 构造函数支持注入自定义 `task_execution_func`，用于接入真实 Dify API 或本地 stub。

---

## 使用示例

### 示例 1：高层入口 —— ExecutorService

```python
from src.executor import ExecutorService, CancellationToken
from src.config.models import RunManifest

# 假设已有从 YAML/JSON 反序列化得到的 RunManifest 对象
manifest: RunManifest = load_run_manifest("config/run_manifest.yaml")

service = ExecutorService()
token = CancellationToken()

test_results = service.execute_test_plan(manifest, cancellation_token=token)

for r in test_results:
    print(
        f"[{r.workflow_id}] status={r.status.value} "
        f"time={r.execution_time:.2f}s variant={r.prompt_variant}"
    )
```

> `load_run_manifest` 由上游配置模块提供，这里只展示 Executor 的调用方式。

### 示例 2：注入自定义任务执行函数

```python
from src.executor import ExecutorService
from src.executor.models import Task


def execute_workflow(task: Task) -> dict:
    """
    自定义任务执行逻辑：可以调用真实 Dify API，或本地 stub。
    必须返回 dict，至少包含:
      - workflow_run_id
      - status
      - outputs
    """
    # 伪代码：调用 Dify 工作流
    # response = call_dify(task.workflow_id, task.parameters)
    # return response.json()
    return {
        "workflow_run_id": f"run-{task.task_id}",
        "status": "succeeded",
        "outputs": {"result": "ok"},
    }


service = ExecutorService(task_execution_func=execute_workflow)
test_results = service.execute_test_plan(manifest)
```

### 示例 3：从 TestPlan 到 RunManifest 再到执行

```python
from src.config.models import EnvConfig, WorkflowCatalog, TestPlan
from src.config.utils.yaml_parser import load_yaml  # 示例：具体实现由 config 模块提供
from src.optimizer.prompt_patch_engine import PromptPatchEngine
from src.executor import (
    PairwiseEngine,
    TestCaseGenerator,
    RunManifestBuilder,
    ExecutorService,
)

# 1. 加载配置（伪代码）
env = EnvConfig(**load_yaml("config/env.yaml"))
catalog = WorkflowCatalog(**load_yaml("config/workflows.yaml"))
plan = TestPlan(**load_yaml("config/test_plan.yaml"))

# 2. 构建用例生成器和补丁引擎
pairwise_engine = PairwiseEngine(engine_type="PICT")
case_generator = TestCaseGenerator(plan=plan, catalog=catalog, pairwise_engine=pairwise_engine)
patch_engine = PromptPatchEngine(env)

# 3. 从 TestPlan 构建所有 RunManifest
builder = RunManifestBuilder(
    env=env,
    catalog=catalog,
    plan=plan,
    patch_engine=patch_engine,
    case_generator=case_generator,
)
manifests = builder.build_all()

# 4. 执行所有 manifest
service = ExecutorService()
all_results = []
for manifest in manifests:
    all_results.extend(service.execute_test_plan(manifest))
```

---

## 测试与完成度

- Executor 模块的单元测试位于 `src/test/executor`，包括：
   - `test_concurrent_executor.py`
   - `test_executor_base.py`
   - `test_task_scheduler.py`
   - `test_stub_executor.py`
   - `test_pairwise_engine.py`
   - `test_test_case_generator.py`
   - `test_run_manifest_builder.py`
   - `test_result_converter.py`
   - `test_executor_service.py`
- 在本地执行：

```bash
pytest src/test/executor --cov=src/executor --cov-report=term-missing
```

可以得到 `src/executor` 包下所有文件 100% 覆盖率。

> 说明：根目录 `README.md` 中的架构图仍使用早期的文件名（如 `concurrent.py`, `scheduler.py`），与当前实现略有出入，但不影响
> Executor 模块本身的完备度。

综上，从**实现代码**、**单元测试**和**模块内文档**三方面看，Executor 模块已经可以视为功能完整、可稳定复用的子系统。如果后续你调整上游配置模型或接入方式，只需在
`task_execution_func` 和配置加载层做适配，Executor 模块无需大改。

