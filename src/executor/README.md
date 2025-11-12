# 调用执行模块

## 功能概述

负责任务的并发执行和调度管理，提供高性能的并发执行器，支持大规模工作流测试和任务调度。

## 模块组成

### 1. 并发执行器 (concurrent.py)
- 多线程/多进程并发
- 任务队列管理
- 资源池控制
- 异常处理机制

### 2. 任务调度器 (scheduler.py)
- 任务优先级管理
- 定时任务执行
- 负载均衡调度
- 资源监控管理

## 功能特性

- 🚀 高性能并发执行
- ⚡ 智能负载均衡
- 📊 实时资源监控
- 🔄 任务自动重试
- ⏰ 定时任务支持
- 🛡️ 异常恢复机制

## 使用示例

```python
# 并发执行
from src.executor import ConcurrentExecutor

executor = ConcurrentExecutor(
    max_workers=10,
    timeout=300,
    retry_count=3
)

tasks = [
    {"workflow_id": "wf1", "inputs": {...}},
    {"workflow_id": "wf2", "inputs": {...}},
    {"workflow_id": "wf3", "inputs": {...}}
]

results = executor.run_tasks(tasks)
print(f"执行完成，成功: {len(results.successful)}, 失败: {len(results.failed)}")

# 任务调度
from src.executor import TaskScheduler

scheduler = TaskScheduler()

# 添加定时任务
scheduler.add_cron_task(
    name="daily_test",
    workflow_id="test_workflow",
    schedule="0 2 * * *",  # 每天2点执行
    inputs={"test_data": "auto_generated"}
)

# 启动调度器
scheduler.start()
```

## 配置参数

```yaml
executor:
  concurrent:
    max_workers: 10
    worker_type: "thread"  # thread/process
    queue_size: 1000
    task_timeout: 300
    retry_count: 3
    retry_delay: 5

  scheduler:
    max_concurrent_tasks: 50
    task_priorities: ["high", "medium", "low"]
    resource_limits:
      cpu_percent: 80
      memory_percent: 70
      network_kbps: 1000

```

## 执行器类型

### 1. 线程池执行器
```python
executor = ThreadPoolExecutor(
    max_workers=20,
    queue_size=100,
    daemon=True
)
```

### 2. 进程池执行器
```python
executor = ProcessPoolExecutor(
    max_workers=8,
    queue_size=50,
    memory_limit="512MB"
)
```

### 3. 异步执行器
```python
executor = AsyncExecutor(
    max_coroutines=100,
    event_loop_policy="default"
)
```

## 任务定义

### 基础任务结构
```python
@dataclass
class Task:
    id: str
    workflow_id: str
    inputs: Dict[str, Any]
    priority: str = "medium"
    timeout: int = 300
    retry_count: int = 3
    dependencies: List[str] = None
    metadata: Dict[str, Any] = None
```

### 任务执行结果
```python
@dataclass
class TaskResult:
    task_id: str
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    metrics: Dict[str, Any] = None
    timestamp: datetime = None
```

## 调度策略

### 1. 优先级调度
- 高: 100分
- 中: 50分
- 低: 10分

### 2. 负载均衡
- CPU使用率权重: 40%
- 内存使用率权重: 30%
- 网络带宽权重: 20%
- 任务队列长度权重: 10%

### 3. 资源限制
```yaml
resource_limits:
  cpu_cores: 8
  memory_mb: 8192
  disk_io_mb_per_sec: 100
  network_mbps: 100
```

## 监控指标

### 执行状态监控
- 活跃任务数量
- 队列长度
- 成功率
- 平均执行时间
- 资源使用率

### 性能指标
- 吞吐量 (任务/秒)
- 延迟分布 (P50, P95, P99)
- 错误率趋势
- 资源利用率

## 异常处理

### 任务执行异常
```python
class TaskExecutionException(Exception):
    def __init__(self, task_id: str, error: Exception):
        self.task_id = task_id
        self.error = error
        super().__init__(f"Task {task_id} failed: {error}")
```

### 资源限制异常
```python
class ResourceLimitException(Exception):
    def __init__(self, resource_type: str, current: float, limit: float):
        self.resource_type = resource_type
        self.current = current
        self.limit = limit
        super().__init__(f"Resource {resource_type} exceeded: {current}/{limit}")
```

### 超时异常
```python
class TaskTimeoutException(Exception):
    def __init__(self, task_id: str, timeout: int):
        self.task_id = task_id
        self.timeout = timeout
        super().__init__(f"Task {task_id} timed out after {timeout}s")
```

## 高级功能

### 任务依赖管理
```python
# 支持任务间依赖关系
tasks = [
    Task(id="task1", dependencies=[]),
    Task(id="task2", dependencies=["task1"]),
    Task(id="task3", dependencies=["task1", "task2"])
]

executor.run_with_dependencies(tasks)
```

### 动态资源调整
```python
# 根据负载动态调整工作线程数量
executor = DynamicExecutor(
    min_workers=5,
    max_workers=50,
    auto_scale_threshold=0.8
)
```

### 故障恢复
```python
# 自动故障恢复机制
executor = FaultTolerantExecutor(
    checkpoint_enabled=True,
    auto_recovery=True,
    health_check_interval=30
)
```

## 性能优化建议

1. **并发数调优**
   - 根据CPU核心数设置工作线程
   - 监控系统资源使用情况
   - 避免过多上下文切换

2. **任务队列管理**
   - 合理设置队列大小
   - 使用优先级队列
   - 定期清理过期任务

3. **内存管理**
   - 控制并发任务数量
   - 及时释放任务结果
   - 监控内存使用趋势

4. **网络优化**
   - 使用连接池
   - 设置合理的超时时间
   - 实现请求重试机制