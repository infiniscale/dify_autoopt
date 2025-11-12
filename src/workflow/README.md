# 工作流管理模块

## 功能概述

负责Dify工作流的完整生命周期管理，包括工作流发现、运行和发布，支持自动化测试和批量操作。

## 模块组成

### 1. 工作流发现 (discovery.py)
- 自动发现可用工作流
- 工作流元数据获取
- 工作流依赖分析
- 工作流分类管理

### 2. 工作流运行 (runner.py)
- 工作流执行控制
- 运行状态监控
- 输入输出管理
- 错误处理重试

### 3. 工作流发布 (publisher.py)
- 工作流版本管理
- 发布流程控制
- 回滚机制
- 发布状态跟踪

## 功能特性

- 🔍 智能工作流发现
- 🚀 并发执行控制
- 📊 运行状态监控
- 🔄 自动重试机制
- 📝 详细执行日志
- 🎯 批量操作支持

## 使用示例

```python
# 工作流发现
from src.workflow import WorkflowDiscovery

discovery = WorkflowDiscovery()
workflows = discovery.discover_workflows()

# 工作流运行
from src.workflow import WorkflowRunner

runner = WorkflowRunner()
result = runner.run_workflow(
    workflow_id="workflow_123",
    inputs={"param1": "value1"},
    timeout=300
)

# 工作流发布
from src.workflow import WorkflowPublisher

publisher = WorkflowPublisher()
publisher.publish_workflow(
    workflow_id="workflow_123",
    version="1.0.0"
)
```

## 配置参数

```yaml
workflow:
  discovery:
    include_drafts: false
    max_workflows: 1000
    cache_ttl: 3600

  runner:
    default_timeout: 300
    max_concurrent: 10
    retry_count: 3
    retry_delay: 5

  publisher:
    validation_required: true
    backup_on_publish: true
    rollback_on_failure: true
```

## 数据格式

### 工作流元数据
```json
{
  "id": "workflow_123",
  "name": "测试工作流",
  "description": "这是一个测试工作流",
  "version": "1.0.0",
  "inputs": {
    "param1": {"type": "string", "required": true},
    "param2": {"type": "number", "required": false}
  },
  "outputs": {
    "result": {"type": "object"}
  }
}
```

### 运行结果
```json
{
  "success": true,
  "execution_id": "exec_456",
  "result": {},
  "metrics": {
    "execution_time": 15.2,
    "tokens_used": 150,
    "cost": 0.05
  }
}
```

## 错误处理

- 工作流不存在异常
- 权限不足异常
- 输入参数验证失败
- 运行超时异常
- 网络连接异常
- 发布失败异常

## 最佳实践

1. **工作流发现**
   - 使用过滤器提高查询效率
   - 缓存常用工作流信息
   - 定期更新工作流元数据

2. **工作流运行**
   - 设置合理超时时间
   - 批量操作使用队列管理
   - 监控运行状态和性能

3. **工作流发布**
   - 发布前进行充分测试
   - 使用版本号管理更新
   - 准备快速回滚方案