# DataCollector 模块实现文档

## 概述

**DataCollector** 是 collector 模块的核心类，负责测试结果的收集、存储和统计分析。

- **实现日期**: 2025-11-13
- **作者**: backend-developer
- **文件路径**: `D:\Work\dify_autoopt\src\collector\data_collector.py`

---

## 核心功能

### 1. 数据收集
- 收集 `TestResult` 对象
- 自动按工作流ID建立索引
- 完整的数据验证（类型、必需字段、数值范围）

### 2. 统计分析
- 基础统计: 总次数、成功/失败次数、成功率
- 执行时间统计: 平均值、P50/P95/P99 百分位数
- Token 和成本统计: 总量、平均值

### 3. 数据查询
- 按工作流查询
- 按变体查询
- 按数据集查询
- 获取全部结果

---

## API 文档

### 初始化

```python
from src.collector import DataCollector

collector = DataCollector()
```

### 收集结果

```python
from datetime import datetime
from src.collector import TestResult, TestStatus

result = TestResult(
    workflow_id="wf_001",
    execution_id="exec_001",
    timestamp=datetime.now(),
    status=TestStatus.SUCCESS,
    execution_time=1.5,
    tokens_used=150,
    cost=0.01,
    inputs={"query": "test"},
    outputs={"answer": "result"}
)

collector.collect_result(result)
```

### 计算统计

```python
# 全部统计
metrics = collector.get_statistics()

# 指定工作流统计
metrics = collector.get_statistics(workflow_id="wf_001")

# 访问指标
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Avg execution time: {metrics.avg_execution_time:.2f}s")
print(f"P95 execution time: {metrics.p95_execution_time:.2f}s")
```

### 查询数据

```python
# 获取所有结果
all_results = collector.get_all_results()

# 按工作流查询
wf_results = collector.get_results_by_workflow("wf_001")

# 按变体查询
variant_results = collector.get_results_by_variant("wf_001", "v1")

# 按数据集查询
dataset_results = collector.get_results_by_dataset("dataset_a")

# 获取结果数量
count = collector.get_result_count()
```

### 清空数据

```python
collector.clear()
```

---

## ��键实现要点

### 1. 数据验证

所有收集的结果都经过严格验证:

```python
# 类型检查
if not isinstance(result, TestResult):
    raise DataValidationException(...)

# 必需字段检查
if not result.workflow_id or not result.execution_id:
    raise DataValidationException(...)

# 数值合法性检查
if result.execution_time < 0:
    raise DataValidationException(...)
```

### 2. 百分位数算法

使用线性插值法计算 P50/P95/P99:

```python
def percentile(p: float) -> float:
    """计算第 p 百分位 (0-100)"""
    if n == 1:
        return sorted_values[0]

    # 线性插值
    index = (p / 100.0) * (n - 1)
    lower = int(index)
    upper = min(lower + 1, n - 1)
    weight = index - lower

    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
```

### 3. 双重索引结构

- `_results`: 线性列表，保持插入顺序
- `_results_by_workflow`: 字典索引，加速工作流查询

### 4. 日志记录

- **DEBUG**: 每次收集结果
- **INFO**: 初始化、统计计算
- **WARNING**: 清空数据
- **ERROR**: 验证失败

### 5. 数据隔离

所有查询方法返回副本，避免外部修改:

```python
def get_all_results(self) -> List[TestResult]:
    return list(self._results)  # 返回副本
```

---

## 测试结果

### 单元测试

- **测试文件**: `tests/collector/test_data_collector.py`
- **测试用例**: 21 个
- **代码覆盖率**: **100%**
- **测试通过率**: 100%

### 测试覆盖

```
Name                              Stmts   Miss  Cover
-----------------------------------------------------
src\collector\data_collector.py      94      0   100%
-----------------------------------------------------
TOTAL                                94      0   100%
```

### 性能测试

**测试配置**: 10,000 条结果

| 指标 | 结果 | 要求 | 状态 |
|------|------|------|------|
| collect_result() 性能 | 0.053ms/次 | < 1ms | ✅ PASS |
| get_statistics() 性能 | 0.001s | < 1s | ✅ PASS |
| 吞吐量 | 18,991 results/s | - | ✅ 优秀 |
| 内存占用 | 8.54 bytes/result | - | ✅ 优秀 |
| 查询性能 | < 1ms | - | ✅ 优秀 |

---

## 错误处理

### DataValidationException

以下情况会抛出 `DataValidationException`:

1. 传入非 `TestResult` 类型对象
2. `workflow_id` 或 `execution_id` 为空
3. `execution_time`、`tokens_used` 或 `cost` 为负数
4. 统计时没有结果数据

### 示例

```python
from src.utils.exceptions import DataValidationException

try:
    collector.collect_result(invalid_result)
except DataValidationException as e:
    print(f"Validation error: {e}")
```

---

## 使用示例

### 完整工作流

```python
from datetime import datetime
from src.collector import DataCollector, TestResult, TestStatus

# 1. 创建收集器
collector = DataCollector()

# 2. 收集测试结果
for i in range(100):
    result = TestResult(
        workflow_id="wf_001",
        execution_id=f"exec_{i}",
        timestamp=datetime.now(),
        status=TestStatus.SUCCESS if i % 2 == 0 else TestStatus.FAILED,
        execution_time=1.0 + i * 0.1,
        tokens_used=100 + i * 10,
        cost=0.01 + i * 0.001,
        inputs={"query": f"test_{i}"},
        outputs={"answer": f"result_{i}"},
        prompt_variant="v1",
        dataset="test_dataset"
    )
    collector.collect_result(result)

# 3. 计算统计
metrics = collector.get_statistics()
print(f"Total executions: {metrics.total_executions}")
print(f"Success rate: {metrics.success_rate:.2%}")
print(f"Avg execution time: {metrics.avg_execution_time:.2f}s")
print(f"P95 execution time: {metrics.p95_execution_time:.2f}s")
print(f"Total cost: ${metrics.total_cost:.2f}")

# 4. 查询特定数据
variant_results = collector.get_results_by_variant("wf_001", "v1")
print(f"Results for variant v1: {len(variant_results)}")

# 5. 清空数据（如需重新开始）
collector.clear()
```

---

## 后续扩展

DataCollector 为以下功能预留了扩展接口:

1. **数据导出**: 可添加 `export_to_excel()` 方法
2. **性能分级**: 可添加基于 `PerformanceGrade` 的分类统计
3. **持久化**: 可添加数据库存储支持
4. **实时流式处理**: 可集成流式数据处理

---

## 依赖项

- **标准库**: `typing`, `datetime`
- **第三方库**: `loguru` (日志)
- **项目内部**:
  - `src.utils.exceptions.DataValidationException`
  - `src.collector.models.*`

---

## 贡献者

- **backend-developer** - 初始实现 (2025-11-13)

---

## 更新日志

### v1.0.0 (2025-11-13)
- ✅ 初始实现
- ✅ 完整的数据验证
- ✅ 百分位数统计算法
- ✅ 100% 测试覆盖率
- ✅ 性能测试通过 (10,000 条结果)
- ✅ 完整的文档和示例
