# 结果采集模块

## 功能概述

负责测试数据的收集、导出和分类，提供全面的数据采集和Excel报告生成功能，支持多格式数据导出。

## 模块组成

### 1. 数据收集器 (data_collector.py)
- 测试结果实时采集
- 性能指标收集
- 错误日志记录
- 数据清洗和标准化

### 2. Excel导出器 (excel_exporter.py)
- Excel格式数据导出
- 多工作表报告生成
- 图表和可视化
- 格式化和样式设置

### 3. 结果分类器 (classifier.py)
- 结果分类标记
- 状态自动识别
- 异常数据检测
- 质量评估评级

## 功能特性

- 📊 实时数据采集
- 📈 Excel报告生成
- 🏷️ 智能分类标记
- 🔍 异常数据检测
- 📋 多格式导出
- 💾 数据持久化存储

## 使用示例

```python
# 数据收集
from src.collector import DataCollector

collector = DataCollector()

# 开始收集数据
collector.start_collection()
collector.collect_result({
    "workflow_id": "wf001",
    "execution_time": 2.5,
    "success": True,
    "tokens_used": 150,
    "input_params": {"prompt": "test data"},
    "output": "some result"
})

# 获取统计结果
stats = collector.get_statistics()
print(f"总执行次数: {stats.total_executions}")
print(f"成功率: {stats.success_rate:.2%}")

# Excel导出
from src.collector import ExcelExporter

exporter = ExcelExporter()
exporter.export_to_excel(
    data=collector.get_all_data(),
    output_file="test_results.xlsx",
    include_charts=True
)

# 结果分类
from src.collector import ResultClassifier

classifier = ResultClassifier()
classification = classifier.classify_results(collector.get_all_results())
print(f"性能优秀: {classification.excellent}")
print(f"性能良好: {classification.good}")
print(f"性能较差: {classification.poor}")
```

## 数据格式

### 测试结果数据结构
```python
@dataclass
class TestResult:
    workflow_id: str
    execution_id: str
    timestamp: datetime
    success: bool
    execution_time: float
    tokens_used: int
    cost: float
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
```

### 性能指标
```python
@dataclass
class PerformanceMetrics:
    avg_execution_time: float
    p95_execution_time: float
    p99_execution_time: float
    throughput: float  # 每秒处理数
    success_rate: float
    error_rate: float
    resource_usage: Dict[str, float]
```

## Excel报告结构

### 工作表组织

#### 1. 测试概览
```
| 字段 | 说明 |
|------|------|
| 工作流ID | 唯一标识符 |
| 测试时间 | 执行时间戳 |
| 执行次数 | 总测试次数 |
| 成功次数 | 成功执行数 |
| 失败次数 | 失败执行数 |
| 成功率 | 成功百分比 |
| 平均耗时 | 秒 |
| P95耗时 | 95分位耗时 |
| P99耗时 | 99分位耗时 |
| 总Token数 | 消耗的Token |
| 总成本 | 费用合计 |
```

#### 2. 性能分析
```
| 工作流ID | 执行时间 | Token使用 | 成本 | 成功率 | 性能等级 |
|----------|----------|-----------|------|--------|----------|
| wf001    | 2.3s     | 150       | 0.05 | 98.5%  | 优秀     |
| wf002    | 5.1s     | 280       | 0.12 | 95.2%  | 良好     |
```

#### 3. 错误分析
```
| 错误类型 | 出现次数 | 占比 | 典型错误信息 |
|----------|----------|------|--------------|
| 超时错误 | 15       | 60%  | 任务执行超时 |
| 网络错误 | 5        | 20%  | 连接失败     |
| 参数错误 | 3        | 12%  | 输入参数无效 |
| 其他错误 | 2        | 8%   | 未知错误     |
```

#### 4. 提示词优化
```
| 原始提示词 | 优化后提示词 | 性能提升 | 提升幅度 | 优化建议 |
|------------|--------------|----------|----------|----------|
| ...        | ...          | +35%     | 显著     | 简化逻辑 |
```

## 分类算法

### 性能分类标准
```python
def classify_performance(execution_time: float, token_efficiency: float) -> str:
    """性能分类算法"""
    if execution_time < 2.0 and token_efficiency > 0.8:
        return "优秀"
    elif execution_time < 5.0 and token_efficiency > 0.6:
        return "良好"
    elif execution_time < 10.0 and token_efficiency > 0.4:
        return "一般"
    else:
        return "较差"
```

### 异常检测算法
```python
def detect_anomalies(results: List[TestResult]) -> List[Anomaly]:
    """异常数据检测"""
    anomalies = []
    execution_times = [r.execution_time for r in results]

    # 基于统计的异常检测
    mean = np.mean(execution_times)
    std = np.std(execution_times)
    threshold = mean + 3 * std

    for result in results:
        if result.execution_time > threshold:
            anomalies.append(Anomaly(
                type="performance_anomaly",
                result=result,
                severity="high",
                description=f"执行时间异常: {result.execution_time:.2f}s"
            ))

    return anomalies
```

## 配置参数

```yaml
collector:
  # 数据采集配置
  collection:
    batch_size: 100
    flush_interval: 30  # 秒
    enable_compression: True
    backup_enabled: True
    backup_interval: 3600  # 小时

  # Excel导出配置
  excel_export:
    template_path: "templates/report_template.xlsx"
    auto_width: True
    include_charts: True
    chart_types: ["bar", "line", "pie"]
    max_rows_per_sheet: 10000
    date_format: "YYYY-MM-DD HH:mm:ss"

  # 分类配置
  classification:
    performance_thresholds:
      excellent: {"execution_time": 2.0, "token_efficiency": 0.8}
      good: {"execution_time": 5.0, "token_efficiency": 0.6}
      fair: {"execution_time": 10.0, "token_efficiency": 0.4}
      poor: {"execution_time": float("inf"), "token_efficiency": 0.0}

    anomaly_detection:
      enable_statistical: True
      enable_ml: False
      confidence_level: 0.95
      window_size: 100
```

## 高级功能

### 数据清洗
```python
# 自动数据清洗
collector = DataCollector(
    enable_cleaning=True,
    cleaning_rules={
        "remove_duplicates": True,
        "fill_missing_values": "auto",
        "outlier_detection": True,
        "format_validation": True
    }
)
```

### 增量收集
```python
# 支持增量数据收集
collector = IncrementalCollector(
    collection_id: str,
    checkpoint_interval: 60,
    resume_on_restart: True
)
```

### 自定义分类器
```python
class CustomClassifier ResultClassifier):
    def classify(self, result: TestResult) -> str:
        # 自定义分类逻辑
        if result.workflow_id.startswith("critical_"):
            return self.classify_critical_workflow(result)
        else:
            return super().classify(result)
```

## 错误处理

### 数据采集异常
- 数据格式错误
- 存储空间不足
- 网络连接异常
- 并发访问冲突

### Excel导出异常
- 文件权限错误
- 磁盘空间不足
- 大文件处理异常
- 格式转换错误

### 分类异常
- 数据缺失异常
- 分类规则冲突
- 阈值设置错误
- 算法执行失败

## 性能优化

1. **内存管理**
   - 批量数据处理
   - 及時释放内存
   - 使用生成器处理大数据

2. **存储优化**
   - 数据压缩存储
   - 索引优化查询
   - 分区存储管理

3. **导出优化**
   - 流式导出大文件
   - 异步导出处理
   - 分片导出管理