# 报告模块

## 功能概述

负责测试结果的智能分析、报告生成和优化建议，提供基于AI的深度分析和智能优化方案。

## 模块组成

### 1. 结果分析器 (analyzer.py)
- 测试结果深度分析
- 性能趋势分析
- 关联性分析
- 根因分析

### 2. 报告生成器 (generator.py)
- 多格式报告生成
- 智能报告模板
 可视化图表生成
- 自动报告分发

### 3. 优化建议器 (optimizer.py)
- 性能优化建议
- 提示词改进方案
- 资源配置优化
- 最佳实践推荐

## 功能特性

- 🧠 AI智能分析
- 📊 多维度数据挖掘
- 📄 专业报告生成
- 💡 优化建议推荐
- 📈 趋势预测分析
- 🎯 精准问题定位

## 使用示例

```python
# 结果分析
from src.report import ResultAnalyzer

analyzer = ResultAnalyzer()

# 基础分析
basic_analysis = analyzer.analyze_basic_metrics(test_results)
print(f"平均执行时间: {basic_analysis.avg_execution_time}")

# 趋势分析
trend_analysis = analyzer.analyze_trends(historical_data)
print(f"性能趋势: {trend_analysis.performance_trend}")

# 关联性分析
correlation = analyzer.analyze_correlations(test_results)
print(f"影响最大的因素: {correlation.top_factors}")

# 报告生成
from src.report import ReportGenerator

generator = ReportGenerator()

# 生成HTML报告
html_report = generator.generate_html_report(
    analysis_results=basic_analysis,
    template="modern",
    output_file="report.html"
)

# 生成PDF报告
pdf_report = generator.generate_pdf_report(
    analysis_results=basic_analysis,
    output_file="report.pdf",
    include_charts=True
)

# 优化建议
from src.report import OptimizationAdvisor

advisor = OptimizationAdvisor()
recommendations = advisor.get_recommendations(test_results, analysis_data)

for rec in recommendations:
    print(f"建议: {rec.description}")
    print(f"预期提升: {rec.expected_improvement}")
    print(f"实施难度: {rec.difficulty}")
```

## 分析维度

### 1. 性能分析
```python
@dataclass
class PerformanceAnalysis:
    execution_time_metrics: Dict[str, float]  # 平均值、P95、P99等
    throughput_metrics: Dict[str, float]        # 吞吐量指标
    resource_utilization: Dict[str, float]      # 资源使用率
    scalability_metrics: Dict[str, float]      # 可扩展性指标
    efficiency_score: float                      # 效率评分 (0-100)
```

### 2. 质量分析
```python
@dataclass
class QualityAnalysis:
    success_rate: float                         # 成功率
    error_distribution: Dict[str, int]           # 错误分布
    reliability_score: float                     # 可靠性评分
    consistency_metrics: Dict[str, float]       # 一致性指标
    stability_trend: str                        # 稳定性趋势
```

### 3. 成本分析
```python
@dataclass
class CostAnalysis:
    token_consumption: Dict[str, float]         # Token消耗统计
    cost_breakdown: Dict[str, float]            # 成本分解
    cost_efficiency: float                       # 成本效率
    optimization_potential: float               # 优化潜力
    roi_estimation: float                       # 投资回报率估算
```

## 报告模板

### 1. 执行摘要报告
```markdown
# 测试执行摘要

## 核心指标
- **总体评分**: 85/100
- **成功率**: 95.2%
- **平均执行时间**: 2.3s
- **成本效率**: 优秀

## 主要发现
1. 性能表现稳定，P99响应时间 < 5s
2. 成本控制良好，Token使用效率高
3. 发现3个性能瓶颈需要优化

## 改进建议
- 优化提示词长度，预计提升15%效率
- 调整并发参数，建议提升至10个并发
```

### 2. 详细分析报告
```markdown
# 详细性能分析报告

## 1. 执行时间分析
| 指标 | 数值 | 评估标准 |
|------|------|----------|
| 平均值 | 2.3s | ✓ 优秀 |
| P50 | 2.1s | ✓ 优秀 |
| P95 | 3.8s | ✓ 良好 |
| P99 | 4.9s | ✓ 良好 |

## 2. 错误分析
- 超时错误: 2.1%
- 网络错误: 1.3%
- 参数错误: 0.8%
- 其他错误: 0.6%

## 3. 趋势分析
```

### 3. 优化建议报告
```markdown
# 智能优化建议

## 高优先级建议
1. **提示词优化**
   - 当前效率: 75%
   - 优化后预期: 90%
   - 实施难度: 中等
   - 预期收益: +20% 性能提升

2. **并发配置调优**
   - 当前并发数: 5
   - 建议并发数: 10
   - 实施难度: 低
   - 预期收益: +25% 吞吐量
```

## AI分析算法

### 1. 性能分析算法
```python
class PerformanceAnalyzer:
    def analyze_performance(self, results: List[TestResult]) -> PerformanceAnalysis:
        """深度性能分析"""
        execution_times = [r.execution_time for r in results]

        # 基础统计
        avg_time = np.mean(execution_times)
        p50 = np.percentile(execution_times, 50)
        p95 = np.percentile(execution_times, 95)
        p99 = np.percentile(execution_times, 99)

        # 效率评分算法
        efficiency_score = self.calculate_efficiency_score(execution_times)

        return PerformanceAnalysis(
            execution_time_metrics={
                "avg": avg_time, "p50": p50,
                "p95": p95, "p99": p99
            },
            efficiency_score=efficiency_score
        )

    def calculate_efficiency_score(self, times: List[float]) -> float:
        """效率评分算法"""
        # 基于多个维度计算效率分
        speed_score = self.calculate_speed_score(times)
        consistency_score = self.calculate_consistency_score(times)

        return (speed_score * 0.7 + consistency_score * 0.3) * 100
```

### 2. 根因分析算法
```python
def analyze_root_cause(self, failures: List[TestFailure]) -> List[Cause]:
    """根因分析"""
    causes = []

    # 聚类分析
    error_clusters = self.cluster_failures(failures)

    # 关联性分析
    for cluster in error_clusters:
        common_patterns = self.find_common_patterns(cluster)
        if common_patterns:
            causes.append(Cause(
                type="common_pattern",
                description=common_patterns.description,
                confidence=common_patterns.confidence,
                affected_workflows=cluster.workflows
            ))

    return causes
```

### 3. 趋势预测算法
```python
def predict_performance_trend(self, historical_data: List[PerformanceData]) -> Trend:
    """性能趋势预测"""
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    # 准备数据
    X = np.array([[i] for i in range(len(historical_data))])
    y = np.array([data.avg_execution_time for data in historical_data])

    # 多项式回归
    poly_features = PolynomialFeatures(degree=2)
    X_poly = poly_features.fit_transform(X)

    model = LinearRegression()
    model.fit(X_poly, y)

    # 预测未来趋势
    future_X = np.array([[len(historical_data) + i] for i in range(1, 6)])
    future_X_poly = poly_features.transform(future_X)
    predictions = model.predict(future_X)

    return Trend(
        predictions=predictions.tolist(),
        confidence=model.score(X_poly, y),
        trend_direction=self._calculate_trend_direction(predictions)
    )
```

## 优化建议生成

### 1. 性能优化建议
```python
class PerformanceOptimizer:
    def generate_suggestions(self, analysis: PerformanceAnalysis) -> List[OptimizationSuggestion]:
        suggestions = []

        # 执行时间优化
        if analysis.execution_time_metrics["p99"] > 5.0:
            suggestions.append(
                OptimizationSuggestion(
                    category="performance",
                    title="优化执行延迟",
                    description="P99执行时间过长，建议优化推理算法",
                    implementation="减少提示词长度，使用更高效的模型",
                    expected_improvement="20-30% 延迟降低",
                    priority="high",
                    difficulty="medium"
                )
            )

        # 并发优化
        if analysis.efficiency_score < 70:
            suggestions.append(
                OptimizationSuggestion(
                    category="concurrency",
                    title="提升并发能力",
                    description="当前效率较低，建议增加并发数",
                    implementation="将并发数从5增加到10",
                    expected_improvement="25% 吞吐量提升",
                    priority="medium",
                    difficulty="low"
                )
            )

        return suggestions
```

### 2. 成本优化建议
```python
def suggest_cost_optimizations(self, cost_analysis: CostAnalysis) -> List[CostSuggestion]:
    suggestions = []

    # Token使用优化
    if cost_analysis.token_efficiency < 0.8:
        suggestions.append(
            CostSuggestion(
                area="token_optimization",
                description="Token使用效率有待提升",
                current_efficiency=cost_analysis.token_efficiency,
                target_efficiency=0.9,
                actions=[
                    "精简提示词内容",
                    "使用更高效的模型",
                    "实现结果缓存"
                ],
                estimated_savings="15-20%"
            )
        )

    return suggestions
```

## 可视化图表

### 1. 性能趋势图
```python
def create_performance_chart(self, data: List[PerformanceData]) -> Chart:
    """创建性能趋势图"""
    return Chart(
        type="line",
        title="性能趋势分析",
        x_axis="时间",
        y_axis="执行时间(秒)",
        datasets=[
            Dataset(name="平均时间", data=data.avg_times),
            Dataset(name="P95时间", data=data.p95_times),
            Dataset(name="P99时间", data=data.p99_times)
        ]
    )
```

### 2. 错误分布饼图
```python
def create_error_distribution_chart(self, errors: List[ErrorData]) -> Chart:
    """创建错误分布图"""
    return Chart(
        type="pie",
        title="错误类型分布",
        datasets=[
            Dataset(name错误类型", data=error_distribution),
            Dataset("占比", data=error_percentages)
        ]
    )
```

## 配置参数

```yaml
report:
  # 分析器配置
  analyzer:
    enable_trend_analysis: True
    enable_root_cause_analysis: True
    enable_ml_prediction: True
    confidence_threshold: 0.8
    min_sample_size: 30

  # 报告生成器配置
  generator:
    template_dir: "templates/reports"
    output_dir: "reports"
    formats: ["html", "pdf", "excel"]
    include_charts: True
    auto_distribute: True
    distribution_list: ["admin@company.com"]

  # 优化建议器配置
  optimizer:
    enable_ai_recommendations: True
    suggestion_categories: ["performance", "cost", "quality", "reliability"]
    max_suggestions_per_category: 5
    min_improvement_threshold: 0.05  # 5%提升才建议
```

## 高级功能

### 1. 自动化报告分发
```python
# 定时报告生成和分发
scheduler = ReportScheduler()
scheduler.add_cron_job(
    name="daily_performance_report",
    schedule="0 9 * * *",  # 每天9点
    recipients=["team@company.com"],
    template="daily_report",
    format="html"
)
```

### 2. 智能异常检测
```python
# 基于ML的异常检测
anomaly_detector = MLAnomalyDetector()
anomalies = anomaly_detector.detect(
    data=performance_data,
    sensitivity=0.95,
    min_anomaly_score=0.8
)
```

### 3. 自定义分析规则
```python
# 自定义分析规则
custom_analyzer = CustomAnalyzer()
custom_analyzer.add_rule(
    condition=lambda x: x.execution_time > 10,
    action=RuleAction.SUGGEST_OPTIMIZATION,
    message="执行时间过长，建议优化"
)
```