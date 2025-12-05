# 智能优化模块

## 功能概述

负责Dify工作流中LLM提示词的智能提取、分析和优化，提供基于大模型的自动化提示词优化和版本管理功能。

## 模块组成

### 1. 提示词提取 (prompt_extractor.py)
- 工作流提示词自动识别
- 多层级提示词提取
- 提示词上下文分析
- 结构化数据转换

### 2. LLM分析器 (llm_analyzer.py)
- 提示词效果评估
- 多维度质量分析
- 性能基准测试
- 智能评分算法

### 3. 优化引擎 (optimization_engine.py)
- AI驱动的优化算法
- 多策略优化方案
- 迭代式改进机制
- A/B测试验证

### 4. 版本管理 (version_manager.py)
- 提示词版本控制
- 回滚机制管理
- 变更历史追踪
- 效果对比分析

## 功能特性

- 🔍 智能提示词提取
- 🧠 AI效果评估
- 🎯 自动优化建议
- 📚 完整版本管理
- 🔄 迭代改进机制
- 📊 效果对比分析

## 使用示例

```python
# 提示词提取
from src.optimizer import PromptExtractor

extractor = PromptExtractor()
prompts = extractor.extract_prompts_from_workflow(workflow_id="wf001")

for prompt in prompts:
    print(f"提示词ID: {prompt.id}")
    print(f"内容: {prompt.text[:100]}...")
    print(f"上下文: {prompt.context}")
    print("---")

# LLM分析
from src.optimizer import LLMAnalyzer

analyzer = LLMAnalyzer()
analysis = analyzer.analyze_prompt(prompt_text)
print(f"清晰度评分: {analysis.clarity_score}")
print(f"相关性评分: {analysis.relevance_score}")
print(f"效率评分: {analysis.efficiency_score}")
print(f"综合评分: {analysis.overall_score}")

# 优化引擎
from src.optimizer import OptimizationEngine

optimizer = OptimizationEngine()
optimization_result = optimizer.optimize(
    original_prompt=prompt_text,
    target_metrics=["clarity", "efficiency", "accuracy"],
    optimization_strategy="iterative"
)

print(f"优化后提示词: {optimization_result.optimized_prompt}")
print(f"预期提升: {optimization_result.expected_improvement}")

# 版本管理
from src.optimizer import VersionManager

version_manager = VersionManager()
version = version_manager.create_version(
    prompt_id="prompt_001",
    prompt_text=original_prompt,
    optimization_result=optimization_result
)
print(f"版本号: {version.version}")
print(f"创建时间: {version.created_at}")
```

## 提示词数据结构

### 提示词对象
```python
@dataclass
class Prompt:
    id: str
    workflow_id: str
    node_id: str
    text: str
    role: str  # system, user, assistant
    context: Dict[str, Any]
    variables: List[str]  # 变量占位符
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
```

### 提示词分析结果
```python
@dataclass
class PromptAnalysis:
    prompt_id: str
    clarity_score: float      # 清晰度评分 (0-100)
    relevance_score: float    # 相关性评分 (0-100)
    efficiency_score: float  # 效率评分 (0-100)
    safety_score: float       # 安全性评分 (0-100)
    overall_score: float      # 综合评分 (0-100)
    issues: List[str]        # 发现的问题
    suggestions: List[str]   # 改进建议
    metrics: Dict[str, float]  # 各项指标
```

### 优化结果
```python
@dataclass
class OptimizationResult:
    original_prompt: str
    optimized_prompt: str
    improvement_score: float     # 整体提升分数
    improvements: Dict[str, float]  # 各维度提升
    strategy_used: str          # 使用的优化策略
    confidence: float           # 优化结果置信度
    validation_results: Dict[str, Any]  # 验证结果
```

## 优化策略

### 1. 聚焦优化
```python
def focus_optimization(prompt: str, target_aspect: str) -> str:
    """针对特定方面的优化"""
    if target_aspect == "clarity":
        # 提高提示词清晰度
        return simplify_language(prompt)
    elif target_aspect == "efficiency":
        # 提高效率，减少Token使用
        return compress_prompt(prompt)
    elif target_aspect == "safety":
        # 增强安全性
        return add_safety_constraints(prompt)
```

### 2. 多目标优化
```python
def multi_objective_optimization(prompt: str, weights: Dict[str, float]) -> str:
    """多目标优化算法"""
    # 基于权重平衡多个优化目标
    candidates = []
    for strategy in optimization_strategies:
        candidate = apply_strategy(prompt, strategy)
        score = evaluate_multi_objective(candidate, weights)
        candidates.append((candidate, score))

    # 选择最优候选
    return max(candidates, key=lambda x: x[1][0])
```

### 3. 迭代优化
```python
def iterative_optimization(prompt: str, max_iterations: int = 5) -> OptimizationResult:
    """迭代优化算法"""
    current_prompt = prompt
    history = []

    for iteration in range(max_iterations):
        # 生成优化候选
        candidates = generate_optimization_candidates(current_prompt)

        # 评估候选
        best_candidate = max(candidates, key=lambda x: evaluate_quality(x))

        # 检查是否达到优化目标
        if is_optimal(best_candidate, current_prompt):
            break

        current_prompt = best_candidate
        history.append({
            "iteration": iteration + 1,
            "prompt": current_prompt,
            "score": evaluate_quality(current_prompt)
        })

    return OptimizationResult(
        original_prompt=prompt,
        optimized_prompt=current_prompt,
        improvement_score=calculate_improvement(prompt, current_prompt),
        optimization_history=history
    )
```

## 版本管理系统

### 1. 版本创建
```python
def create_prompt_version(prompt_id: str, prompt_text: str,
                         changes: List[Change], author: str) -> PromptVersion:
    """创建新版本"""
    # 生成版本号
    version = generate Semantic Version(last_version_of(prompt_id))

    # 验证版本
    validate_prompt_version(prompt_text, version)

    # 存储版本
    version_record = PromptVersion(
        prompt_id=prompt_id,
        version=version,
        text=prompt_text,
        changes=changes,
        author=author,
        created_at=datetime.now()
    )

    save_version(version_record)
    return version_record
```

### 2. 版本对比
```python
def compare_prompt_versions(v1: PromptVersion, v2: PromptVersion) -> ComparisonResult:
    """版本对比分析"""
    diff = calculate_text_diff(v1.text, v2.text)

    performance_metrics = compare_performance_metrics(v1, v2)

    return ComparisonResult(
        version1=v1.version,
        version2=v2.version,
        text_changes=diff,
        performance_improvement=performance_metrics,
        recommendation=generate_comparison_recommendation(diff, performance_metrics)
    )
```

### 3. 版本回滚
```python
def rollback_to_version(prompt_id: str, target_version: str, reason: str) -> bool:
    """回滚到指定版本"""
    target_version_prompt = get_version(prompt_id, target_version)

    if not target_version_prompt:
        raise VersionNotFoundError(f"Version {target_version} not found")

    # 验证回滚安全性
    if not is_safe_rollback(target_version_prompt):
        return False

    # 执行回滚
    current_version = rollback(prompt_id, target_version_prompt.text, reason)

    # 创建回滚记录
    create_rollback_record(prompt_id, target_version, reason, current_version)

    return True
```

## AI分析算法

### 1. 质量评估算法
```python
class PromptQualityEvaluator:
    def evaluate_clarity(self, prompt: str) -> float:
        """评估提示词清晰度"""
        # 基于语言复杂度分析
        readability_score = self.calculate_readability(prompt)

        # 基于指令明确性分析
        instruction_clarity = self.analyze_instruction_clarity(prompt)

        # 基于结构化程度分析
        structure_score = self.analyze_structure(prompt)

        return (readability_score * 0.4 +
                instruction_clarity * 0.4 +
                structure_score * 0.2) * 100

    def evaluate_efficiency(self, prompt: str) -> float:
        """评估提示词效率"""
        token_count = self.count_tokens(prompt)
        information_density = self.calculate_information_density(prompt)

        # 效率评分：信息密度 / token数量
        efficiency = information_density / token_count
        return min(efficiency * 100, 100)

    def evaluate_safety(self, prompt: str) -> float:
        """评估提示词安全性"""
        # 检查潜在的有害内容
        safety_issues = self.detect_safety_issues(prompt)

        # 检查偏见内容
        bias_score = self.detect_bias(prompt)

        # 检查隐私泄露风险
        privacy_risk = self.assess_privacy_risk(prompt)

        base_score = 100
        safety_penalty = (safety_issues * 30 + bias_score * 40 + privacy_risk * 30)

        return max(base_score - safety_penalty, 0)
```

### 2. 优化算法
```python
class PromptOptimizer:
    def optimize_structure(self, prompt: str) -> str:
        """优化提示词结构"""
        # 识别结构化模式
        structure_patterns = self.identify_structure_patterns(prompt)

        # 优化结构顺序
        optimized = self.reorder_prompt_structure(prompt, structure_patterns)

        # 添加必要的分隔符
        optimized = self.add_structural_separators(optimized)

        return optimized

    def optimize_language(self, prompt: str) -> str:
        """优化提示词语言"""
        # 简化复杂句子
        simplified = self.simplify_sentences(prompt)

        # 统一术语使用
        terminology_standardized = self.standardize_terminology(simplified)

        # 增强指令明确性
        enhanced_clarity = self.enhance_instruction_clarity(terminology_standardized)

        return enhanced_clarity

    def optimize_variables(self, prompt: str) -> str:
        """优化提示词变量"""
        # 识别变量占位符
        variables = self.extract_variables(prompt)

        # 优化变量定义
        optimized_variables = self.optimize_variable_definitions(variables)

        # 替换变量使用
        optimized = self.replace_variables(prompt, optimized_variables)

        return optimized
```

## 配置参数

```yaml
optimizer:
  # 提示词提取配置
  extraction:
    max_prompt_length: 10000
    extraction_strategies: ["llm_call", "text_template", "json_structure"]
    enable_context_analysis: True
    variable_pattern: "\\{\\{(\\w+)\\}\\}"

  # 分析器配置
  analyzer:
    evaluation_metrics:
      - clarity
      - relevance
      - efficiency
      - safety
      - accuracy
    model_for_analysis: "gpt-4"
    analysis_batch_size: 10
    confidence_threshold: 0.8

  # 优化引擎配置
  optimization:
    strategies:
      - name: "focus_clarity"
        weight: 0.3
        enabled: True
      - name: "focus_efficiency"
        weight: 0.3
        enabled: True
      - name: "focus_safety"
        weight: 0.2
        enabled: True
      - name: "multi_objective"
        weight: 0.2
        enabled: True

    max_iterations: 5
    improvement_threshold: 0.05  # 5%提升才接受
    validation_sample_size: 20

  # 版本管理配置
  versioning:
    max_versions_kept: 50
    auto_backup: True
    backup_interval: 24  # 小时
    enable_branching: True
    merge_policy: "squash"
```

## 高级功能

### 1. 智能提示词模板
```python
def generate_prompt_template(workflows: List[Workflow]) -> PromptTemplate:
    """基于学习生成提示词模板"""
    # 分析工作流模式
    patterns = analyze_workflow_patterns(workflows)

    # 生成通用模板
    template = create_template_from_patterns(patterns)

    # 验证模板有效性
    validated_template = validate_template(template)

    return validated_template
```

### 2. A/B测试支持
```python
def run_ab_test(prompt_a: str, prompt_b: str,
                test_data: List[TestCase]) -> ABTestResult:
    """运行A/B测试"""
    # 并行测试两个版本
    results_a = run_prompt_test(prompt_a, test_data)
    results_b = run_prompt_test(prompt_b, test_data)

    # 统计分析
    statistical_analysis = compare_results_statistics(results_a, results_b)

    # 生成测试报告
    test_result = ABTestResult(
        prompt_a_score=calculate_score(results_a),
        prompt_b_score=calculate_score(results_b),
        significant_difference=statistical_analysis.is_significant,
        confidence_interval=statistical_analysis.confidence_interval,
        winner=statistical_analysis.winner
    )

    return test_result
```

### 3. 集成学习优化
```python
def ensemble_optimization(prompt: str, optimizers: List[Optimizer]) -> str:
    """集成学习优化"""
    # 运行多个优化器
    candidates = []
    for optimizer in optimizers:
        candidate = optimizer.optimize(prompt)
        score = evaluate_candidate(candidate)
        candidates.append((candidate, score))

    # 使用集成方法选择最优结果
    ensemble_result = ensemble_selection(candidates)

    return ensemble_result.best_candidate
```

## 错误处理

### 1. 提取异常
- 提示词格式错误
- 上下文解析失败
- 变量识别错误
- 权限访问受限

### 2. 分析异常
- 模型调用失败
- 评分算法异常
- 数据预处理错误
- 超时处理

### 3. 优化异常
- 优化策略冲突
- 生成结果异常
- 验证条件不满足
- 资源限制突破

### 4. 版本控制异常
- 版本冲突
- 回滚失败
- 历史记录损坏
- 合并冲突处理

## 最佳实践

1. **提示词设计**
   - 明确目标和约束
   - 使用结构化格式
   - 避免歧义和复杂句式
   - 合理使用变量和模板

2. **优化策略**
   - 基于数据驱动决策
   - 采用渐进式优化
   - 重视验证和测试
   - 保持版本控制

3. **版本管理**
   - 清晰的版本命名
   - 详细的变更记录
   - 快速回滚机制
   - 定期清理历史版本