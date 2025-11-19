# Optimizer模块实际使用链路和功能说明

## 📋 目录
1. [运行链路](#运行链路)
2. [提取功能说明](#提取功能说明)
3. [使用示例](#使用示例)
4. [API参考](#api参考)

---

## 🔄 运行链路

### 完整优化流程

```
用户调用
   ↓
OptimizerService.run_optimization_cycle(workflow_id)
   ↓
1️⃣ 提取阶段 (_extract_prompts)
   → PromptExtractor.extract_from_workflow()
   → 📦 提取 **所有** LLM节点的prompts
   ↓
2️⃣ 分析阶段 (对每个prompt)
   → PromptAnalyzer.analyze_prompt()
   → 📊 评分：清晰度、效率、信息密度
   → 📝 创建baseline版本
   ↓
3️⃣ 决策阶段 (对每个prompt)
   → ScoringRules.should_optimize()
   → ❓ 分数 < 80分？有Critical问题？
   → ✅ 需要优化 → 继续
   → ❌ 不需要 → 跳过
   ↓
4️⃣ 优化阶段 (多策略+迭代)
   → 尝试所有配置的策略
   → OptimizationEngine.optimize()
   → 🔄 最多迭代N次直到达到质量阈值
   → 🎯 选择最佳结果
   ↓
5️⃣ 版本管理阶段
   → VersionManager.create_version()
   → 📚 保存优化后的版本
   ↓
6️⃣ 生成Patch阶段
   → _create_prompt_patch()
   → 🔧 生成PromptPatch对象
   ↓
返回 List[PromptPatch]
```

### 关键流程细节

```python
# 伪代码展示内部逻辑
def run_optimization_cycle(workflow_id):
    # 1. 提取所有LLM prompts（一次性全部提取）
    prompts = extractor.extract_from_workflow(workflow_dsl)
    # 返回: [prompt1, prompt2, prompt3, ...]

    patches = []

    # 2. 遍历每个prompt（逐个处理）
    for prompt in prompts:
        # 2.1 分析baseline
        analysis = analyzer.analyze_prompt(prompt)

        # 2.2 判断是否需要优化
        if not should_optimize(analysis):
            continue  # 跳过高质量prompt

        # 2.3 尝试多种策略优化
        best_result = None
        for strategy in config.strategies:
            result = optimize_with_iterations(
                prompt,
                strategy,
                max_iterations=3
            )
            if result.is_better_than(best_result):
                best_result = result

        # 2.4 保存最佳结果
        if best_result.confidence >= threshold:
            version_manager.create_version(optimized_prompt)
            patch = create_patch(prompt, best_result)
            patches.append(patch)

    return patches  # 返回需要应用的patches
```

---

## 🔍 提取功能说明

### 问题回答：一次性提取 vs 指定节点？

#### ✅ **当前实现：一次性提取所有LLM节点**

**提取策略：**
```python
# PromptExtractor.extract_from_workflow()
def extract_from_workflow(workflow_dict, workflow_id):
    """
    提取逻辑：
    1. 遍历workflow中的所有节点
    2. 对每个节点调用 extract_from_node()
    3. 只提取 node_type == "llm" 的节点
    4. 忽略其他类型节点（code, http, template等）
    """
    nodes = find_all_nodes(workflow_dict)
    prompts = []

    for node in nodes:
        if node.type == "llm":  # 只处理LLM节点
            prompt = extract_from_node(node)
            prompts.append(prompt)

    return prompts  # 返回所有LLM prompts
```

**支持的节点类型：**
- ✅ `llm` - LLM节点（会提取）
- ✅ `question-classifier` - 问题分类器（会提取）
- ✅ `if-else` - 条件判断（会提取system_prompt）
- ❌ `code` - 代码节点（跳过）
- ❌ `http-request` - HTTP请求（跳过）
- ❌ `template-transform` - 模板转换（跳过）
- ❌ 其他非LLM节点（跳过）

#### ⭐ **也支持单节点提取（通过API）**

虽然主流程是全量提取，但你可以单独使用：

```python
# 方式1: 提取所有LLM节点（推荐）
prompts = extractor.extract_from_workflow(workflow_dict, "wf_001")
# 返回: [prompt1, prompt2, prompt3]

# 方式2: 提取单个节点（如果你只想优化特定节点）
single_node = workflow_dict["graph"]["nodes"][0]
prompt = extractor.extract_from_node(single_node, "wf_001")
# 返回: 单个Prompt对象或None
```

### 提取后的数据结构

```python
# 每个提取的Prompt对象包含：
Prompt(
    id="wf_001_llm_1",              # 唯一ID
    workflow_id="wf_001",            # 所属workflow
    node_id="llm_1",                 # 节点ID
    node_type="llm",                 # 节点类型
    text="You are a helpful...",     # prompt文本
    role="system",                   # 角色（system/user/assistant）
    variables=["user_input", "context"],  # 变量列表
    context={                        # 上下文信息
        "model": "gpt-4",
        "temperature": 0.7
    },
    extracted_at=datetime.now()      # 提取时间
)
```

---

## 💡 使用示例

### 场景1: 优化整个工作流（最常用）

```python
from src.optimizer import OptimizerService
from src.config.models import WorkflowCatalog

# 1. 初始化服务
catalog = WorkflowCatalog.from_yaml("workflows.yaml")
service = OptimizerService(catalog=catalog)

# 2. 运行优化（自动提取所有LLM prompts）
patches = service.run_optimization_cycle(
    workflow_id="wf_001",
    strategy="auto"  # 自动选择最佳策略
)

# 3. 查看结果
print(f"需要优化的prompts: {len(patches)}个")
for patch in patches:
    print(f"节点 {patch.selector.by_id}: {patch.strategy.content[:50]}...")
```

**输出示例：**
```
需要优化的prompts: 2个
节点 llm_1: You are a highly skilled assistant specializing in...
节点 llm_3: Analyze the following text and extract key info...
```

### 场景2: 使用高级配置（多策略+迭代）

```python
from src.optimizer.models import OptimizationConfig, OptimizationStrategy

# 配置多策略优化
config = OptimizationConfig(
    strategies=[
        OptimizationStrategy.CLARITY_FOCUS,
        OptimizationStrategy.EFFICIENCY_FOCUS,
        OptimizationStrategy.AUTO,
    ],
    max_iterations=3,      # 每个策略最多迭代3次
    min_confidence=0.7,    # 置信度阈值
    score_threshold=75.0   # 低于75分才优化
)

# 运行优化
patches = service.run_optimization_cycle(
    workflow_id="wf_001",
    config=config
)
```

**内部行为：**
```
提取到3个LLM prompts → [llm_1, llm_2, llm_3]

处理 llm_1:
  分析: score=72 → 需要优化
  尝试策略1 clarity_focus:
    迭代1: score=75, confidence=0.65
    迭代2: score=78, confidence=0.72 ✅ 达到阈值
  尝试策略2 efficiency_focus:
    迭代1: score=76, confidence=0.68
  尝试策略3 auto:
    选择strategy=clarity → 复用策略1结果
  → 选择最佳结果: 策略1的迭代2

处理 llm_2:
  分析: score=85 → 跳过（分数够高）

处理 llm_3:
  分析: score=68 → 需要优化
  （同样的多策略流程...）

返回: [patch_for_llm_1, patch_for_llm_3]
```

### 场景3: 只分析不优化

```python
# 只想看看prompts质量如何
report = service.analyze_workflow("wf_001")

print(f"工作流包含 {report['prompt_count']} 个prompts")
print(f"平均分数: {report['average_score']:.1f}")
print(f"是否需要优化: {report['needs_optimization']}")

for prompt_info in report['prompts']:
    print(f"""
    Prompt: {prompt_info['prompt_id']}
    总分: {prompt_info['overall_score']}
    清晰度: {prompt_info['clarity_score']}
    效率: {prompt_info['efficiency_score']}
    问题数: {prompt_info['issues_count']}
    """)
```

### 场景4: 手动提取和优化单个prompt

```python
# 如果你只想优化特定的prompt
from src.optimizer.models import Prompt

# 手动创建或提取单个prompt
prompt = Prompt(
    id="custom_prompt",
    workflow_id="wf_001",
    node_id="llm_1",
    node_type="llm",
    text="这是一个质量不高的prompt",
    role="user",
    variables=[]
)

# 优化单个prompt
result = service.optimize_single_prompt(
    prompt=prompt,
    strategy="clarity_focus"
)

print(f"原始prompt: {result.original_prompt}")
print(f"优化后: {result.optimized_prompt}")
print(f"改进分数: {result.improvement_score}")
print(f"置信度: {result.confidence}")
```

---

## 📚 API参考

### OptimizerService 主要方法

#### 1. `run_optimization_cycle()`
**完整优化流程（推荐使用）**

```python
def run_optimization_cycle(
    workflow_id: str,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    strategy: Optional[str] = None,  # "auto", "clarity_focus", "efficiency_focus", "structure_focus"
    config: Optional[OptimizationConfig] = None
) -> List[PromptPatch]
```

**功能：**
- ✅ 自动提取所有LLM prompts
- ✅ 逐个分析和优化
- ✅ 生成可应用的patches

**返回：**
- `List[PromptPatch]` - 需要应用到workflow的修改

#### 2. `analyze_workflow()`
**只分析不优化**

```python
def analyze_workflow(
    workflow_id: str
) -> Dict[str, Any]
```

**功能：**
- ✅ 提取所有prompts
- ✅ 分析质量
- ❌ 不进行优化

**返回：**
```python
{
    "workflow_id": "wf_001",
    "prompt_count": 3,
    "average_score": 76.5,
    "needs_optimization": True,
    "prompts": [
        {
            "prompt_id": "wf_001_llm_1",
            "overall_score": 72.0,
            "clarity_score": 68.0,
            "efficiency_score": 76.0,
            "issues_count": 2,
            "issues": [...]
        },
        ...
    ]
}
```

#### 3. `optimize_single_prompt()`
**优化单个prompt**

```python
def optimize_single_prompt(
    prompt: Prompt,
    strategy: str = "auto"
) -> OptimizationResult
```

**功能：**
- ✅ 只优化一个prompt
- ✅ 不访问workflow
- ✅ 返回优化结果

### PromptExtractor 提取方法

#### 1. `extract_from_workflow()`
**提取所有LLM prompts（全量）**

```python
def extract_from_workflow(
    workflow_dict: Dict[str, Any],
    workflow_id: Optional[str] = None
) -> List[Prompt]
```

**行为：**
- 遍历所有节点
- 只提取LLM类型节点
- 返回所有符合条件的prompts

#### 2. `extract_from_node()`
**提取单个节点（按需）**

```python
def extract_from_node(
    node: Dict[str, Any],
    workflow_id: str
) -> Optional[Prompt]
```

**行为：**
- 只处理一个节点
- 非LLM节点返回None
- 返回单个Prompt对象

---

## 🎯 设计理念

### 为什么一次性提取所有prompts？

**优势：**
1. ✅ **批量优化效率高** - 一次分析整个workflow
2. ✅ **上下文完整** - 可以考虑prompts之间的关系
3. ✅ **结果一致** - 所有prompts使用相同的质量标准
4. ✅ **版本管理** - 便于追踪整个workflow的演进

**灵活性：**
- 虽然一次性提取，但可以选择性优化
- `should_optimize()` 会跳过高质量的prompts
- 用户可以通过阈值控制优化范围

### 提取 vs 优化的分离

```
提取阶段（PromptExtractor）:
  → 目标: 找到所有LLM prompts
  → 策略: 全量提取
  → 输出: List[Prompt]

优化阶段（OptimizerService）:
  → 目标: 改进低质量prompts
  → 策略: 选择性优化（基于分数）
  → 输出: List[PromptPatch]（只包含需要修改的）
```

这种设计确保了：
- 提取是全面的（不遗漏）
- 优化是精准的（只改进需要的）

---

## 📊 总结

### 提取功能

| 功能 | 实现方式 | 使用场景 |
|------|---------|----------|
| **全量提取** | `extract_from_workflow()` | ✅ 主流程（推荐） |
| **单节点提取** | `extract_from_node()` | ✅ 手动控制 |
| **过滤非LLM** | 自动跳过 | ✅ 内置逻辑 |

### 运行链路

1. **输入**: workflow_id
2. **提取**: 所有LLM prompts（一次性）
3. **分析**: 逐个评分
4. **筛选**: 只优化低分的
5. **优化**: 多策略+迭代
6. **输出**: PromptPatch列表（只包含需要修改的）

### 实际使用

**最简单的用法：**
```python
patches = service.run_optimization_cycle("wf_001")
# 自动提取 → 自动分析 → 自动优化 → 返回patches
```

**最灵活的用法：**
```python
# 1. 先分析
report = service.analyze_workflow("wf_001")

# 2. 决定是否优化
if report['average_score'] < 80:
    # 3. 自定义配置
    config = OptimizationConfig(strategies=[...])
    patches = service.run_optimization_cycle("wf_001", config=config)
```

---

**生成时间**: 2025-11-18
**适用版本**: Optimizer v1.0 (生产就绪)
