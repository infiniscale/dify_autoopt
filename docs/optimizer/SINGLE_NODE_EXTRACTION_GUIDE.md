# 单节点提取操作指南

## 📋 目录
1. [基本用法](#基本用法)
2. [从Workflow中提取特定节点](#从workflow中提取特定节点)
3. [手动构造节点提取](#手动���造节点提取)
4. [完整示例](#完整示例)
5. [常见场景](#常见场景)

---

## 🔧 基本用法

### 方法签名

```python
def extract_from_node(
    self,
    node: Dict[str, Any],      # 节点字典
    workflow_id: str           # 所属workflow ID
) -> Optional[Prompt]:         # 返回Prompt对象或None
```

**参数说明：**
- `node`: 从workflow DSL中提取的单个节点字典
- `workflow_id`: 该节点所属的workflow标识符

**返回值：**
- 如果是LLM节点：返回`Prompt`对象
- 如果不是LLM节点：返回`None`

---

## 📦 从Workflow中提取特定节点

### 场景1: 提取指定ID的节点

```python
from src.optimizer import PromptExtractor
import yaml

# 1. 加载workflow DSL
with open("workflow.yaml", "r", encoding="utf-8") as f:
    workflow_dsl = yaml.safe_load(f)

# 2. 创建提取器
extractor = PromptExtractor()

# 3. 找到所有节点
nodes = workflow_dsl["graph"]["nodes"]

# 4. 根据node_id找到特定节点
target_node_id = "llm_1"
target_node = None

for node in nodes:
    if node.get("id") == target_node_id:
        target_node = node
        break

# 5. 提取这个节点的prompt
if target_node:
    prompt = extractor.extract_from_node(target_node, "wf_001")

    if prompt:
        print(f"成功提取prompt:")
        print(f"  ID: {prompt.id}")
        print(f"  文本长度: {len(prompt.text)}字符")
        print(f"  变量数: {len(prompt.variables)}")
        print(f"  内容预览: {prompt.text[:100]}...")
    else:
        print(f"节点 {target_node_id} 不是LLM节点或提取失败")
else:
    print(f"未找到节点 {target_node_id}")
```

### 场景2: 提取第N个LLM节点

```python
# 只提取第一个LLM节点
def extract_first_llm_node(workflow_dsl, workflow_id):
    extractor = PromptExtractor()
    nodes = workflow_dsl["graph"]["nodes"]

    for node in nodes:
        # 尝试提取，如果是LLM节点会返回Prompt对象
        prompt = extractor.extract_from_node(node, workflow_id)
        if prompt:
            return prompt  # 返回第一个找到的LLM prompt

    return None  # 没找到任何LLM节点

# 使用
first_llm_prompt = extract_first_llm_node(workflow_dsl, "wf_001")
if first_llm_prompt:
    print(f"第一个LLM节点: {first_llm_prompt.node_id}")
```

### 场景3: 按条件筛选节点

```python
# 提取所有包含特定变量的LLM节点
def extract_nodes_with_variable(workflow_dsl, workflow_id, variable_name):
    extractor = PromptExtractor()
    nodes = workflow_dsl["graph"]["nodes"]
    prompts_with_var = []

    for node in nodes:
        prompt = extractor.extract_from_node(node, workflow_id)
        if prompt and variable_name in prompt.variables:
            prompts_with_var.append(prompt)

    return prompts_with_var

# 使用：找出所有使用了 {{user_input}} 变量的prompts
prompts = extract_nodes_with_variable(
    workflow_dsl,
    "wf_001",
    "user_input"
)
print(f"找到 {len(prompts)} 个使用 user_input 变量的prompts")
```

---

## 🎨 手动构造节点提取

### 场景4: 直接传入节点数据

如果你已经有了节点的数据结构，可以直接提取：

```python
from src.optimizer import PromptExtractor

extractor = PromptExtractor()

# 手动构造一个节点字典（符合Dify DSL格式）
node = {
    "id": "llm_1",
    "data": {
        "type": "llm",
        "title": "LLM节点",
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "prompt_template": [
            {
                "role": "system",
                "text": "You are a helpful assistant specializing in {{domain}}."
            },
            {
                "role": "user",
                "text": "Please help me with: {{user_input}}"
            }
        ],
        "temperature": 0.7,
        "max_tokens": 2000
    }
}

# 提取prompt
prompt = extractor.extract_from_node(node, "custom_workflow")

if prompt:
    print(f"提取成功！")
    print(f"Prompt ID: {prompt.id}")
    print(f"文本: {prompt.text}")
    print(f"变量: {prompt.variables}")  # ['domain', 'user_input']
    print(f"角色: {prompt.role}")       # 'system'
```

---

## 💼 完整示例

### 示例1: 交互式选择要优化的���点

```python
from src.optimizer import PromptExtractor, OptimizerService
import yaml

def interactive_node_optimization():
    """交互式选择并优化特定节点"""

    # 1. 加载workflow
    with open("workflow.yaml", "r") as f:
        workflow_dsl = yaml.safe_load(f)

    workflow_id = workflow_dsl.get("id", "unknown")
    nodes = workflow_dsl["graph"]["nodes"]

    # 2. 找出所有LLM节点
    extractor = PromptExtractor()
    llm_prompts = []

    print("正在扫描LLM节点...")
    for idx, node in enumerate(nodes):
        prompt = extractor.extract_from_node(node, workflow_id)
        if prompt:
            llm_prompts.append((idx, node, prompt))

    # 3. 显示可选的LLM节点
    print(f"\n找到 {len(llm_prompts)} 个LLM节点：\n")
    for i, (idx, node, prompt) in enumerate(llm_prompts):
        print(f"{i+1}. {prompt.node_id}")
        print(f"   文本预览: {prompt.text[:80]}...")
        print(f"   变量数: {len(prompt.variables)}")
        print()

    # 4. 让用户选择
    choice = int(input("请选择要优化的节点编号 (1-{}): ".format(len(llm_prompts))))
    selected_idx, selected_node, selected_prompt = llm_prompts[choice - 1]

    # 5. 优化这个节点
    print(f"\n正在优化节点 {selected_prompt.node_id}...")
    service = OptimizerService()

    result = service.optimize_single_prompt(
        prompt=selected_prompt,
        strategy="auto"
    )

    # 6. 显示结果
    print("\n优化完���！")
    print(f"原始prompt:\n{result.original_prompt}\n")
    print(f"优化后:\n{result.optimized_prompt}\n")
    print(f"改进分数: {result.improvement_score:.1f}")
    print(f"置信度: {result.confidence:.2%}")
    print(f"变更说明:")
    for change in result.changes:
        print(f"  - {change.description}")

# 运行
interactive_node_optimization()
```

**运行示例：**
```
正在扫描LLM节点...

找到 3 个LLM节点：

1. llm_1
   文本预览: You are a customer service assistant. Help users with their inquiries...
   变量数: 2

2. llm_2
   文本预览: Analyze the sentiment of the following text and classify it as positive, ne...
   变量数: 1

3. llm_3
   文本预览: Generate a summary of: {{document_text}}
   变量数: 1

请选择要优化的节点编号 (1-3): 1

正在优化节点 llm_1...

优化完成！
原始prompt:
You are a customer service assistant. Help users with their inquiries.

优化后:
You are a professional customer service assistant specializing in providing clear,
helpful responses. Your role is to assist users with {{inquiry_type}} inquiries,
ensuring accuracy and empathy in every interaction.

改进分数: 12.5
置信度: 85.00%
变更说明:
  - Added role clarification and specialization
  - Integrated variable {{inquiry_type}} for context
  - Enhanced professionalism and tone
```

### 示例2: 批量处理但分别优化

```python
def optimize_nodes_separately(workflow_dsl, workflow_id):
    """
    提取所有LLM节点，但分别优化每个节点
    （与全量优化不同，这里可以为每个节点使用不同的策略）
    """
    extractor = PromptExtractor()
    service = OptimizerService()

    nodes = workflow_dsl["graph"]["nodes"]
    results = []

    for node in nodes:
        # 单独提取每个节点
        prompt = extractor.extract_from_node(node, workflow_id)

        if not prompt:
            continue

        # 根据节点特征选择不同策略
        if "sentiment" in prompt.text.lower():
            strategy = "clarity_focus"  # 情感分析需要清晰
        elif "summary" in prompt.text.lower():
            strategy = "efficiency_focus"  # 摘要需要简洁
        else:
            strategy = "auto"  # 其他自动选择

        print(f"优化 {prompt.node_id} 使用策略: {strategy}")

        # 优化
        result = service.optimize_single_prompt(prompt, strategy)
        results.append({
            "node_id": prompt.node_id,
            "strategy": strategy,
            "result": result
        })

    return results

# 使用
results = optimize_nodes_separately(workflow_dsl, "wf_001")

for item in results:
    print(f"\n节点: {item['node_id']}")
    print(f"策略: {item['strategy']}")
    print(f"改进: {item['result'].improvement_score:.1f}分")
```

---

## ���� 常见场景

### 场景A: 只优化关键节点

```python
# 定义关键节点列表
CRITICAL_NODES = ["llm_main", "llm_classifier", "llm_summarizer"]

def optimize_critical_nodes_only(workflow_dsl, workflow_id):
    extractor = PromptExtractor()
    service = OptimizerService()

    nodes = workflow_dsl["graph"]["nodes"]
    patches = []

    for node in nodes:
        node_id = node.get("id")

        # 只处理关键节点
        if node_id not in CRITICAL_NODES:
            continue

        prompt = extractor.extract_from_node(node, workflow_id)
        if prompt:
            result = service.optimize_single_prompt(prompt)
            # 生成patch...
            patches.append(result)

    return patches
```

### 场景B: 根据节点位置提取

```python
# 只优化workflow开头和结尾的节点
def optimize_boundary_nodes(workflow_dsl, workflow_id):
    extractor = PromptExtractor()
    nodes = workflow_dsl["graph"]["nodes"]

    # 提取第一个和最后一个LLM节点
    first_llm = None
    last_llm = None

    for node in nodes:
        prompt = extractor.extract_from_node(node, workflow_id)
        if prompt:
            if first_llm is None:
                first_llm = prompt
            last_llm = prompt  # 不断更新，最后一个就是末尾

    return first_llm, last_llm
```

### 场景C: 按质量分数提取

```python
def extract_low_quality_nodes(workflow_dsl, workflow_id, threshold=70):
    """只提取低质量的节点"""
    extractor = PromptExtractor()
    analyzer = PromptAnalyzer()

    nodes = workflow_dsl["graph"]["nodes"]
    low_quality_prompts = []

    for node in nodes:
        prompt = extractor.extract_from_node(node, workflow_id)
        if prompt:
            # 分析质量
            analysis = analyzer.analyze_prompt(prompt)

            if analysis.overall_score < threshold:
                low_quality_prompts.append({
                    "prompt": prompt,
                    "score": analysis.overall_score,
                    "issues": analysis.issues
                })

    # 按分数排序（最差的在前）
    low_quality_prompts.sort(key=lambda x: x["score"])

    return low_quality_prompts

# 使用
low_quality = extract_low_quality_nodes(workflow_dsl, "wf_001", threshold=75)

print(f"发现 {len(low_quality)} 个低质量节点：")
for item in low_quality:
    print(f"  {item['prompt'].node_id}: {item['score']:.1f}分")
    print(f"    问题: {[i.description for i in item['issues'][:3]]}")
```

---

## 📊 节点数据结构参考

### Dify节点的典型结构

```python
# LLM节点示例
{
    "id": "llm_1",
    "data": {
        "type": "llm",
        "title": "LLM节点名称",
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "mode": "chat",
            "completion_params": {
                "temperature": 0.7,
                "max_tokens": 2000
            }
        },
        "prompt_template": [
            {
                "role": "system",
                "text": "You are a {{role}}."
            },
            {
                "role": "user",
                "text": "{{user_input}}"
            }
        ]
    }
}

# Question Classifier节点示例
{
    "id": "classifier_1",
    "data": {
        "type": "question-classifier",
        "title": "问题分类器",
        "classes": [...],
        "query_variable_selector": ["sys", "query"],
        "model": {...}
    }
}

# 条件节点示例（带system_prompt）
{
    "id": "ifelse_1",
    "data": {
        "type": "if-else",
        "title": "条件判断",
        "system_prompt": "Evaluate if {{condition}} is met",
        "cases": [...]
    }
}
```

---

## ⚙️ API参考

### extract_from_node() 完整参数

```python
def extract_from_node(
    node: Dict[str, Any],
    workflow_id: str
) -> Optional[Prompt]:
    """
    Args:
        node: 节点字典，包含:
            - id: 节点ID (必需)
            - data: 节点数据 (必需)
                - type: 节点类型 (必需)
                - prompt_template: prompt模板 (LLM节点必需)
                - 其他配置...

        workflow_id: workflow标识符

    Returns:
        Prompt对象（如果是LLM节点）
        None（如果不是LLM节点或提取失败）

    Raises:
        无（所有异常都被捕获并返回None）
    """
```

### 提取后的Prompt对象

```python
Prompt(
    id="wf_001_llm_1",           # workflow_id + node_id
    workflow_id="wf_001",         # 所属workflow
    node_id="llm_1",              # 节点ID
    node_type="llm",              # 节点类型
    text="You are...",            # 完整prompt文本
    role="system",                # 角色（system/user/assistant）
    variables=["var1", "var2"],   # 提取的变量列表
    context={                     # 上下文信息
        "model": "gpt-4",
        "temperature": 0.7
    },
    extracted_at=datetime.now()   # 提取时间
)
```

---

## 🎯 总结

### 单节点提取的三种方式

| 方式 | 使用场景 | 代码示例 |
|------|---------|---------|
| **从workflow中提取** | 已有workflow DSL | `extractor.extract_from_node(nodes[0], "wf_001")` |
| **手动构造** | 自定义节点数据 | `extractor.extract_from_node(custom_node, "wf_001")` |
| **条件筛选** | 提取符合条件的 | 遍历nodes并按条件filter |

### 核心要点

1. ✅ `extract_from_node()` 只处理**单个节点**
2. ✅ 非LLM节点返回 `None`（自动跳过）
3. ✅ 需要手动遍历nodes数组来提取多个
4. ✅ 可以结合条件实现灵活的提取策略
5. ✅ 返回的Prompt对象可以直接用于优化

### 与全量提取的对比

```python
# 全量提取（自动遍历）
prompts = extractor.extract_from_workflow(workflow_dsl, "wf_001")
# 内部自动遍历所有节点，返回所有LLM prompts

# 单节点提取（手动控制）
for node in workflow_dsl["graph"]["nodes"]:
    prompt = extractor.extract_from_node(node, "wf_001")
    if prompt:
        # 处理这个prompt
        ...
```

---

**生成时间**: 2025-11-18
**适用版本**: Optimizer v1.0
