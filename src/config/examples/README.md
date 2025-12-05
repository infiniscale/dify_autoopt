# YAML 配置示例文件

本目录包含 YAML 配置模块的示例配置文件，帮助你快速开始使用。

## 📁 文件列表

| 文件 | 说明 | 对应模型 |
|------|------|---------|
| `env_config.example.yaml` | 环境配置示例 | `EnvConfig` |
| `workflow_catalog.example.yaml` | 工作流目录示例 | `WorkflowCatalog` |
| `test_plan.example.yaml` | 测试计划示例 | `TestPlan` |

## 🚀 快速开始

### 1. 创建你的配置文件

```bash
# 进入项目根目录
cd /path/to/dify_autoopt

# 创建配置目录（如果不存在）
mkdir -p config

# 复制示例文件
cp src/config/examples/env_config.example.yaml config/env_config.yaml
cp src/config/examples/workflow_catalog.example.yaml config/workflow_catalog.yaml
cp src/config/examples/test_plan.example.yaml config/test_plan.yaml

# 复制环境变量文件
cp .env.example .env
```

### 2. 配置环境变量

编辑 `.env` 文件，填写你的 API 凭证：

```bash
# 编辑 .env
vim .env  # 或使用你喜欢的编辑器

# 必填项
DIFY_API_TOKEN=your_actual_token_here
EVALUATOR_API_KEY=your_openai_key_here
```

### 3. 修改配置文件

#### 3.1 环境配置 (`config/env_config.yaml`)

```yaml
# 修改这些必填项
dify:
  base_url: "https://your-dify-instance.com"  # 你的 Dify 实例地址
  auth:
    primary_token: "${DIFY_API_TOKEN}"        # 使用环境变量

model_evaluator:
  provider: "openai"                           # 或 anthropic、azure
  model_name: "gpt-4"
  api_key: "${EVALUATOR_API_KEY}"
```

#### 3.2 工作流目录 (`config/workflow_catalog.yaml`)

```yaml
# 添加你的工作流
workflows:
  - id: "my_workflow"                    # 唯一 ID
    label: "我的工作流"
    type: "chatflow"                     # 或 workflow
    dsl_path: "workflows/my_flow.yaml"   # DSL 文件路径
    nodes:
      - node_id: "llm_1"
        type: "llm"
        path: "/graph/nodes/0"
        prompt_fields:
          - "data.prompt_template"
```

#### 3.3 测试计划 (`config/test_plan.yaml`)

```yaml
# 指定要测试的工作流
workflows:
  - catalog_id: "my_workflow"            # 引用 catalog 中的 id
    enabled: true
    dataset_refs:
      - "test_dataset_1"

# 定义测试数据
test_data:
  datasets:
    - name: "test_dataset_1"
      parameters:
        query:
          type: "string"
          values: ["测试问题1", "测试问题2"]
```

## 📖 配置文件详解

### 环境配置 (`env_config.yaml`)

**核心字段**：
- `dify.base_url`: Dify API 地址（必填）
- `dify.auth.primary_token`: API Token（必填，建议用环境变量）
- `model_evaluator`: 评估模型配置
- `io_paths`: I/O 路径配置
- `logging`: 日志配置

**环境变量展开**：
```yaml
# 使用 ${VAR_NAME} 语法引用环境变量
primary_token: "${DIFY_API_TOKEN}"
api_key: "${EVALUATOR_API_KEY}"
output: "${OUTPUT_DIR}"  # 如果环境变量未设置，会保持原样
```

### 工作流目录 (`workflow_catalog.yaml`)

**核心字段**：
- `workflows[].id`: 唯一标识符（不要改变）
- `workflows[].dsl_path`: DSL 文件路径
- `workflows[].nodes`: 节点索引（用于 Prompt Patch）

**节点索引**：
```yaml
nodes:
  - node_id: "llm_main"              # 节点唯一 ID
    label: "主对话模型"              # 人类可读名称
    type: "llm"                      # 节点类型
    path: "/graph/nodes/0"           # JSON Pointer 路径
    prompt_fields:                   # Prompt 字段列表
      - "data.prompt_template"
      - "data.system_prompt"
```

**节点类型**：
- `llm`: LLM 大语言模型
- `knowledge-retrieval`: 知识库检索
- `code`: 代码执行
- `http-request`: HTTP 请求
- `if-else`: 条件判断
- `tool`: 工具调用
- `start` / `end`: 开始/结束节点

### 测试计划 (`test_plan.yaml`)

**核心字段**：
- `workflows`: 要测试的工作流列表
- `test_data.datasets`: 测试数据集
- `execution`: 执行策略
- `validation`: 验证规则

**Prompt 优化变体**：
```yaml
prompt_optimization:
  - variant_id: "baseline"           # 变体唯一 ID
    weight: 0.5                      # 权重（0-1）
    fallback_variant: null           # 失败时回退到哪个变体
    nodes:                           # Prompt 修改列表
      - selector:
          by_id: "llm_main"          # 按 ID 选择节点
        strategy:
          mode: "replace"            # replace | prepend | append | template
          content: "新的 Prompt"
```

**测试数据集**：
```yaml
datasets:
  - name: "my_dataset"               # 数据集名称
    scenario: "normal"               # normal | boundary | error | custom

    # 方式 1: 参数化输入（用于 Workflow）
    parameters:
      query:
        type: "string"
        values: ["问题1", "问题2"]
      user_id:
        type: "int"
        range: {min: 1, max: 100}

    # 方式 2: 对话流（用于 Chatflow）
    conversation_flows:
      - title: "多轮对话"
        steps:
          - role: "user"
            message: "你好"
            wait_for_response: true
```

## 🎯 最佳实践

### 1. 安全管理

✅ **应该做**：
```yaml
# 使用环境变量存储敏感信息
primary_token: "${DIFY_API_TOKEN}"
```

❌ **不应该做**：
```yaml
# 不要硬编码 API Token
primary_token: "sk-xxxxxxxxxxxxxx"  # 危险！
```

### 2. 文件组织

```
project/
├── .env                          # 环境变量（不要提交）
├── .env.example                  # 环境变量模板
└── config/                       # 配置目录
    ├── env_config.yaml           # 环境配置（不要提交）
    ├── workflow_catalog.yaml     # 工作流目录（可提交）
    └── test_plan.yaml            # 测试计划（可提交）
```

### 3. 版本控制

在 `.gitignore` 中添加：
```gitignore
# 敏感配置文件
.env
config/env_config.yaml

# 临时文件
*.log
output/
logs/
```

### 4. 跨环境配置

```bash
# 开发环境
config/
├── env_config.dev.yaml
├── workflow_catalog.yaml
└── test_plan.dev.yaml

# 测试环境
config/
├── env_config.test.yaml
├── workflow_catalog.yaml
└── test_plan.test.yaml

# 生产环境
config/
├── env_config.prod.yaml
├── workflow_catalog.yaml
└── test_plan.prod.yaml
```

加载时指定环境：
```python
env = loader.load_env(Path(f"config/env_config.{env_name}.yaml"))
```

## 🔍 验证配置

### 使用 Python 验证

```python
from pathlib import Path
from src.config.loaders import ConfigLoader, ConfigValidator

# 初始化加载器
loader = ConfigLoader()

# 加载配置
env = loader.load_env(Path("config/env_config.yaml"))
catalog = loader.load_catalog(Path("config/workflow_catalog.yaml"))
plan = loader.load_test_plan(Path("config/test_plan.yaml"))

# 验证配置
validator = ConfigValidator(catalog)
validator.validate_all(env, plan)

print("✅ 配置验证通过！")
```

### 常见错误

**错误 1**：环境变量未设置
```
ConfigurationError: Primary token cannot be empty
```
**解决**：设置 `DIFY_API_TOKEN` 环境变量

**错误 2**：引用不存在的 workflow
```
ConfigReferenceError: Workflow 'xxx' not found in catalog
```
**解决**：检查 `test_plan.yaml` 中的 `catalog_id` 是否匹配

**错误 3**：无效的 URL 格式
```
ValidationError: Invalid URL format
```
**解决**：确保 URL 以 `http://` 或 `https://` 开头

## 📚 相关文档

- [YAML 配置模块 README](../README.md)
- [技术规范文档](../YAML_Module_Full%20version_Technical_Specification.md)
- [流程图文档](../YAML_Module_Full%20version_Flowcharts.md)

## 💡 获取帮助

如有问题，请：
1. 查看 [src/config/README.md](../README.md) 的故障排除章节
2. 检查配置文件的注释和示例
3. 在项目仓库提交 Issue
