# YAML 配置模块

Dify 自动化测试框架的综合配置管理系统。

## 概述

YAML 配置模块提供了三层配置系统，分离环境设置、工作流定义和测试计划。特性包括：

- **Pydantic V2** 类型安全验证
- **环境变量展开**，使用 `${VAR_NAME}` 语法
- **跨文件引用验证**
- **模块化架构**，易于维护

## 🚀 快速开始

### 使用示例配置文件

本模块提供了完整的示例配置文件，帮助你快速上手：

```bash
# 1. 复制示例配置文件
cp src/config/examples/env_config.example.yaml config/env_config.yaml
cp src/config/examples/workflow_catalog.example.yaml config/workflow_catalog.yaml
cp src/config/examples/test_plan.example.yaml config/test_plan.yaml

# 2. 复制环境变量模板
cp .env.example .env

# 3. 编辑配置文件以匹配你的环境
vim .env  # 设置 DIFY_API_TOKEN 等环境变量
vim config/env_config.yaml
```

📖 **详细说明**：查看 [examples/README.md](./examples/README.md) 获取完整的使用指南和最佳实践。

## 架构

```
src/config/
├── models/          # Pydantic 数据模型
│   ├── env_config.py       # 环境配置
│   ├── workflow_catalog.py # 工作流仓库
│   ├── test_plan.py        # 测试计划定义
│   ├── run_manifest.py     # 测试执行清单
│   └── common.py           # 共享模型 (RateLimit 等)
├── loaders/         # 文件加载和验证
│   ├── config_loader.py    # YAML 文件加载与环境变量展开
│   └── config_validator.py # 跨文件一致性验证
├── utils/           # 工具类
│   ├── exceptions.py       # 自定义异常类
│   └── yaml_parser.py      # YAML 解析工具
└── examples/        # 示例配置文件
    ├── README.md                          # 示例文件使用指南
    ├── env_config.example.yaml            # 环境配置示例
    ├── workflow_catalog.example.yaml      # 工作流目录示例
    └── test_plan.example.yaml             # 测试计划示例
```

## 配置文件

### 1. 环境配置 (`env_config.yaml`)

定义运行时环境设置、API 凭证和 I/O 路径。

**结构：**
```yaml
meta:
  version: "1.0"
  environment: "dev|test|prod"
  updated_by: "<用户名>"

dify:
  base_url: "https://api.dify.ai"  # 必须以 http:// 或 https:// 开头
  tenant_id: "可选的租户ID"
  auth:
    primary_token: "${DIFY_API_TOKEN}"  # 环境变量展开
    fallback_tokens: []
  rate_limits:
    per_minute: 60
    burst: 10

model_evaluator:
  provider: "openai|anthropic|azure"
  model_name: "gpt-4"
  api_key: "${EVALUATOR_API_KEY}"  # 可选，可使用环境变量

io_paths:
  workflow_dsl: "./workflows"
  test_data: "./data"
  output: "./output"
  logs: "./logs"

logging:
  level: "INFO"
  format: "json"

defaults: {}  # 可选的默认值
```

**验证规则：**

- `base_url` 必须以 `http://` 或 `https://` 开头
- `primary_token` 不能为空
- 缺失的路径会触发警告（不是错误）

### 2. 工作流目录 (`workflow_repository.yaml`)

包含元数据和节点结构的可用工作流注册表。

**结构：**

```yaml
meta:
  version: "1.0"
  source: "dify_api"
  last_synced: "2025-01-13T10:00:00Z"

workflows:
  - id: "chatbot_v1"              # 稳定的唯一标识符
    label: "客服机器人"             # 可读的名称
    type: "chatflow"               # workflow | chatflow
    version: "1.2.0"
    dsl_path: "workflows/chatbot_v1.yaml"
    checksum: "sha256:abc123..."
    nodes:
      - node_id: "llm_node_1"
        label: "主 LLM"
        type: "llm"
        path: "/graph/nodes/3"
        prompt_fields: ["/graph/nodes/3/data/prompt"]
      - node_id: "tool_node_1"
        label: "网页搜索"
        type: "tool"
        path: "/graph/nodes/5"
        prompt_fields: []
    resources:
      knowledge_bases: ["kb_001", "kb_002"]
      tools: ["web_search", "calculator"]
    tags: ["客服", "聊天机器人"]
```

**字段约定：**

- `id`: 唯一、稳定的标识符（永不改变）
- `label`: 显示名称（可以改变）
- `type`: `workflow`（无状态）或 `chatflow`（维护对话状态）
- `dsl_path`: 工作流 DSL 文件的相对或绝对路径
- `nodes`: 用于提示词修补的可选节点结构索引
   - `path`: JSON Pointer 格式（如 `/graph/nodes/3`）
   - `prompt_fields`: 可编辑的提示词字段路径列表

### 3. 测试计划 (`test_plan.yaml`)

定义测试执行策略、数据和提示词优化变体。

**结构：**

```yaml
meta:
  plan_id: "regression_2025_01"
  owner: "qa_team"
  description: "季度回归测试"
  created_at: "2025-01-13"

workflows:
  - catalog_id: "chatbot_v1"      # 引用 workflow_repository.yaml
    enabled: true
    weight: 1.0                    # 优先级/采样权重
    dataset_refs: ["normal_cases", "edge_cases"]
    prompt_optimization:           # 可选的提示词变体测试
      - variant_id: "baseline"
        description: "原始提示词"
        weight: 0.5
        nodes:
          - selector:
              by_id: "llm_node_1"
            strategy:
              mode: "replace"
              content: "你是一个有帮助的助手。"
      - variant_id: "enhanced"
        weight: 0.5
        fallback_variant: "baseline"  # 必须引用存在的变体
        nodes:
          - selector:
              by_id: "llm_node_1"
            strategy:
              mode: "template"
              template:
                inline: "你是 {{ role }}。{{ instruction }}"
                variables:
                  role: "专业客服人员"
                  instruction: "请简洁且有帮助地回答。"

test_data:
  datasets:
    - name: "normal_cases"
      scenario: "normal"
      description: "正常路径场景"
      parameters:
        query:
          type: "string"
          values: ["如何重置密码？", "你们的营业时间是？"]
        user_id:
          type: "int"
          range: {min: 1, max: 1000, step: 1}
      conversation_flows:  # 用于 chatflow 测试
        - title: "多轮支持"
          steps:
            - role: "user"
              message: "你好，我需要帮助"
              wait_for_response: true
            - role: "user"
              message: "我无法登录"
      pairwise_dimensions: ["query", "user_id"]  # 用于组合测试
      weight: 1.0
    - name: "edge_cases"
      scenario: "boundary"
      parameters:
        query:
          type: "string"
          values: ["", "x" * 10000]  # 空和超长输入
      weight: 0.5
  strategy:
    pairwise_mode: "PICT"  # PICT | IPO | naive
    sampling_method: "weighted"

execution:
  concurrency: 5                   # 并行执行数
  batch_size: 10
  rate_control:
    per_minute: 100
    burst: 20
  backoff_seconds: 2.0
  retry_policy:
    max_attempts: 3
    backoff_seconds: 2.0
    backoff_multiplier: 1.5
  stop_conditions:
    max_failures: 10
    timeout_minutes: 60

validation:
  output_format_check: true
  min_response_length: 10
  max_response_length: 5000

outputs:
  report_format: "json"
  save_intermediate: true
  artifacts_dir: "./output/artifacts"
```

**核心概念：**

**提示词优化：**

- `variant_id`: 工作流内唯一，用于 A/B 测试
- `fallback_variant`: 如果变体失败，使用此变体（必须存在）
- `selector`: 如何查找要修补的节点
   - `by_id`: 精确节点 ID 匹配
   - `by_label`: 模糊标签匹配
   - `by_type`: 节点类型匹配（如 "llm"）
   - `by_path`: JSON Pointer 路径
- `strategy.mode`:
   - `replace`: 替换整个提示词
   - `prepend`: 添加到现有提示词之前
   - `append`: 添加到现有提示词之后
   - `template`: 使用 Jinja2 模板和变量

**测试数据：**

- `scenario`: `normal` | `boundary` | `error` | `custom`
- `pairwise_dimensions`: 用于成对组合测试的参数
- `conversation_flows`: 用于 chatflow 测试的多轮对话

## 使用示例

### 基础加载

```python
from src.config.loaders import ConfigLoader, ConfigValidator

# 初始化加载器
loader = ConfigLoader()

# 加载配置
env = loader.load_env(Path("env_config.yaml"))
catalog = loader.load_catalog(Path("workflow_repository.yaml"))
plan = loader.load_test_plan(Path("test_plan.yaml"))

# 验证跨文件一致性
validator = ConfigValidator(catalog)
validator.validate_all(env, plan)
```

### 环境变量展开

```python
import os

# 设置环境变量
os.environ["DIFY_API_TOKEN"] = "secret_token_123"
os.environ["OUTPUT_DIR"] = "/var/output"

# 加载配置 - 变量会自动展开
env = loader.load_env(Path("env_config.yaml"))

print(env.dify.auth.primary_token)  # "secret_token_123"
print(env.io_paths["output"])       # Path("/var/output")
```

### 访问测试数据

```python
# 加载测试计划
plan = loader.load_test_plan(Path("test_plan.yaml"))

# 访问数据集
for dataset in plan.datasets:
    print(f"数据集: {dataset.name}")
    print(f"场景: {dataset.scenario}")

# 获取特定数据集
dataset = plan.get_dataset("normal_cases")
if dataset:
    print(f"参数: {dataset.parameters.keys()}")
```

### 验证

```python
from src.config.loaders import ConfigValidator
from src.config.utils.exceptions import ConfigReferenceError, ConfigurationError

try:
    validator = ConfigValidator(catalog)
    validator.validate_plan(plan)
except ConfigReferenceError as e:
    print(f"引用错误: {e}")  # 工作流/数据集未找到
except ConfigurationError as e:
    print(f"配置错误: {e}")  # 无效的配置
```

## 字段验证器

### DifyConfig

- `base_url`: 必须以 `http://` 或 `https://` 开头
- 在 Pydantic 模型层级通过 `@field_validator` 验证

### EnvConfig

- `io_paths`: 自动将字符串路径转换为 `Path` 对象
- 使用 `mode='before'` 进行预验证转换

### PromptStrategy

- `mode`: 必须是 `['replace', 'prepend', 'append', 'template']` 之一
- 对于无效模式会抛出 `ValueError`

## 错误处理

### 异常层次

```
Exception
└── YamlModuleError         # YAML 模块基础错误
    ├── ConfigurationError  # 配置格式/验证错误
    ├── ConfigReferenceError # 跨引用验证失败（重命名避免与内建冲突）
    ├── DSLParseError       # DSL 解析错误
    ├── PatchTargetMissing  # Prompt Patch 目标未找到
    ├── TemplateRenderError # 模板渲染失败
    └── CaseGenerationError # 测试用例生成失败
```

### 常见错误

1. **ConfigurationError**: 无效的 YAML 格式、缺少必填字段
   ```python
   try:
       env = loader.load_env(path)
   except ConfigurationError as e:
       print(f"加载配置失败: {e}")
   ```

2. **ConfigReferenceError**: 引用的工作流/数据集未找到
   ```python
   try:
       validator.validate_plan(plan)
   except ConfigReferenceError as e:
       print(f"无效引用: {e}")
   ```

## 安全考虑

### 密钥管理

**应该做：**

- ✅ 对敏感数据使用环境变量: `"${DIFY_API_TOKEN}"`
- ✅ 将凭证存储在 `.env` 文件中（添加到 `.gitignore`）
- ✅ 使用密钥管理系统（HashiCorp Vault、AWS Secrets Manager）

**不应该做：**

- ❌ 在 YAML 文件中硬编码 API 令牌
- ❌ 将 `.env` 文件提交到版本控制
- ❌ 记录敏感配置值

### 路径遍历防护

- 所有路径都通过 `Path` 对象解析
- 用户输入在 Pydantic 模型层级验证
- 文件系统读取器在加载前检查文件存在性

## 测试

### 测试覆盖率

配置模块当前测试覆盖率：**100%** ✅

```
Name                                     Stmts   Miss  Cover
--------------------------------------------------------------
src\config\__init__.py                       4      0   100%
src\config\loaders\__init__.py               3      0   100%
src\config\loaders\config_loader.py         60      0   100%
src\config\loaders\config_validator.py      46      0   100%
src\config\models\__init__.py                6      0   100%
src\config\models\common.py                 12      0   100%
src\config\models\env_config.py             34      0   100%
src\config\models\run_manifest.py           25      0   100%
src\config\models\test_plan.py             106      0   100%
src\config\models\workflow_catalog.py       35      0   100%
src\config\utils\__init__.py                 0      0   100%
src\config\utils\exceptions.py              16      0   100%
src\config\utils\yaml_parser.py             45      0   100%
--------------------------------------------------------------
TOTAL                                      392      0   100%
```

**质量指标：**

- ✅ 91 个测试，全部通过
- ✅ 测试执行时间：0.56秒
- ✅ 所有边界情况和错误路径均已覆盖
- ✅ 生产就绪状态

### 运行测试

```bash
# 运行所有配置测试
python -m pytest src/test/config/ -v

# 带覆盖率运行
python -m pytest src/test/config/ --cov=src/config --cov-report=term-missing

# 生成HTML覆盖率报告
python -m pytest src/test/config/ --cov=src/config --cov-report=html

# 运行特定测试文件
python -m pytest src/test/config/test_config_loader.py -v

# 运行特定测试类
python -m pytest src/test/config/test_models.py::TestDifyConfigUrlValidation -v
```

### 测试结构

```
src/test/config/
├── test_config_loader.py      # 配置加载器测试（20个测试）
│   ├── TestFileSystemReader (5 tests)
│   ├── TestConfigLoaderEnvExpansion (6 tests)
│   ├── TestConfigLoaderLoadEnv (3 tests)
│   ├── TestConfigLoaderLoadCatalog (3 tests)
│   └── TestConfigLoaderLoadTestPlan (3 tests)
├── test_config_validator.py   # 跨文件验证测试（11个测试）
│   ├── TestConfigValidatorEnv (4 tests)
│   ├── TestConfigValidatorPlan (3 tests)
│   ├── TestConfigValidatorPromptVariants (3 tests)
│   └── TestConfigValidatorAll (1 test)
├── test_models.py             # Pydantic模型测试（18个测试）
│   ├── TestWorkflowEntryProperties (2 tests)
│   ├── TestWorkflowCatalogMethods (4 tests)
│   ├── TestTestPlanMethods (2 tests)
│   ├── TestPromptStrategyValidation (5 tests)
│   ├── TestEnvConfigPathValidation (2 tests)
│   └── TestDifyConfigUrlValidation (3 tests)
├── test_smoke.py              # 基础冒烟测试（7个测试）
│   ├── TestModelsSmokeTest (2 tests)
│   ├── TestLoaderSmokeTest (2 tests)
│   ├── TestExecutorSmokeTest (2 tests)
│   └── TestOptimizerSmokeTest (1 test)
└── test_yaml_parser.py        # YAML解析器测试（35个测试）
    ├── TestYamlParserLoad (6 tests)
    ├── TestYamlParserDump (4 tests)
    ├── TestYamlParserGetNodeByPath (8 tests)
    ├── TestYamlParserSetFieldValue (7 tests)
    ├── TestYamlParserGetFieldValue (7 tests)
    └── TestYamlParserIntegration (3 tests)

总计：91 个测试，全部通过 ✅
```

### 测试覆盖的关键场景

**边界情况测试：**

- ✅ 空文件处理
- ✅ 缺失环境变量
- ✅ 无效YAML语法
- ✅ 不存在的路径
- ✅ 重复ID检测
- ✅ 无效引用检测
- ✅ 深层嵌套结构
- ✅ Unicode字符处理

**验证器测试：**

- ✅ URL格式验证（http/https）
- ✅ 空token检测
- ✅ 路径类型转换
- ✅ 提示词策略模式验证
- ✅ 跨文件引用一致性

**集成测试：**

- ✅ 完整加载-修改-保存流程
- ✅ 复杂DSL文档操作
- ✅ 环境变量展开
- ✅ 多层验证链

## 迁移指南

### 从 Pydantic V1 到 V2

所有验证器已迁移到 Pydantic V2：

**之前（V1）：**

```python
@validator('base_url')
def validate_url(cls, v):
    if not v.startswith(('http://', 'https://')):
        raise ValueError(f"无效 URL: {v}")
    return v
```

**之后（V2）：**

```python
@field_validator('base_url')
@classmethod
def validate_url(cls, value: str) -> str:
    if not value.startswith(('http://', 'https://')):
        raise ValueError(f"无效 URL: {value}")
    return value
```

## 更新日志

### 2025-11-13 v2 - 质量与安全增强

**代码质量改进：**

- ✅ 为所有 18 个模型添加 Pydantic 严格模式（`extra='forbid'`, `validate_assignment=True`）
- ✅ 重命名 `ReferenceError` 为 `ConfigReferenceError`（避免与 Python 内建冲突）
- ✅ 性能优化：`defined_datasets` 验证从 O(n²) 优化到 O(n)
- ✅ 使用 `SecretStr` 保护 API tokens，防止日志泄露

**测试增强：**

- ✅ 新增 `yaml_parser.py` 完整测试套件（35 个测试用例）
- ✅ 测试覆盖率从 87% 提升到 **97%**
- ✅ yaml_parser 覆盖率从 22% 提升到 **100%**
- ✅ 总测试数从 38 个增加到 **73 个**
- ✅ 测试目录统一迁移到项目根目录 `tests/`

**架构改进：**

- ✅ 文档完整中文本地化（529 行）
- ✅ 所有异常类型层次结构完善
- ✅ 添加详细的流程图和技术规范文档

### 2025-11-13 v1 - 重构与质量改进

**架构：**

- ✅ 从 26 个文件减少到 13 个文件（减少 50%）
- ✅ 移除过度设计的 facades、adapters、builders
- ✅ 将组件移至适当的模块（executor、optimizer）

**代码质量：**

- ✅ 修复 pairwise engine 中的缓存截断 bug
- ✅ 修复 fallback variant 验证（现在使用两次遍历）
- ✅ 为所有配置文件添加环境变量展开
- ✅ 移除未使用的 `_schema_cache` 属性
- ✅ 迁移所有验证器到 Pydantic V2
- ✅ 使用 `TestDataConfig` 模型加强 `TestPlan.test_data` 类型

**测试：**

- ✅ 测试覆盖率从 0% 提升到 87%
- ✅ 添加 31 个单元测试（包括冒烟测试共 38 个）
- ✅ 所有测试通过，无 Pydantic 弃用警告

## 故障排除

### 问题：环境变量未展开

**症状：** `${VAR_NAME}` 在配置中显示为字面值

**解决方案：** 确保在加载配置前设置变量

```bash
export DIFY_API_TOKEN="your_token"
python your_script.py
```

### 问题：有效工作流的 ConfigReferenceError

**症状：** "工作流 'X' 在目录中未找到"，但它确实存在

**解决方案：** 检查 `catalog_id` 精确匹配（区分大小写）

```yaml
# workflow_repository.yaml
workflows:
  - id: "chatbot_v1"  # 必须完全匹配

# test_plan.yaml
workflows:
  - catalog_id: "chatbot_v1"  # ✅ 正确
  - catalog_id: "Chatbot_v1"  # ❌ 错误（大小写不匹配）
```

### 问题：Pydantic 验证错误

**症状：** `ValidationError: 1 validation error for EnvConfig`

**解决方案：** 检查必填字段和数据类型

```python
# 使用 ValidationError.errors() 查看详情
try:
    env = loader.load_env(path)
except ValidationError as e:
    for error in e.errors():
        print(f"字段: {error['loc']}, 错误: {error['msg']}")
```

## 贡献

### 添加新模型

1. 在 `src/config/models/` 下的适当文件中创建模型
2. 添加到 `__init__.py` 导出
3. 在 `tests/config/` 中编写单元测试
4. 更新此 README 的字段约定

### 代码风格

- 使用 Pydantic V2 `@field_validator`（不是 V1 `@validator`）
- 为所有参数和返回值包含类型提示
- 为所有类和公共方法添加文档字符串
- 遵循现有命名约定

## 许可证

[项目许可信息]

## 支持

如有问题或疑问：

- 在项目仓库中提交 issue
- 联系：[团队/维护者联系方式]
