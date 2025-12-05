# Optimizer配置模块

此目录包含Optimizer模块的LLM配置Python代码。

## 📁 文件结构

```
src/optimizer/config/
├── __init__.py              # 导出LLMConfig, LLMProvider, LLMConfigLoader
├── llm_config.py            # LLM配置Pydantic模型
└── llm_config_loader.py     # 配置加载器（支持YAML、环境变量、字典）
```

## 🗂️ 配置文件位置

配置文件位于项目根目录: **`config/llm.yaml`**

遵循12-Factor App原则，配置文件与代码分离。

## 🔧 核心组件

### LLMConfig (llm_config.py)
Pydantic模型，定义LLM配置结构：
- 支持4种provider: OPENAI, ANTHROPIC, LOCAL, STUB
- 11个配置字段（provider, model, temperature, max_tokens, api_key_env等）
- 内置参数验证（temperature范围、max_tokens限制等）
- API密钥安全（使用SecretStr，从环境变量读取）

### LLMConfigLoader (llm_config_loader.py)
配置加载器，支持多种加载方式：
- `from_yaml(path)` - 从YAML文件加载
- `from_env()` - 从环境变量加载
- `from_dict(dict)` - 从字典加载
- `default()` - 返回默认STUB配置
- `auto_load()` - 自动降级加载（env vars → yaml → default）

## 💡 使用示例

### 从YAML文件加载
```python
from src.optimizer.config import LLMConfigLoader

config = LLMConfigLoader.from_yaml("config/llm.yaml")
print(f"Using {config.provider} with model {config.model}")
```

### 从环境变量加载
```python
import os
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = "gpt-4-turbo-preview"
os.environ["OPENAI_API_KEY"] = "sk-..."

config = LLMConfigLoader.from_env()
```

### 自动加载（推荐）
```python
# 自动尝试：环境变量 → config/llm.yaml → 默认STUB
config = LLMConfigLoader.auto_load()
```

## 🔄 与ModelEvaluator的关系

### 核心字段重复
LLMConfig与`src/config/models/common.py`中的ModelEvaluator有核心字段重复：
- `provider`, `model`, `temperature`, `max_tokens`

### 为什么不合并？
1. **职责不同**:
   - ModelEvaluator: 测试评分（简单LLM调用）
   - LLMConfig: 提示词优化（需要缓存、成本控制）

2. **复杂度不同**:
   - ModelEvaluator: 4字段
   - LLMConfig: 11字段

3. **实现状态**:
   - ModelEvaluator: 🟡 占位符（executor未完全实现）
   - LLMConfig: ✅ 生产就绪

### 未来计划
TODO: 如果executor真正实现LLM评估，考虑提取共同基类
- 创建 `src/config/models/llm_base_config.py`
- ModelEvaluator和LLMConfig都继承基类
- 参见: config/README.md "LLM配置说明"

## 📚 相关文档

- **配置文件位置**: config/README.md
- **LLM集成架构**: LLM_INTEGRATION_ARCHITECTURE.md
- **Optimizer使用指南**: src/optimizer/README.md

## 🔐 安全提示

- ✅ 使��� `api_key_env` 引用环境变量（不要硬编码API密钥）
- ✅ 配置文件在.gitignore中（只提交.example文件）
- ✅ 显示配置时API密钥自动掩码（get_display_config()）
