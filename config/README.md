# 环境配置文件目录

此目录包含环境相关的配置文件（YAML格式），遵循[12-Factor App](https://12factor.net/config)原则，将配置与代码分离。

## 📁 配置文件列表

### 1. llm.yaml - LLM API配置
**用途**: Optimizer模块的提示词优化LLM配置
**模块**: src/optimizer
**配置代码**: src/optimizer/config/llm_config.py

**字段**:
- `provider`: LLM提供商 (openai, anthropic, local, stub)
- `model`: 模型名称 (如 gpt-4-turbo-preview)
- `api_key_env`: API密钥环境变量名
- `temperature`: 温度参数 (0.0-2.0)
- `max_tokens`: 最大token数
- `enable_cache`: 是否启用缓存
- `cache_ttl`: 缓存过期时间（秒）
- `cost_limits`: 成本限制配置

**示例**: 参考 llm.yaml.example

### 2. logging_config.yaml - 日志系统配置
**用途**: 系统级日志配置
**加载**: 应用启动时加载

## 🗂️ 配置文件 vs 配置代码

### 配置文件 (此目录)
- **位置**: `config/` (项目根目录)
- **格式**: YAML
- **内容**: 环境特定参数、API密钥、路径等
- **版本控制**: `.example`文件提交，实际配置文件在.gitignore中

### 配置代码
- **系统级配置**: `src/config/` - 环境配置、工作流目录、测试计划
- **模块级配置**: `src/<module>/config/` - 模块专用配置逻辑
  - `src/optimizer/config/` - Optimizer专用LLM配置

## 🔄 LLM配置说明

项目中有两处LLM相关配置：

### ModelEvaluator (src/config/models/common.py)
- **用途**: Executor模块的测试结果自动评分
- **配置文件**: config/env_config.yaml
- **字段**: provider, model_name, temperature, max_tokens (简单配置)
- **状态**: 🟡 占位符 - executor尚未完全实现LLM评估功能

### LLMConfig (src/optimizer/config/llm_config.py)
- **用途**: Optimizer模块的提示词优化
- **配置文件**: config/llm.yaml
- **字段**: 11个字段，包含缓存、成本控制、重试等
- **状态**: ✅ 已实现 - 生产环境使用

### 为什么分离？
1. **职责不同**: 简单评分 vs 复杂优化
2. **复杂度不同**: ModelEvaluator简单（4字段），LLMConfig复杂（11字段）
3. **演进路径**: Optimizer需要更多功能，Evaluator可能保持简单

### 未来计划
如果executor真正实现完整的LLM评估功能，考虑提取共同基类：
- 创建 `src/config/models/llm_base_config.py`
- ModelEvaluator和LLMConfig都继承LLMBaseConfig
- 消除核心字段重复

## 🔐 安全最佳实践

### API密钥管理
1. **永不硬编码**: API密钥不应出现在YAML文件中
2. **环境变量**: 使用 `api_key_env: OPENAI_API_KEY` 引用环境变量
3. **`.env`文件**: 本地开发使用 `.env` 文件（已在.gitignore中）
4. **`.example`文件**: 提交 `.example` 模板文件，不包含真实密钥

### 文件权限
```bash
# 确保配置文件不被提交
echo "config/llm.yaml" >> .gitignore
echo "config/env_config.yaml" >> .gitignore

# 复制示例文件
cp config/llm.yaml.example config/llm.yaml

# 编辑并填入真实配置
vim config/llm.yaml
```

## 📚 相关文档

- **系统级配置**: src/config/README.md
- **Optimizer配置**: src/optimizer/config/README.md
- **LLM集成架构**: LLM_INTEGRATION_ARCHITECTURE.md

## 🛠️ 快速开始

```bash
# 1. 复制示例配置
cp config/llm.yaml.example config/llm.yaml

# 2. 设置环境变量
export OPENAI_API_KEY="sk-your-key-here"

# 3. 编辑配置文件
vim config/llm.yaml

# 4. 验证配置
python -c "from src.optimizer.config import LLMConfigLoader; print(LLMConfigLoader.from_yaml('config/llm.yaml'))"
```

## 📝 遵循的原则

### 12-Factor App 配置原则
1. ✅ 配置与代码严格分离
2. ✅ 环境变量优先
3. ✅ 易于在不同环境间切换
4. ✅ 不在版本控制中存储敏感信息

### 项目配置模式
- **YAML文件** (config/) - 环境特定配置
- **Python代码** (src/config/, src/<module>/config/) - 配置模型和加载逻辑
- **清晰职责** - 每个配置文件有明确的用途和所属模块
