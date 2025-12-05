# 配置管理模块

## 功能概述

负责项目的配置文件管理，支持YAML格式配置文件的加载、验证和生效，提供灵活的配置管理机制。

## 模块组成

### 1. YAML配置加载 (yaml_loader.py)
- YAML文件解析
- 环境变量替换
- 配置文件合并
- 热加载支持

### 2. 配置验证 (validator.py)
- 配置格式验证
- 必填字段检查
- 数据类型验证
- 自定义验证规则

## 功能特性

- 📝 YAML格式支持
- 🔧 环境变量注入
- ✅ 配置自动验证
- 🔄 热加载更新
- 🏗️ 分层配置管理
- 🛡️ 输入安全验证

## 使用示例

```python
# 配置加载
from src.config import ConfigLoader

loader = ConfigLoader()
config = loader.load_config("config/config.yaml")

# 访问配置参数
dify_url = config["dify"]["base_url"]
timeout = config["execution"]["timeout"]

# 环境变量支持
# config.yaml中的配置
# dify:
#   base_url: ${DIFY_BASE_URL}
#   api_key: ${DIFY_API_KEY}

# 配置验证
from src.config import ConfigValidator

validator = ConfigValidator()
is_valid = validator.validate(config)
```

## 配置文件结构

### 主配置文件 (config.yaml)
```yaml
# Dify平台配置
dify:
  base_url: "https://your-dify-instance.com"
  api_base: "https://your-dify-instance.com/v1"

# 认证配置
auth:
  username: "your_username"
  password: "your_password"
  api_key: "your_api_key"

# 工作流配置
workflows:
  - name: "test_workflow_1"
    inputs:
      file_list: ["path/to/file1", "path/to/file2"]
      num_list: [1, 2, 3]
      string_list: ["text1", "text2"]

# 优化配置
optimization:
  llm_model: "gpt-4"
  max_iterations: 5
  optimization_strategy: "gradient_descent"

# 执行配置
execution:
  concurrency: 5
  timeout: 300
  retry_count: 3

# 日志配置
logging:
  level: "INFO"
  file: "logs/app.log"
  max_size: "10MB"
  backup_count: 5
```

### 环境配置 (.env)
```bash
# Dify平台配置
DIFY_BASE_URL=https://your-dify-instance.com
DIFY_API_KEY=your_api_key

# 认证配置
DIFY_USERNAME=your_username
DIFY_PASSWORD=your_password

# 数据库配置
DATABASE_URL=sqlite:///data/app.db
REDIS_URL=redis://localhost:6379

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## 配置验证规则

### 必填字段检查
```python
required_fields = {
    "dify": ["base_url", "api_base"],
    "auth": ["username", "password"],
    "execution": ["concurrency", "timeout"]
}
```

### 数据类型验证
```python
type_mapping = {
    "dify.base_url": str,
    "auth.username": str,
    "execution.concurrency": int,
    "execution.timeout": int
}
```

### 自定义验证规则
```python
custom_validators = {
    "dify.base_url": lambda x: x.startswith("https://"),
    "execution.concurrency": lambda x: x > 0,
    "execution.timeout": lambda x: x > 0
}
```

## 高级功能

### 配置文件分层
```yaml
# 可以使用includes合并多个配置文件
includes:
  - "config/base.yaml"
  - "config/${ENV}.yaml"
  - "config/local.yaml"
```

### 配置模板
```yaml
# 使用模板语法动态生成配置
templates:
  api_endpoint: "https://${dify.base_url}/v${api_version}"
```

### 敏感信息加密
```yaml
# 支持加密存储敏感配置
secrets:
  api_key: "!encrypt(base64_encrypted_key)"
  password: "!encrypt(base64_encrypted_password)"
```

## 配置加载顺序

1. 默认配置 (config/default.yaml)
2. 环境配置 (config/{ENV}.yaml)
3. 本地配置 (config/local.yaml)
4. 环境变量替换
5. 命令行参数

## 错误处理

- 配置文件不存在异常
- YAML格式错误异常
- 必填字段缺失异常
- 数据类型不匹配异常
- 自定义验证失败异常
- 环境变量未设置异常

## 最佳实践

1. **文件组织**
   - 使用环境分离不同配置
   - 敏感信息使用环境变量
   - 保持配置文件简洁明了

2. **配置验证**
   - 定义完整的验证规则
   - 提供清晰的错误信息
   - 在应用启动时验证

3. **安全性**
   - 加密存储敏感配置
   - 避免在代码中硬编码配置
   - 定期更新密钥和密码

4. **可维护性**
   - 使用有意义的配置名
   - 添加配置说明和示例
   - 版本控制配置变更