# Dify自动化测试与提示词优化工具

## 项目概述

这是一个基于大语言模型的Dify自动化测试与提示词优化工具，旨在自动获取Dify工作流并对其中的LLM提示词进行智能分析和优化。

## 功能特性

### 核心功能
- 🔐 **身份认证管理** - 自动化登录、会话管理和API密钥管理
- 🔄 **工作流管理** - 自动发现、获取、运行和发布Dify工作流
- 📝 **配置管理** - 基于YAML的灵活配置系统
- 🚀 **并发执行** - 支持高并发测试和性能评估
- 📊 **结果采集** - 全面的测试数据收集和分析
- 📈 **智能报告** - 基于AI的测试结果分析和优化建议
- 🔧 **通用工具** - 完善的辅助功能模块

### 提示词优化
- 🔍 **自动提取** - 智能识别工作流中的LLM提示词
- 🧠 **智能分析** - 基于静态分析和AI驱动的质量评估
- 🎯 **算法优化** - 基于静态分析和可选baseline指标的多策略优化
- 📚 **版本管理** - 完整的提示词版本控制和回滚机制

### 新增功能 (v1.1)
- 🧪 **测试驱动优化** - 基于真实测试结果的多维度优化决策
- 📊 **多指标监控** - 成功率、响应时间、成本、错误分布全方位分析
- 🔢 **语义化版本** - 自动根据改进幅度生成major/minor/patch版本号

## 系统架构

```
src/
├── auth/                    # 身份与权限模块
│   ├── __init__.py
│   ├── login.py            # 登录认证
│   ├── session.py          # 会话管理
│   └── api_key.py          # API密钥管理
├── workflow/                # 工作流管理模块
│   ├── __init__.py
│   ├── discovery.py        # 工作流发现
│   ├── runner.py           # 工作流运行
│   └── publisher.py        # 工作流发布
├── config/                 # 配置管理模块
│   ├── __init__.py
│   ├── yaml_loader.py      # YAML配置加载
│   └── validator.py        # 配置验证
├── executor/               # 调用执行模块
│   ├── __init__.py
│   ├── concurrent.py       # 并发执行器
│   └── scheduler.py        # 任务调度器
├── collector/              # 结果采集模块
│   ├── __init__.py
│   ├── data_collector.py   # 数据收集器
│   ├── excel_exporter.py   # Excel导出器
│   └── classifier.py       # 结果分类器
├── report/                # 报告模块
│   ├── __init__.py
│   ├── analyzer.py         # 结果分析器
│   ├── generator.py        # 报告生成器
│   └── optimizer.py        # 优化建议器
├── optimizer/             # 智能优化模块（新增）
│   ├── __init__.py
│   ├── prompt_extractor.py # 提示词提取
│   ├── prompt_analyzer.py  # Prompt质量分析器(规则+启发式)
│   ├── optimization_engine.py # 优化引擎
│   └── version_manager.py  # 版本管理
├── utils/                 # 通用工具模块
│   ├── __init__.py
│   ├── logger.py           # 日志管理
│   ├── http_client.py      # HTTP客户端
│   └── exceptions.py       # 异常定义
main.py               # 主程序入口（根目录）
```

## 技术栈

- **语言**: Python 3.8+
- **配置管理**: PyYAML
- **数据处理**: pandas, numpy
- **HTTP客户端**: aiohttp, requests
- **并发处理**: asyncio, threading
- **数据存储**: SQLite, Redis
- **报告生成**: openpyxl, matplotlib
- **测试框架**: pytest
- **日志系统**: loguru

## 快速开始

### 1. 环境准备
```bash
# 克隆项目
git clone https://github.com/infiniscale/dify_autoopt.git
cd dify_autoopt

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
```

### 2. 配置设置（单一配置文件）

本项目使用单一配置文件 `config/config.yaml`，包含以下顶层块：

- `meta`、`dify`、`auth`、`variables`、`workflows`、`execution`、`optimization`、`io_paths`、`logging`

示例（节选）：
```yaml
meta:
  version: "1.0.0"
  environment: "development"

dify:
  base_url: "https://your-dify-instance.com"

auth:
  # 二选一：优先 api_key；否则使用 username/password
  api_key: "${DIFY_API_TOKEN}"
  # username: "your_username"
  # password: "your_password"

workflows:
  - id: "wf_text_classify"
    name: "文本分类工作流"
    inputs: { }
    parameters: { }

execution:
  concurrency: 5
  timeout: 300
  retry_count: 3

optimization:
  strategy: "auto"   # auto | clarity_focus | efficiency_focus | structure_focus | llm_guided
  max_iterations: 3

io_paths:
  output_dir: "./outputs"
  logs_dir: "./logs"

logging:
  level: "INFO"      # DEBUG | INFO | WARNING | ERROR | CRITICAL
  format: "simple"   # simple | structured
  console_enabled: true
  file_enabled: true
```

### 3. 运行

```bash
# 基础运行（使用默认配置路径 config/config.yaml）
python main.py --mode test

# 指定配置路径
python main.py --mode test --config config/config.yaml

# 覆盖运行时配置（可多次 --set，使用 dot-path）
python main.py --mode test --config config/config.yaml \
  --set logging.level=DEBUG \
  --set optimization.strategy=auto

# 生成测试报告（JSON）
python main.py --mode test --report report.json
```

说明：
- 未设置 `--config` 时自动尝试 `config/config.yaml`；如不存在则使用内置默认配置（会在日志中提示）。
- `.env` 会在启动前自动加载（如未安装 `python-dotenv`，则忽略）。
- 日志配置优先从 `config/config.yaml` 的 `logging` 块读取。

## 单元测试

本项目使用 pytest 进行单元测试，测试文件位于 `src/test/`，命名为 `test_*.py`。

快速运行
```bash
# 从项目根目录运行所有测试
python -m pytest -q

# 仅运行日志相关测试
pytest -q -k logger

# 仅运行 utils 目录下的测试
pytest -q -k utils

# 覆盖率报告（推荐）
pytest --cov=src --cov-report=term-missing
```

约定与提示
- 测试目录：`src/test/`（与 `src/` 结构对应）。
- 命名规范：文件 `test_*.py`，函数 `test_*`。
- 日志模块样例：参见 `src/test/utils/test_logger_basic.py` 与扩展用例；验证初始化、文件写入与上下文功能。
- 测试不应访问真实 Dify 端点，对 I/O 或网络进行隔离/伪造。

## 使用指南

### 基础测试

1. 在配置文件中指定要测试的工作流
2. 设置测试参数（并发数、重试次数等）
3. 运行测试命令
4. 查看生成的测试报告

### 提示词优化

1. 指定要优化的工作流ID
2. 配置优化策略和迭代次数
3. 系统自动分析现有提示词
4. 生成优化建议和新版本
5. 验证优化效果

### 批量测试

支持批量测试多个工作流：
```yaml
workflows:
  - name: "workflow_group_1"
    workflows: ["id1", "id2", "id3"]
    common_inputs:
      base_path: "/data/test"

  - name: "workflow_group_2"
    workflows: ["id4", "id5"]
    common_inputs:
      parameters: {...}
```

## 输出结果

### Excel报告格式

每个测试会生成包含以下工作表的Excel文件：

- `测试概览` - 整体测试结果统计
- `性能分析` - 响应时间、成功率等性能指标
- `错误分析` - 失败案例详细分析
- `提示词优化` - 提示词改进建议
- `趋势分析` - 历史对比趋势

### 优化报告

提示词优化结果包括：
- 原始提示词 vs 优化后提示词对比
- 性能提升量化和效果评估
- 多个优化版本对比
- 推荐的最佳实践

## 开发指南

### 添加新功能模块

1. 在相应的模块目录下创建新文件
2. 定义接口和实现逻辑
3. 添加单元测试
4. 更新配置文件结构

### 扩展优化算法

在 `src/optimizer/` 目录下添加新的优化策略：
```python
class CustomOptimizer(BaseOptimizer):
    def optimize(self, prompts, metrics):
        # 实现自定义优化逻辑
        pass
```

## 配置参考

### 配置校验
启动时会对关键配置进行严格校验并输出结构化日志（`config.bootstrap`）：
- 必填：`dify.base_url`；`auth.api_key` 或 `auth.username/password` 二者其一
- 建议：`logging.level`、`optimization.strategy`、`execution` 参数范围

如需运行时覆盖配置，可通过 `--set a.b.c=value` 多次指定，覆盖会写入临时 YAML 并作为本次引导的有效配置。
```bash
# 验证配置文件是否正确
python -m dify_opt.utils.validator config/config.yaml
```

## 故障排除

### 常见问题

1. **认证失败** - 检查API密钥和用户凭据
2. **工作流找不到** - 确认工作流ID和权限设置
3. **并发超限** - 调整并发数配置
4. **内存不足** - 减少批量处理的数量

### 日志查看

```bash
# 查看详细日志
tail -f logs/dify_opt.log

# 查看错误日志
grep ERROR logs/dify_opt.log
```

## 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 联系方式

- 项目主页: https://github.com/infiniscale/dify_autoopt
- 问题反馈: https://github.com/infiniscale/dify_autoopt/issues

---

*最后更新: 2025-11-19*
