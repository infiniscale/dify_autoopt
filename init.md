# 初始化指南 (Initialization Guide)

## 概述

本文档包含初始化项目时需要执行的插件安装命令。

## 插件安装 (Plugin Installation)

以下是从 marketplace 安装所需的 Claude Code 插件的命令：

### 添加 marketplace
```bash
/plugin marketplace add wshobson/agents
```

### 安装插件
```bash
# Python 开发插件
/plugin install python-development

# 后端开发插件
/plugin install backend-development

# 全栈编排插件
/plugin install full-stack-orchestration

# 代码审查 AI 插件
/plugin install code-review-ai

# 代码文档插件
/plugin install code-documentation

# 调试工具包插件
/plugin install debugging-toolkit

# Git PR 工作流插件
/plugin install git-pr-workflows

# 单元测试插件
/plugin install unit-testing
```

## 注意事项

1. **重启要求**: 安装插件后需要重启 Claude Code 才能加载新插件。
2. **重复安装**: 如果某个插件已经安装，系统会提示该插件已存在，无需重复安装。
3. **Marketplace**: 所有插件都来自 `wshobson/agents` marketplace。

---

*Last updated: 2025-11-12*