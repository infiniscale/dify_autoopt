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

## gitflow 相关安装

什么是 Git Flow？
Git Flow 是一种结构化的分支模型，用于管理发布、开发、特性、热修复等流程。文章中指出，主要分支类型包括：
	•	main：生产环境可部署代码。  ￼
	•	develop：整合特性分支，作为下一个发布分支的基础。  ￼
	•	feature/...：开发新的功能。  ￼
	•	release/...：准备发布的分支。  ￼
	•	hotfix/...：生产紧急修复。  ￼

自动化组件简介
为了在 Claude Code 中实现 Git Flow 自动化，文章提出以下几个关键组件：  ￼
	•	子代理 (Subagent)：专注于 Git Flow 操作的 AI 助手。
	•	Slash 命令 (Slash Commands)：快捷命令，用于快速触发 Git Flow 操作。
	•	钩子 (Hooks)：自动执行校验与流程规范（如分支命名验证、防止直接 push 到 main/develop）。
	•	状态行 (Statusline)：实时在终端界面显示当前 Git Flow 状态。
	•	设置 (Settings)：权限与环境的配置。

### 快速安装与配置命令
```bash

# 安装 Git Flow 管理子代理

npx claude-code-templates@latest --agent=git/git-flow-manager --yes

# 安装状态行
npx claude-code-templates@latest --setting=statusline/git-branch-statusline --yes

# 安装 Slash 命令
npx claude-code-templates@latest --command=git/feature --yes
npx claude-code-templates@latest --command=git/release --yes
npx claude-code-templates@latest --command=git/hotfix --yes
npx claude-code-templates@latest --command=git/finish --yes
npx claude-code-templates@latest --command=git/flow-status --yes

# 安装钩子
npx claude-code-templates@latest --hook=git/validate-branch-name --yes
npx claude-code-templates@latest --hook=git/prevent-direct-push --yes
```

工作流程示例
文章以一个从新功能开发到发布，再到热修复的完整流程为例：  ￼
	1.	查看状态： /flow-status
	2.	新建功能分支： /feature user-auth
	3.	完成功能后合并： /finish
	4.	创建准备发布分支： /release v1.2.0
	5.	完成发布： /finish
	6.	紧急修复生产问题： /hotfix critical-security-patch
	7.	修复后完成热修复： /finish

钩子配置说明
	•	分支命名验证：安装 –hook=git/validate-branch-name 后，如果新建分支名称不符合规范，会被阻止或提醒。  ￼
	•	禁止直接 push 到 main 或 develop：安装 –hook=git/prevent-direct-push 后，系统会阻止直接 push。  ￼

小贴士 / 注意事项
	•	在项目根目录保持良好的 CLAUDE.md 或 .claude/agents/ 结构，以便 Claude Code 代理正确识别和执行流程。
	•	在团队使用前，建议先在一个 sandbox 分支上测试该自动化流程，以避免误操作。
	•	虽然自动化提升效率，但仍需人工 code review 和版本发布批准环节，避免完全依赖自动代理。
	•	分支命名规范、提交信息规范、合并策略等流程需事先在团队达成一致，并在 CLAUDE.md 中明确。


## GitFlow 配置检查命令

检查当前仓库的 GitFlow 配置状态：
```bash
# 使用 git-flow-manager 代理检查配置
/agent run git-flow-manager "Check the gitflow configuration in this repository and provide a comparative status table"
```

## 注意事项

1. **重启要求**: 安装插件后需要重启 Claude Code 才能加载新插件。
2. **重复安装**: 如果某个插件已经安装，系统会提示该插件已存在，无需重复安装。
3. **Marketplace**: 所有插件都来自 `wshobson/agents` marketplace。

---

*Last updated: 2025-11-12*
