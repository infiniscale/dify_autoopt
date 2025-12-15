# 项目开发计划（螺旋模型）

> 更新时间：2025-01-18

> 项目：dify_autoopt
> 文档目的：记录当前完成度与待办，基于螺旋模型制定迭代计划与测试计划，指导后续交付。

## 一、范围与目标
- 目标：提供可运行的“测试/优化/报告”一体化 CLI 工具，具备稳定的执行器、优化器、配置与采集能力，并有完善的日志与测试体系。
- 范围：`src/optimizer/`、`src/executor/`、`src/collector/`、`src/config/` 现有能力整合与打通主入口；补齐最小可用的 `workflow/` 与 `report/`；完善日志模块；补齐依赖与 CI；建立覆盖率与质量门禁。

## 二、现状快照（完成/未完成）
- 已完成/近期调整
  - 轻量规则优化器：基于执行结果生成补丁、副本写出 patched YAML，不改原文件；可选 LLM/dspy 重写；prompt 审核前置。
  - 工作流执行：支持从 unified config 读取 inputs/api_key，执行并落盘 JSON + summary；optimize 模式可调度执行→优化→报告。
  - YAML 读取：`optimizer/yaml_loader` 支持 output_dir/output 键，安全目录名。
  - 依赖更新：`requirements.txt` 补齐 LLM/dspy 及核心库版本。
- 未完成/差距
  - 调度器缺失：无统一的 dry_run/optimize 调度、并发/迭代控制、验证回路。
  - 日志模块：部分示例 API 仍未实现/对齐。
  - 文档/配置：大量 docs 被删，示例配置键不统一（output vs output_dir），用户指引不足。
  - 测试：未跑最新单测；大量 legacy optimizer 测试被 skip/依赖缺失模块。
  - DSL 导出：optimize 流程未保证执行前自动导出 DSL，依赖已有文件。
  - CI：仍缺流水线与覆盖率门禁。

## 三、约束与风险
- 约束：
  - 不应将敏感凭据写入代码/配置；通过 `.env` 与环境变量管理。
  - 单测不得访问真实 Dify 端点；需使用 stub/fake。
- 主要风险与缓解：
  - 文档与实现不一致导致使用失败 → 优先收敛入口与日志 API，提供最小可用闭环与示例。
  - 外部依赖变动与网络不稳定 → 引入重试与可配置超时，测试中隔离网络。
  - 模块集成复杂 → 分阶段小步迭代，每轮交付可验证产物与回滚方案。

## 四、螺旋模型迭代计划

> 每一轮包含：目标/约束识别 → 风险评估 → 开发与验证 → 计划下一轮。

### 迭代 0：运行骨架与依赖收敛（T+1 周）
- 目标：让三种模式可运行（允许 stub），日志与依赖对齐。
- 交付物：
  - main：清理注释逻辑，保证 `--mode test|optimize|report` 可跑；optimize 前自动导出 DSL。
  - 日志：对齐示例所用 API（或降级示例），避免导入时抛未初始化异常。
  - 依赖/CI：更新 requirements，初步 GH Actions（ruff/black/pytest -q）。
- 验证：本地跑通三模式最小流程；单测基础套件通过。

### 迭代 1：调度器与优化闭环（T+2 周）
- 目标：引入 Scheduler 支持 dry_run/optimize，支持并发、迭代与验证闭环。
- 交付物：
  - Scheduler：配置化并发、模式切换、max_iterations、终止条件。
  - Verifier：对 patched DSL 小样本重跑，评估失败率/相似度/约束覆盖。
  - 输出契约：每轮 patched YAML、patches、validation 报告落盘。
- 验证：E2E（stub/或最小真实）覆盖执行→优化→验证→迭代。

### 迭代 2：健壮性与可观测性（T+2 周）
- 目标：并发/速率控制、日志增强、错误隔离。
- 交付物：速率/重试策略、结构化日志上下文、性能/错误指标、更多约束/策略切换（规则/LLM/dspy）。
- 验证：并发压测，错误路径、回退路径验证，覆盖率提升。

### 迭代 3：文档/配置对齐与发布（T+1 周）
- 目标：文档、示例、配置键统一（output_dir vs output），发布说明与回滚方案。
- 验证：CI 全绿，文档与实现一致。

## 五、工作分解（WBS，按迭代）
- 迭代 0
  - CLI 骨架与参数解析
  - 替换配置导入链路（`ConfigLoader/Validator`）
  - 日志最小补齐或示例调整
  - 依赖与 CI（ruff/black/pytest-cov）
- 迭代 1
  - `workflow.discovery/runner` 最小实现
  - `report` 简报导出
  - `main.py` 模式分发整合
  - E2E 与错误路径测试
- 迭代 2
  - 日志增强与指标统计
  - 工作流筛选与报告维度扩展
  - 并发/性能测试与稳态策略
- 迭代 3
  - 文档/样例对齐
  - 发布流程与门禁强化

## 六、测试计划
- 策略
  - 分层测试：单元 → 组件/集成 → 端到端 → 非功能（并发/性能/稳健性）。
  - 隔离外部依赖：通过 stub/fake，禁止真实网络与真实 Dify 调用。
  - 风险驱动：对入口整合、并发执行、日志/采集路径与配置加载优先覆盖。
- 覆盖率目标
  - 基线：全局 ≥ 80%；关键模块（executor、optimizer、config.loader/validator、logger）≥ 85%。
- 用例类型
  - 单元：模型校验、加载/校验器、执行器调度/限流、优化策略、日志格式化与装饰器。
  - 组件/集成：执行器+采集器流水、优化器+版本存储、配置跨文件引用校验。
  - E2E：`--mode test|optimize|report` 串起最小闭环（使用样例 catalog 与 stub）。
  - 错误路径：配置缺失/格式错误、网络异常（模拟）、超时/重试、队列积压、磁盘写失败（模拟）。
  - 非功能：并发 10/50/100 压力下的吞吐与延迟、日志写入开销评估。
- 测试数据与环境
  - 使用 `config/examples/` 提供 env/catalog/plan 示例；`.env.example` 指引本地变量。
  - 临时目录与内存存储，避免真实 I/O；pytest fixtures 统一注入。
- 工具与流水线
  - `pytest`/`pytest-cov`、`ruff`、`black`；GitHub Actions 集成安装依赖并执行覆盖率与格式检查。
- 进入/退出准入
  - 进入：模块接口稳定、样例配置可用、关键路径打桩完成。
  - 退出：CI 全绿、覆盖率达标、关键 E2E 与错误路径通过、文档与实现一致。

## 七、完成判定（DoD）
- CLI 三模式可运行并输出预期产物（允许 stub）。
- 单测通过且覆盖率达标；关键错误路径覆盖。
- 日志结构化字段与上下文在关键路径可追踪。
- 文档、示例、配置与实现一致；CI 门禁生效。

## 八、变更与发布流程
- 分支：Git Flow（feature/fix/hotfix），`.claude/hooks/` 维持规范。
- 提交与 PR：小步提交，PR 附目的/测试计划/截图（报告类）。
- 发布：拟定 Release Notes 与回滚方案，必要时打 Tag。

## 九、开放问题（跟踪）
- DSL 导出与 optimize 时机：是否强制 optimize 前导出？导出失败时的回退策略。
- 配置键统一：`io_paths.output_dir` vs `io_paths.output`，需要统一或兼容方案。
- Scheduler 策略：迭代终止条件、验证样本量、策略切换（规则/LLM/dspy）的默认值。
- 日志 API：是否补齐示例中引用的所有接口，或调整示例。

## 十、设计方案（入口瘦身 + 调度器接管）— 2025-01-18
- 主入口职责最小化
  - 仅解析 CLI 参数（config/catalg/mode/set/log-config 等），加载 .env，初始化日志，启动调度器。
  - 禁止在 main 中直接调用执行/优化逻辑；main 只负责把配置路径/模式/覆盖项传递给调度器。
- 调度器（Scheduler）职责
  - 读取配置：统一入口，加载 runtime（unified config），解析 `scheduler` 段（mode、concurrency、max_iterations、validation_sample、criteria、strategies、dry_run 标志）。
  - 模式分发：`mode=dry_run|optimize|report`，dry_run 仅做 DSL 导出/静态检查，optimize 走执行→优化→验证→迭代，report 调用报告生成。
  - 工作流分发：按 concurrency 并行不同 workflow；单 workflow 内的迭代串行，保留每轮产物（runs、patches、patched.yml、validation 报告）。
  - 迭代与终止：基于 criteria（failure_rate/similarity/constraint_coverage/latency）判断达标；不达标则按 strategies 切换（rule→LLM→dspy 或保守→激进），直到 max_iterations。
  - 验证：Verifier 组件对 patched YAML 取样本重跑（可用 stub 或真实执行），生成验证指标。
  - 产物：每轮写 `run_*.json`、`runs_summary.json`、`prompt_patches_iter_{n}.json`、`app_{id}_patched_iter_{n}.yml`、`validation_report_iter_{n}.json`，汇总 `scheduler_summary.json`。
  - 回退策略：导出失败或执行失败时记录错误并跳过；LLM/dspy 不可用则回退规则；验证恶化则保留上一轮最佳补丁。
- 配置接口（建议）
  - `scheduler.mode`、`scheduler.concurrency`、`scheduler.max_iterations`、`scheduler.validation_sample_size`、`scheduler.criteria.{failure_rate,similarity,constraint_coverage,latency}`、`scheduler.strategies`、`scheduler.dry_run`。
  - `optimization.llm` 保持用于 judge/rewrite/dspy；允许 disable。
- 风险与缓解
  - 并发/速率限制：调度器内置速率/重试策略；可配置节流。
  - 成本控制：dry_run 默认不打真实 API；optimize 验证可用 stub/sample。
  - 可追溯：所有修改写副本，不改原 YAML；日志上下文记录 workflow/iteration。

## 十、里程碑与时间线（建议）
- 迭代 0：T+1 周 — 可运行骨架与 CI 基线。
- 迭代 1：T+2 周 — 功能闭环与 E2E。
- 迭代 2：T+2 周 — 健壮性、性能与可观测性。
- 迭代 3：T+1 周 — 文档与发布。

— 完 —
