# 主程序入口

## 功能概述

Dify自动化测试与提示词优化工具的主入口文件，整合所有功能模块，提供统一的命令行接口和服务启动功能。

## 核心职责

- 🚀 应用程序启动和初始化
- ⚙️ 配置加载和验证
- 🎯 命令行参数解析
- 🔄 各模块协调和管理
- 📊 生命周期管理
- 🛠️ 调试和开发工具

## 使用方法

### 基础执行
```bash
# 运行基础测试
python src/main.py --mode test

# 运行提示词优化
python src/main.py --mode optimize --workflow-id wf001

# 生成测试报告
python src/main.py --mode report --output report.xlsx
```

### 高级配置
```bash
# 使用自定义配置文件
python src/main.py --config custom_config.yaml --mode test

# 指定日志级别
python src/main.py --mode optimize --log-level DEBUG

# 启用详细输出
python src/main.py --mode test --verbose

# 指定输出格式
python src/main.py --mode report --output-format json
```

## 命令行参数

```bash
$ python src/main.py --help

usage: main.py [-h] [--config CONFIG] [--mode {test,optimize,report,serve}]
               [--workflow-id WORKFLOW_ID] [--output OUTPUT]
               [--output-format {excel,json,html}] [--log-level {DEBUG,INFO,WARNING,ERROR}]
               [--verbose] [--dry-run] [--no-retry] [--max-concurrency MAX_CONCURRENCY]

Dify自动化测试与提示词优化工具

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG, -c CONFIG
                        配置文件路径 (default: config/config.yaml)
  --mode {test,optimize,report,serve}, -m {test,optimize,report,serve}
                        运行模式 (default: test)
  --workflow-id WORKFLOW_ID, -w WORKFLOW_ID
                        工作流ID (用于优化模式)
  --output OUTPUT, -o OUTPUT
                        输出文件路径
  --output-format {excel,json,html}, -f {excel,json,html}
                        输出格式 (default: excel)
  --log-level {DEBUG,INFO,WARNING,ERROR}, -l {DEBUG,INFO,WARNING,ERROR}
                        日志级别 (default: INFO)
  --verbose, -v         启用详细输出
  --dry-run             试运行模式，不执行实际操作
  --no-retry            禁用自动重试
  --max-concurrency MAX_CONCURRENCY
                        最大并发数 (default: 5)
```

## 运行模式

### 1. 测试模式 (test)
```bash
python src/main.py --mode test
```

**功能**：
- 自动发现和测试指定的工作流
- 并发执行测试用例
- 收集执行结果和性能数据
- 生成测试报告

**输出**：
- Excel格式的测试报告
- 控制台实时结果
- 详细的执行日志

### 2. 优化模式 (optimize)
```bash
python src/main.py --mode optimize --workflow-id wf001
```

**功能**：
- 提取工作流中的LLM提示词
- 使用AI分析和评估提示词质量
- 生成优化建议和新版本提示词
- 验证优化效果

**输出**：
- 提示词优化报告
- 优化前后的对比数据
- 版本更新记录

### 3. 报告模式 (report)
```bash
python src/main.py --mode report --output report.html --output-format html
```

**功能**：
- 基于历史数据生成深度分析报告
- 多维度数据可视化
- 趋势分析和预测
- 智能优化建议

**输出**：
- 多格式报告 (HTML/PDF/Excel)
- 交互式图表
- 可分享的链接

### 4. 服务模式 (serve)
```bash
python src/main.py --mode serve
```

**功能**：
- 启动Web API服务
- 提供RESTful接口
- 支持远程调用和集成
- 实时监控和状态展示

**输出**：
- Web API服务
- Swagger文档
- 健康检查端点

## 应用架构

### 初始化流程
```python
async def initialize_app(config_path: str) -> Application:
    """应用初始化流程"""
    # 1. 加载配置
    config = ConfigLoader().load_config(config_path)

    # 2. 验证配置
    ConfigValidator().validate(config)

    # 3. 初始化日志系统
    setup_logging(config['logging'])

    # 4. 初始化各个模块
    auth_manager = AuthManager(config['auth'])
    workflow_manager = WorkflowManager(config['workflow'])
    executor = ConcurrentExecutor(config['executor'])
    collector = DataCollector(config['collector'])

    # 5. 创建应用实例
    app = Application(
        config=config,
        auth_manager=auth_manager,
        workflow_manager=workflow_manager,
        executor=executor,
        collector=collector
    )

    # 6. 初始化完成
    logger.info("应用初始化完成")
    return app
```

### 模块 coordination
```python
import asyncio
from typing import Dict, Any

class Application:
    def __init__(self, config: Dict[str, Any], **modules):
        self.config = config
        self.auth_manager = modules['auth_manager']
        self.workflow_manager = modules['workflow_manager']
        self.executor = modules['executor']
        self.collector = modules['collector']
        self.optimizer = modules.get('optimizer')
        self.report_generator = modules.get('report_generator')

    async def run_test_mode(self, workflow_ids: List[str] = None) -> TestResults:
        """运行测试模式"""
        try:
            # 1. 认证
            await self.auth_manager.authenticate()

            # 2. 获取工作流列表
            if not workflow_ids:
                workflows = await self.workflow_manager.discover_workflows()
                workflow_ids = [wf.id for wf in workflows]

            # 3. 准备测试任务
            test_tasks = await self._prepare_test_tasks(workflow_ids)

            # 4. 执行测试
            results = await self.executor.run_tasks(test_tasks)

            # 5. 收集和分析结果
            await self.collector.collect_results(results)

            # 6. 生成报告
            report = await self._generate_test_report(results)

            return TestResults(results=results, report=report)

        except Exception as e:
            logger.error(f"测试执行失败: {e}")
            raise

    async def run_optimize_mode(self, workflow_id: str) -> OptimizationResults:
        """运行优化模式"""
        try:
            # 1. 认证
            await self.auth_manager.authenticate()

            # 2. 提取工作流信息
            workflow = await self.workflow_manager.get_workflow(workflow_id)

            # 3. 提取提示词
            prompts = await self.optimizer.extract_prompts(workflow)

            # 4. 优化提示词
            optimization_results = []
            for prompt in prompts:
                result = await self.optimizer.optimize_prompt(prompt)
                optimization_results.append(result)

            # 5. 验证优化效果
            validation_results = await self._validate_optimizations(
                workflow_id, optimization_results
            )

            # 6. 生成优化报告
            report = await self._generate_optimization_report(
                optimization_results, validation_results
            )

            return OptimizationResults(
                workflow_id=workflow_id,
                optimizations=optimization_results,
                validation=validation_results,
                report=report
            )

        except Exception as e:
            logger.error(f"优化执行失败: {e}")
            raise
```

## 配置系统

### 环境配置
```python
# .env 文件示例
DIFY_BASE_URL=https://api.dify.ai
DIFY_API_KEY=your_api_key_here
DIFY_USERNAME=your_username
DIFY_PASSWORD=your_password

# 数据库配置
DATABASE_URL=sqlite:///data/app.db
REDIS_URL=redis://localhost:6379

# 日志配置
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# 开发配置
DEBUG=False
VERIFY_SSL=True
```

### 运行时配置
```python
# 支持的配置覆盖
app_config = {
    "auth": {
        "timeout": 30,
        "retry_count": 3
    },
    "executor": {
        "max_concurrency": 10,
        "timeout": 300
    },
    "collector": {
        "batch_size": 100,
        "output_format": "excel"
    },
    "optimizer": {
        "max_iterations": 5,
        "optimization_strategy": "gradient_descent"
    }
}
```

## 监控和健康检查

### 健康检查端点
```python
async def health_check() -> Dict[str, Any]:
    """健康检查"""
    checks = {}

    # 数据库连接检查
    try:
        await db.execute("SELECT 1")
        checks["database"] = "healthy"
    except Exception:
        checks["database"] = "unhealthy"

    # 外部服务检查
    try:
        response = await http_client.get("/health")
        checks["external_api"] = "healthy" if response.status == 200 else "unhealthy"
    except Exception:
        checks["external_api"] = "unhealthy"

    # 综合健康状态
    overall_status = "healthy" if all(
        status == "healthy" for status in checks.values()
    ) else "unhealthy"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "checks": checks
    }
```

### 性能监控
```python
import time
from functools import wraps

def monitor_performance(func):
    """性能监控装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            # 记录性能指标
            metrics.record_executionTime(func.__name__, execution_time)

            return result
        except Exception as e:
            execution_time = time.time() - start_time

            # 记录错误指标
            metrics.record_error(func.__name__, e, execution_time)

            raise
    return wrapper
```

## 错误处理和异常管理

### 全局异常处理器
```python
async def global_exception_handler(app: Application):
    """全局异常处理"""
    try:
        yield
    except DifyAuthException as e:
        logger.error(f"认证异常: {e}")
        await handle_auth_error(e)
    except WorkflowExecutionException as e:
        logger.error(f"工作流执行异常: {e}")
        await handle_workflow_error(e)
    except NetworkException as e:
        logger.error(f"网络异常: {e}")
        await handle_network_error(e)
    except Exception as e:
        logger.error(f"未知异常: {e}")
        await handle_unknown_error(e)
    finally:
        await cleanup_resources()
```

### 优雅关闭
```python
async def graceful_shutdown(signum, frame):
    """优雅关闭信号处理"""
    logger.info("收到关闭信号，开始优雅关闭...")

    try:
        # 停止接受新任务
        await executor.stop_accepting_tasks()

        # 等待当前任务完成
        await executor.wait_for_completion()

        # 保存状态
        await state_manager.save_state()

        # 清理资源
        await cleanup_resources()

        logger.info("优雅关闭完成")

    except Exception as e:
        logger.error(f"关闭过程中出现异常: {e}")
        os._exit(1)

    os.exit(0)
```

## 开发和调试

### 调试模式
```bash
# 启用调试模式
python src/main.py --mode test --log-level DEBUG --verbose

# 试运行模式
python src/main.py --mode test --dry-run

# 禁用重试机制
python src/main.py --mode test --no-retry
```

### 性能分析
```bash
# 启用性能分析
python -m cProfile -o profile.stats src/main.py --mode test

# 使用内存分析
python -m memory_profiler src/main.py --mode test
```

### 开发工具集成
```python
# 支持的开发工具
dev_tools = {
    "debugger": "pdb",  # Python调试器
    "profiler": "cProfile",  # 性能分析器
    "memory_profiler": "memory_profiler",  # 内存分析器
    "code_coverage": "coverage.py",  # 代码覆盖率
    "type_checker": "mypy",  # 类型检查
    "linter": "flake8"  # 代码规范检查
}
```

## 部署支持

### Docker部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

EXPOSE 8000
CMD ["python", "src/main.py", "--mode", "serve"]
```

### systemd服务
```ini
[Unit]
Description=Dify AutoOpt Service
After=network.target

[Service]
Type=simple
User=dify-autoopt
WorkingDirectory=/opt/dify-autoopt
ExecStart=/opt/dify-autoopt/venv/bin/python src/main.py --mode serve
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

主程序入口是整个系统的调度中心，负责协调各个子模块的工作，提供统一的用户接口和完善的错误处理机制。