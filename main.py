"""
根目录入口脚本：演示并初始化日志系统

用法：
    python main.py            # 使用默认配置初始化日志
    python main.py config/logging_config.yaml  # 指定日志配置文件

说明：
- 该脚本仅负责日志初始化与简单示例输出，不与业务执行耦合。
- 业务入口仍为 src/main.py（支持 test/optimize/report 模式）。
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional


# 确保可以以包形式导入 src 下的模块
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


async def init_logger(log_cfg: Optional[str] = None) -> None:
    """初始化日志系统并输出示例日志。"""
    from src.utils.logger import (
        setup_logging,
        get_logger,
        _log_manager,
        log_context,
        log_exception,
        log_workflow_trace,
    )

    # 1) 初始化
    cfg = log_cfg or "config/logging_config.yaml"
    await setup_logging(cfg)

    # 2) 设置全局上下文（可选）
    _log_manager.set_global_context(app="dify_autoopt", environment="development")

    # 3) 基础日志
    logger = get_logger("root.main")
    logger.info("Logger initialized", extra={"config": cfg})

    # 4) 上下文示例
    with log_context(request_id="req_root_001"):
        logger.info("message with contextual fields")

    # 5) 工作流阶段跟踪与异常捕获示例
    @log_exception(reraise=False)
    async def demo_task():
        with log_workflow_trace("wf_root_demo", "op_demo", logger):
            logger.info("doing demo work...")
            await asyncio.sleep(0.05)
            # 故意不抛错，仅演示

    await demo_task()


async def main() -> int:
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        await init_logger(cfg)
        return 0
    except Exception as e:
        # 如果初始化失败，尽量打印基础错误信息
        print(f"Logger initialization failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

