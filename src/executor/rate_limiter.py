"""
Executor Module - Rate Limiter

Date: 2025-11-14
Author: backend-developer
Description: 实现令牌桶算法的速率限制器
"""

import time
import threading
from datetime import datetime
from typing import Callable, Optional

from src.config.models import RateLimit


class RateLimiter:
    """令牌桶速率限制器。

    实现令牌桶算法，支持：
    - 固定速率的令牌填充（per_minute）
    - 突发流量（burst）
    - 线程安全

    算法：
    - 桶容量 = burst
    - 填充速率 = per_minute / 60.0 tokens/second
    - 每次请求消耗1个令牌
    - 令牌不足时阻塞等待

    Attributes:
        _rate_limit: 速率限制配置
        _tokens: 当前令牌数
        _last_refill_time: 上次填充时间
        _lock: 线程锁
        _sleep_fn: 休眠函数（依赖注入）
        _now_fn: 时间函数（依赖注入）
    """

    def __init__(
        self,
        rate_limit: RateLimit,
        now_fn: Callable[[], datetime] = datetime.now,
        sleep_fn: Callable[[float], None] = time.sleep
    ) -> None:
        """初始化速率限制器。

        Args:
            rate_limit: 速率限制配置
            now_fn: 时间获取函数
            sleep_fn: 休眠函数
        """
        self._rate_limit = rate_limit
        self._now_fn = now_fn
        self._sleep_fn = sleep_fn

        # 令牌桶状态
        self._tokens = float(rate_limit.burst)  # 初始填满
        self._last_refill_time = now_fn()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> None:
        """获取令牌（阻塞直到令牌可用）。

        实现要点：
        1. 计算自上次填充以来应该新增的令牌数
        2. 填充令牌（不超过 burst 容量）
        3. 检查是否有足够令牌
        4. 如果不足，计算需要等待的时间并休眠
        5. 消耗令牌

        Args:
            tokens: 需要获取的令牌数（默认1）

        Raises:
            ValueError: 当 tokens > burst 时（无法满足）
        """
        if tokens > self._rate_limit.burst:
            raise ValueError(
                f"Requested {tokens} tokens exceeds burst capacity {self._rate_limit.burst}"
            )

        with self._lock:
            while True:
                # 计算自上次填充以来的时间
                now = self._now_fn()
                elapsed_seconds = (now - self._last_refill_time).total_seconds()

                # 计算应该填充的令牌数
                # 填充速率 = per_minute / 60.0 tokens/second
                tokens_to_add = (self._rate_limit.per_minute / 60.0) * elapsed_seconds

                # 填充令牌（不超过 burst）
                self._tokens = min(self._rate_limit.burst, self._tokens + tokens_to_add)
                self._last_refill_time = now

                # 检查是否有足够令牌
                if self._tokens >= tokens:
                    # 消耗令牌
                    self._tokens -= tokens
                    return

                # 令牌不足，计算需要等待的时间
                tokens_needed = tokens - self._tokens
                wait_seconds = tokens_needed / (self._rate_limit.per_minute / 60.0)

                # 休眠等待
                self._sleep_fn(wait_seconds)

    def try_acquire(self, tokens: int = 1) -> bool:
        """尝试获取令牌（非阻塞）。

        Args:
            tokens: 需要获取的令牌数

        Returns:
            bool: 成功获取返回 True，否则返回 False
        """
        if tokens > self._rate_limit.burst:
            return False

        with self._lock:
            # 填充令牌
            now = self._now_fn()
            elapsed_seconds = (now - self._last_refill_time).total_seconds()
            tokens_to_add = (self._rate_limit.per_minute / 60.0) * elapsed_seconds
            self._tokens = min(self._rate_limit.burst, self._tokens + tokens_to_add)
            self._last_refill_time = now

            # 检查并消耗令牌
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    @property
    def available_tokens(self) -> float:
        """获取当前可用令牌数（包含待填充令牌）。

        Returns:
            float: 当前可用令牌数
        """
        with self._lock:
            now = self._now_fn()
            elapsed_seconds = (now - self._last_refill_time).total_seconds()
            tokens_to_add = (self._rate_limit.per_minute / 60.0) * elapsed_seconds
            return min(self._rate_limit.burst, self._tokens + tokens_to_add)
