"""
Token Usage Tracker for LLM API Calls.

Date: 2025-11-18
Author: backend-developer
Description: Thread-safe token usage tracking with cost calculation.
"""

import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class TokenUsageTracker:
    """Thread-safe token usage tracker with cost calculation.

    This tracker monitors API usage across all LLM calls, supporting:
    - Token consumption tracking (input/output)
    - Cost calculation per request and aggregate
    - Request history with timestamps
    - Thread-safe operations
    - Daily/request limit checking

    Example:
        >>> tracker = TokenUsageTracker()
        >>> cost = tracker.track_usage("gpt-4-turbo-preview", 1000, 500)
        >>> print(f"Request cost: ${cost:.4f}")
        >>> stats = tracker.get_stats()
        >>> print(f"Total cost: ${stats['total_cost']:.4f}")
    """

    # Cost per 1K tokens (as of 2025)
    COST_PER_1K_TOKENS: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    })

    # Statistics
    _total_tokens: int = field(default=0)
    _total_cost: float = field(default=0.0)
    _request_count: int = field(default=0)
    _request_history: List[Dict] = field(default_factory=list)
    _total_latency_ms: float = field(default=0.0)

    # Thread safety
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def track_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float = 0.0
    ) -> float:
        """Track token usage and calculate cost.

        Args:
            model: Model name (e.g., "gpt-4-turbo-preview")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Request latency in milliseconds (optional)

        Returns:
            Cost in USD for this request

        Example:
            >>> cost = tracker.track_usage("gpt-4-turbo-preview", 1000, 500)
            >>> print(f"Cost: ${cost:.4f}")
        """
        with self._lock:
            # Get pricing for model
            pricing = self.COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})

            # Calculate cost
            input_cost = (input_tokens / 1000.0) * pricing["input"]
            output_cost = (output_tokens / 1000.0) * pricing["output"]
            total_cost = input_cost + output_cost

            # Update statistics
            total_tokens = input_tokens + output_tokens
            self._total_tokens += total_tokens
            self._total_cost += total_cost
            self._request_count += 1
            self._total_latency_ms += latency_ms

            # Record in history
            self._request_history.append({
                "timestamp": datetime.now(),
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost": total_cost,
                "latency_ms": latency_ms
            })

            return total_cost

    def get_stats(self) -> Dict:
        """Get current usage statistics.

        Returns:
            Dictionary with usage statistics including:
            - request_count: Total number of requests
            - total_tokens: Total tokens used
            - total_cost: Total cost in USD
            - average_latency_ms: Average request latency
            - requests_by_model: Breakdown by model

        Example:
            >>> stats = tracker.get_stats()
            >>> print(f"Total requests: {stats['request_count']}")
        """
        with self._lock:
            # Calculate requests by model
            requests_by_model: Dict[str, int] = {}
            for record in self._request_history:
                model = record["model"]
                requests_by_model[model] = requests_by_model.get(model, 0) + 1

            return {
                "request_count": self._request_count,
                "total_tokens": self._total_tokens,
                "total_cost": self._total_cost,
                "average_latency_ms": (
                    self._total_latency_ms / self._request_count
                    if self._request_count > 0
                    else 0.0
                ),
                "requests_by_model": requests_by_model,
                "history_size": len(self._request_history)
            }

    def check_limits(
        self,
        daily_limit: Optional[float] = None,
        request_limit: Optional[float] = None
    ) -> bool:
        """Check if usage is within specified limits.

        Args:
            daily_limit: Maximum cost per day (USD)
            request_limit: Maximum cost per request (USD)

        Returns:
            True if within limits, False otherwise

        Example:
            >>> if not tracker.check_limits(daily_limit=10.0):
            ...     print("Daily limit exceeded!")
        """
        with self._lock:
            # Check daily limit
            if daily_limit is not None:
                # Calculate cost for last 24 hours
                cutoff_time = datetime.now() - timedelta(days=1)
                daily_cost = sum(
                    record["cost"]
                    for record in self._request_history
                    if record["timestamp"] >= cutoff_time
                )

                if daily_cost > daily_limit:
                    return False

            # Check per-request limit
            if request_limit is not None and self._request_history:
                last_request_cost = self._request_history[-1]["cost"]
                if last_request_cost > request_limit:
                    return False

            return True

    def get_daily_cost(self) -> float:
        """Get total cost for the last 24 hours.

        Returns:
            Cost in USD for the last 24 hours

        Example:
            >>> daily_cost = tracker.get_daily_cost()
            >>> print(f"Last 24h cost: ${daily_cost:.4f}")
        """
        with self._lock:
            cutoff_time = datetime.now() - timedelta(days=1)
            return sum(
                record["cost"]
                for record in self._request_history
                if record["timestamp"] >= cutoff_time
            )

    def reset(self) -> None:
        """Reset all statistics.

        Example:
            >>> tracker.reset()
        """
        with self._lock:
            self._total_tokens = 0
            self._total_cost = 0.0
            self._request_count = 0
            self._request_history.clear()
            self._total_latency_ms = 0.0

    def get_recent_requests(self, limit: int = 10) -> List[Dict]:
        """Get recent request history.

        Args:
            limit: Maximum number of recent requests to return

        Returns:
            List of recent request records (most recent first)

        Example:
            >>> recent = tracker.get_recent_requests(5)
            >>> for req in recent:
            ...     print(f"{req['model']}: ${req['cost']:.4f}")
        """
        with self._lock:
            return list(reversed(self._request_history[-limit:]))
