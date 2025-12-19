"""
Test Suite for RateLimiter

Date: 2025-11-14
Author: qa-engineer
Description: Comprehensive unit tests for RateLimiter with 100% coverage
"""

import threading
import time
from datetime import datetime, timedelta

import pytest

from src.config.models import RateLimit
from src.executor.rate_limiter import RateLimiter


class TestRateLimiterInit:
    """Test RateLimiter initialization."""

    def test_init_with_burst_tokens(self, rate_limit_config, mock_now_fn, mock_sleep_fn):
        """Test that initial tokens equal burst capacity.

        Arrange:
            - RateLimit config with burst=10
        Act:
            - Initialize RateLimiter
        Assert:
            - Initial tokens should equal burst capacity
        """
        limiter = RateLimiter(
            rate_limit=rate_limit_config,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn
        )
        assert limiter.available_tokens == rate_limit_config.burst

    def test_init_stores_config(self, rate_limit_config, mock_now_fn, mock_sleep_fn):
        """Test that configuration is stored correctly.

        Arrange:
            - RateLimit config
        Act:
            - Initialize RateLimiter
        Assert:
            - Configuration should be stored in _rate_limit
        """
        limiter = RateLimiter(
            rate_limit=rate_limit_config,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn
        )
        assert limiter._rate_limit == rate_limit_config

    def test_init_with_dependency_injection(self):
        """Test that dependency injection works correctly.

        Arrange:
            - Custom now_fn and sleep_fn
        Act:
            - Initialize RateLimiter with custom functions
        Assert:
            - Custom functions should be stored
        """
        custom_now = lambda: datetime(2025, 1, 1, 0, 0, 0)
        custom_sleep = lambda s: None
        rate_limit = RateLimit(per_minute=60, burst=10)

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=custom_now,
            sleep_fn=custom_sleep
        )

        assert limiter._now_fn == custom_now
        assert limiter._sleep_fn == custom_sleep


class TestAcquire:
    """Test RateLimiter.acquire() method."""

    def test_acquire_single_token(self, rate_limiter):
        """Test basic single token acquisition.

        Arrange:
            - RateLimiter with available tokens
        Act:
            - Acquire 1 token
        Assert:
            - Should succeed without blocking
        """
        initial_tokens = rate_limiter.available_tokens
        rate_limiter.acquire(1)
        assert rate_limiter.available_tokens == initial_tokens - 1

    def test_acquire_multiple_tokens(self, rate_limiter):
        """Test acquiring multiple tokens at once.

        Arrange:
            - RateLimiter with available tokens
        Act:
            - Acquire 5 tokens
        Assert:
            - Should consume 5 tokens
        """
        initial_tokens = rate_limiter.available_tokens
        rate_limiter.acquire(5)
        assert rate_limiter.available_tokens == initial_tokens - 5

    def test_acquire_consumes_tokens(self, rate_limit_config, mock_now_fn, mock_sleep_fn):
        """Test that tokens are consumed correctly.

        Arrange:
            - RateLimiter with 10 tokens
        Act:
            - Acquire 3 tokens, then 2 tokens
        Assert:
            - Should have 5 tokens remaining
        """
        limiter = RateLimiter(
            rate_limit=rate_limit_config,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn
        )
        limiter.acquire(3)
        limiter.acquire(2)
        assert limiter.available_tokens == 5.0

    def test_acquire_blocks_when_insufficient(self):
        """Test that acquire blocks when tokens are insufficient.

        Arrange:
            - RateLimiter with controllable time
            - Empty token bucket
        Act:
            - Try to acquire tokens
        Assert:
            - Should call sleep_fn with correct wait time
        """
        rate_limit = RateLimit(per_minute=60, burst=10)

        # Simulate time that doesn't advance (so no refill happens)
        base_time = datetime(2025, 1, 1, 0, 0, 0)
        time_calls = [0]

        def mock_now():
            # Return same time for first few calls, then advance after sleep
            if time_calls[0] < 3:
                time_calls[0] += 1
                return base_time
            else:
                # After sleep, time advances enough to have tokens
                return base_time + timedelta(seconds=5)

        sleep_calls = []

        def mock_sleep(s):
            sleep_calls.append(s)
            # Simulate time advancing during sleep
            time_calls[0] = 3

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=mock_now,
            sleep_fn=mock_sleep
        )

        # Exhaust all tokens
        limiter.acquire(10)

        # This should block and sleep
        limiter.acquire(1)

        assert len(sleep_calls) == 1
        assert sleep_calls[0] > 0

    def test_acquire_calculates_wait_time(self):
        """Test correct wait time calculation when blocking.

        Arrange:
            - RateLimiter with per_minute=60 (1 token/second)
            - No available tokens
        Act:
            - Acquire 5 tokens
        Assert:
            - Should calculate wait time as 5 seconds
        """
        rate_limit = RateLimit(per_minute=60, burst=10)

        base_time = datetime(2025, 1, 1, 0, 0, 0)
        time_calls = [0]

        def mock_now():
            if time_calls[0] < 3:
                time_calls[0] += 1
                return base_time
            else:
                return base_time + timedelta(seconds=6)

        sleep_calls = []

        def mock_sleep(s):
            sleep_calls.append(s)
            time_calls[0] = 3

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=mock_now,
            sleep_fn=mock_sleep
        )

        # Exhaust tokens
        limiter.acquire(10)

        # Acquire 5 tokens - should wait ~5 seconds
        limiter.acquire(5)

        assert len(sleep_calls) == 1
        assert abs(sleep_calls[0] - 5.0) < 0.01  # Allow small floating point error

    def test_acquire_refills_tokens_over_time(self):
        """Test that tokens are refilled based on elapsed time.

        Arrange:
            - RateLimiter with per_minute=60 (1 token/second)
            - Time advances by 5 seconds
        Act:
            - Check available tokens after time advance
        Assert:
            - Should have refilled 5 tokens
        """
        rate_limit = RateLimit(per_minute=60, burst=10)

        base_time = datetime(2025, 1, 1, 0, 0, 0)
        time_calls = [0]

        def mock_now():
            call_num = time_calls[0]
            time_calls[0] += 1
            if call_num < 3:
                # First 3 calls: base time (init, try_acquire x2)
                return base_time
            else:
                # Later calls: time has advanced
                return base_time + timedelta(seconds=5)

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=mock_now,
            sleep_fn=lambda s: None
        )

        # Consume 7 tokens
        limiter.try_acquire(7)
        assert limiter.available_tokens == 3.0

        # After 5 seconds, should have refilled 5 tokens (3 + 5 = 8)
        tokens_after = limiter.available_tokens
        assert tokens_after >= 7.5  # Should have ~8 tokens

    def test_acquire_respects_burst_capacity(self):
        """Test that tokens don't exceed burst capacity.

        Arrange:
            - RateLimiter with burst=10
            - Wait longer than needed to fill bucket
        Act:
            - Check available tokens
        Assert:
            - Should not exceed burst capacity
        """
        rate_limit = RateLimit(per_minute=60, burst=10)

        time_sequence = [
            datetime(2025, 1, 1, 0, 0, 0),
            datetime(2025, 1, 1, 0, 1, 0),  # +60 seconds
        ]
        time_index = [0]

        def mock_now():
            current = time_sequence[time_index[0]]
            time_index[0] = min(time_index[0] + 1, len(time_sequence) - 1)
            return current

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=mock_now,
            sleep_fn=lambda s: None
        )

        # Even after 60 seconds, tokens should be capped at burst
        assert limiter.available_tokens == 10.0

    def test_acquire_exceeds_burst_raises_error(self, rate_limiter):
        """Test that requesting more tokens than burst raises ValueError.

        Arrange:
            - RateLimiter with burst=10
        Act:
            - Try to acquire 15 tokens
        Assert:
            - Should raise ValueError
        """
        with pytest.raises(ValueError) as exc_info:
            rate_limiter.acquire(15)

        assert "exceeds burst capacity" in str(exc_info.value)


class TestTryAcquire:
    """Test RateLimiter.try_acquire() method."""

    def test_try_acquire_success(self, rate_limiter):
        """Test successful non-blocking token acquisition.

        Arrange:
            - RateLimiter with available tokens
        Act:
            - Try to acquire 3 tokens
        Assert:
            - Should return True and consume tokens
        """
        initial_tokens = rate_limiter.available_tokens
        result = rate_limiter.try_acquire(3)

        assert result is True
        assert rate_limiter.available_tokens == initial_tokens - 3

    def test_try_acquire_failure(self, rate_limit_config, mock_now_fn, mock_sleep_fn):
        """Test failed non-blocking acquisition when insufficient tokens.

        Arrange:
            - RateLimiter with no available tokens
        Act:
            - Try to acquire tokens
        Assert:
            - Should return False without blocking
        """
        limiter = RateLimiter(
            rate_limit=rate_limit_config,
            now_fn=mock_now_fn,
            sleep_fn=mock_sleep_fn
        )

        # Exhaust all tokens
        limiter.acquire(10)

        # Try to acquire more - should fail
        result = limiter.try_acquire(1)
        assert result is False

    def test_try_acquire_does_not_block(self):
        """Test that try_acquire never blocks/sleeps.

        Arrange:
            - RateLimiter with sleep tracking
            - No available tokens
        Act:
            - Try to acquire tokens
        Assert:
            - Should not call sleep_fn
        """
        rate_limit = RateLimit(per_minute=60, burst=10)

        sleep_calls = []
        mock_sleep = lambda s: sleep_calls.append(s)

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=lambda: datetime(2025, 1, 1, 0, 0, 0),
            sleep_fn=mock_sleep
        )

        # Exhaust tokens
        limiter.acquire(10)

        # Try acquire should not sleep
        limiter.try_acquire(5)

        # Only one sleep call from acquire(10)
        assert len(sleep_calls) == 0

    def test_try_acquire_exceeds_burst(self, rate_limiter):
        """Test that requesting more than burst returns False.

        Arrange:
            - RateLimiter with burst=10
        Act:
            - Try to acquire 15 tokens
        Assert:
            - Should return False (not raise exception)
        """
        result = rate_limiter.try_acquire(15)
        assert result is False


class TestAvailableTokens:
    """Test RateLimiter.available_tokens property."""

    def test_available_tokens_initial(self, rate_limiter, rate_limit_config):
        """Test initial available tokens equals burst.

        Arrange:
            - Newly created RateLimiter
        Act:
            - Check available_tokens
        Assert:
            - Should equal burst capacity
        """
        assert rate_limiter.available_tokens == rate_limit_config.burst

    def test_available_tokens_after_acquire(self, rate_limiter):
        """Test available tokens decrease after acquisition.

        Arrange:
            - RateLimiter with tokens
        Act:
            - Acquire some tokens
            - Check available_tokens
        Assert:
            - Should reflect consumed tokens
        """
        rate_limiter.acquire(4)
        assert rate_limiter.available_tokens == 6.0

    def test_available_tokens_after_refill(self):
        """Test available tokens increase after time passes.

        Arrange:
            - RateLimiter with time progression
        Act:
            - Consume tokens, advance time, check tokens
        Assert:
            - Should show refilled tokens
        """
        rate_limit = RateLimit(per_minute=60, burst=10)

        base_time = datetime(2025, 1, 1, 0, 0, 0)
        time_calls = [0]

        def mock_now():
            call_num = time_calls[0]
            time_calls[0] += 1
            # Calls: 0=init, 1=try_acquire, 2=available_tokens
            if call_num < 2:
                return base_time
            else:
                # After 3 seconds, refill 3 tokens
                return base_time + timedelta(seconds=3)

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=mock_now,
            sleep_fn=lambda s: None
        )

        # Consume 7 tokens
        limiter.try_acquire(7)

        # After 3 seconds, should have ~6 tokens (3 remaining + 3 refilled)
        assert limiter.available_tokens >= 5.5  # 3 remaining + 3 refilled - some buffer

    def test_available_tokens_capped_at_burst(self):
        """Test that available tokens never exceed burst capacity.

        Arrange:
            - RateLimiter with long time passage
        Act:
            - Wait long enough to fill bucket multiple times
        Assert:
            - Available tokens should be capped at burst
        """
        rate_limit = RateLimit(per_minute=60, burst=10)

        time_sequence = [
            datetime(2025, 1, 1, 0, 0, 0),
            datetime(2025, 1, 1, 0, 5, 0),  # +5 minutes
        ]
        time_index = [0]

        def mock_now():
            current = time_sequence[time_index[0]]
            time_index[0] = min(time_index[0] + 1, len(time_sequence) - 1)
            return current

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=mock_now,
            sleep_fn=lambda s: None
        )

        # Even after 5 minutes, should be capped at burst
        assert limiter.available_tokens == 10.0


class TestTokenRefill:
    """Test token refill calculation logic."""

    def test_refill_rate_calculation(self):
        """Test that refill rate is per_minute/60.

        Arrange:
            - RateLimiter with per_minute=60
        Act:
            - Calculate expected tokens after 1 second
        Assert:
            - Should be 1 token per second
        """
        rate_limit = RateLimit(per_minute=60, burst=10)

        base_time = datetime(2025, 1, 1, 0, 0, 0)
        time_calls = [0]

        def mock_now():
            call_num = time_calls[0]
            time_calls[0] += 1
            # Calls: 0=init, 1=try_acquire, 2=available_tokens
            if call_num < 2:
                return base_time
            else:
                # After 1 second
                return base_time + timedelta(seconds=1)

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=mock_now,
            sleep_fn=lambda s: None
        )

        # Consume all tokens
        limiter.try_acquire(10)

        # After 1 second, should have 1 token
        assert limiter.available_tokens >= 0.99  # Allow small floating point error

    def test_refill_after_5_seconds(self):
        """Test correct refill amount after 5 seconds.

        Arrange:
            - RateLimiter with per_minute=120 (2 tokens/second)
        Act:
            - Wait 5 seconds
        Assert:
            - Should refill 10 tokens
        """
        rate_limit = RateLimit(per_minute=120, burst=20)

        base_time = datetime(2025, 1, 1, 0, 0, 0)
        time_calls = [0]

        def mock_now():
            call_num = time_calls[0]
            time_calls[0] += 1
            # Calls: 0=init, 1=try_acquire, 2=available_tokens
            if call_num < 2:
                return base_time
            else:
                # After 5 seconds
                return base_time + timedelta(seconds=5)

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=mock_now,
            sleep_fn=lambda s: None
        )

        # Consume all tokens
        limiter.try_acquire(20)

        # After 5 seconds, should have 10 tokens (2 per second * 5)
        assert abs(limiter.available_tokens - 10.0) < 0.01

    def test_refill_with_fractional_seconds(self):
        """Test refill handles fractional seconds correctly.

        Arrange:
            - RateLimiter with per_minute=60
        Act:
            - Wait 2.5 seconds
        Assert:
            - Should refill 2.5 tokens
        """
        rate_limit = RateLimit(per_minute=60, burst=10)

        base_time = datetime(2025, 1, 1, 0, 0, 0)
        time_calls = [0]

        def mock_now():
            call_num = time_calls[0]
            time_calls[0] += 1
            # Calls: 0=init, 1=try_acquire, 2=available_tokens
            if call_num < 2:
                return base_time
            else:
                # After 2.5 seconds
                return base_time + timedelta(seconds=2.5)

        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=mock_now,
            sleep_fn=lambda s: None
        )

        # Consume all tokens
        limiter.try_acquire(10)

        # After 2.5 seconds, should have 2.5 tokens
        assert abs(limiter.available_tokens - 2.5) < 0.01


class TestThreadSafety:
    """Test thread safety of RateLimiter."""

    def test_concurrent_acquires(self):
        """Test that concurrent acquires are thread-safe.

        Arrange:
            - RateLimiter with real time
            - Multiple threads acquiring tokens
        Act:
            - Launch concurrent acquire operations
        Assert:
            - Total tokens consumed should equal sum of all acquires
            - No race conditions or lost tokens
        """
        rate_limit = RateLimit(per_minute=6000, burst=100)
        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=datetime.now,
            sleep_fn=time.sleep
        )

        results = []

        def acquire_tokens(count):
            try:
                limiter.acquire(count)
                results.append(("success", count))
            except Exception as e:
                results.append(("error", str(e)))

        # Launch 10 threads, each acquiring 5 tokens
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=acquire_tokens, args=(5,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert len(results) == 10
        assert all(status == "success" for status, _ in results)

        # Total consumed should be 50 tokens
        total_consumed = sum(count for status, count in results if status == "success")
        assert total_consumed == 50

        # Remaining should be ~50 (allowing for some refill during execution)
        assert limiter.available_tokens >= 45

    def test_concurrent_try_acquires(self):
        """Test that concurrent try_acquires are thread-safe.

        Arrange:
            - RateLimiter with limited tokens
            - Multiple threads trying to acquire
        Act:
            - Launch concurrent try_acquire operations
        Assert:
            - Total successful acquires should not exceed available tokens
        """
        rate_limit = RateLimit(per_minute=6000, burst=20)
        limiter = RateLimiter(
            rate_limit=rate_limit,
            now_fn=datetime.now,
            sleep_fn=time.sleep
        )

        results = []

        def try_acquire_tokens(count):
            success = limiter.try_acquire(count)
            results.append(success)

        # Launch 10 threads, each trying to acquire 5 tokens
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=try_acquire_tokens, args=(5,))
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should have exactly 10 results
        assert len(results) == 10

        # At most 4 threads should succeed (20 tokens / 5 per thread)
        successful_count = sum(1 for r in results if r is True)
        assert successful_count <= 4
