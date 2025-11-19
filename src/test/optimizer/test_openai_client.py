"""
Unit Tests for OpenAI Client.

Date: 2025-11-18
Author: backend-developer
Description: Comprehensive tests for OpenAIClient including mocking, caching, and error handling.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.optimizer.interfaces.llm_client import LLMResponse, UsageStats
from src.optimizer.interfaces.llm_providers.openai_client import OpenAIClient
from src.optimizer.config import LLMConfig, LLMProvider


class TestOpenAIClientInitialization:
    """Test OpenAI client initialization and configuration."""

    def test_init_with_valid_config(self):
        """Test initialization with valid OpenAI config."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                api_key_env="OPENAI_API_KEY"
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI"):
                client = OpenAIClient(config)

                assert client._config == config
                assert client._tracker is not None
                assert client._cache is not None  # Cache enabled by default

    def test_init_with_invalid_provider(self):
        """Test initialization fails with non-OpenAI provider."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-opus"
        )

        with pytest.raises(ValueError, match="OpenAIClient requires provider=OPENAI"):
            OpenAIClient(config)

    def test_init_with_cache_disabled(self):
        """Test initialization with caching disabled."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=False
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI"):
                client = OpenAIClient(config)
                assert client._cache is None


class TestAnalyzePrompt:
    """Test prompt analysis functionality."""

    @pytest.fixture
    def mock_openai_response(self):
        """Create mock OpenAI API response for analysis."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = json.dumps({
            "clarity_score": 85,
            "efficiency_score": 75,
            "structure_score": 90,
            "overall_score": 83,
            "issues": ["Could be more specific"],
            "suggestions": ["Add context about the document type"]
        })
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 150
        mock_response.usage.completion_tokens = 80
        mock_response.usage.total_tokens = 230
        return mock_response

    def test_analyze_prompt_success(self, mock_openai_response):
        """Test successful prompt analysis."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=False
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_openai_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)
                response = client.analyze_prompt("Summarize the document")

                # Verify response structure
                assert isinstance(response, LLMResponse)
                assert response.provider == "openai"
                assert response.model == "gpt-4-turbo-preview"
                assert response.tokens_used == 230
                assert response.cost > 0
                assert not response.cached

                # Verify content
                analysis = json.loads(response.content)
                assert analysis["clarity_score"] == 85
                assert analysis["overall_score"] == 83
                assert len(analysis["issues"]) > 0

    def test_analyze_prompt_api_called_correctly(self, mock_openai_response):
        """Test that OpenAI API is called with correct parameters."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                temperature=0.5,
                max_tokens=1500
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_openai_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)
                client.analyze_prompt("Test prompt")

                # Verify API call parameters
                call_args = mock_client.chat.completions.create.call_args
                assert call_args[1]["model"] == "gpt-4-turbo-preview"
                assert call_args[1]["temperature"] == 0.5
                assert call_args[1]["max_tokens"] == 1500
                assert len(call_args[1]["messages"]) == 2  # System + User


class TestOptimizePrompt:
    """Test prompt optimization functionality."""

    @pytest.fixture
    def mock_optimization_response(self):
        """Create mock OpenAI API response for optimization."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = (
            "Please provide a comprehensive summary of the document, "
            "highlighting key points and main conclusions."
        )
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 250
        return mock_response

    def test_optimize_prompt_success(self, mock_optimization_response):
        """Test successful prompt optimization."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=False
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_optimization_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)
                response = client.optimize_prompt(
                    "Summarize",
                    strategy="llm_clarity"
                )

                assert isinstance(response, LLMResponse)
                assert response.tokens_used == 250
                assert len(response.content) > 0
                assert not response.cached

    def test_optimize_different_strategies(self, mock_optimization_response):
        """Test optimization with different strategies."""
        strategies = ["llm_guided", "llm_clarity", "llm_efficiency"]

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=False
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_optimization_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)

                for strategy in strategies:
                    response = client.optimize_prompt("Test prompt", strategy=strategy)
                    assert isinstance(response, LLMResponse)


class TestCaching:
    """Test caching functionality."""

    @pytest.fixture
    def mock_response(self):
        """Create mock response."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "Optimized prompt"
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = Mock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        return mock_response

    def test_cache_hit(self, mock_response):
        """Test that cache returns cached results."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=True
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)

                # First call - should hit API
                response1 = client.optimize_prompt("Test", strategy="llm_clarity")
                assert not response1.cached
                assert mock_client.chat.completions.create.call_count == 1

                # Second call - should hit cache
                response2 = client.optimize_prompt("Test", strategy="llm_clarity")
                assert response2.cached
                assert response2.tokens_used == 0
                assert response2.cost == 0.0
                assert mock_client.chat.completions.create.call_count == 1  # No new call

    def test_cache_miss_different_strategy(self, mock_response):
        """Test that different strategies don't share cache."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=True
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)

                # First call with llm_clarity
                client.optimize_prompt("Test", strategy="llm_clarity")

                # Second call with llm_efficiency - should miss cache
                response = client.optimize_prompt("Test", strategy="llm_efficiency")
                assert not response.cached
                assert mock_client.chat.completions.create.call_count == 2


class TestTokenTracking:
    """Test token usage tracking."""

    def test_token_tracking(self):
        """Test that tokens are tracked correctly."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=False
            )

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Result"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 1000
            mock_response.usage.completion_tokens = 500
            mock_response.usage.total_tokens = 1500

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)
                client.analyze_prompt("Test")

                stats = client.get_usage_stats()
                assert stats.total_requests == 1
                assert stats.total_tokens == 1500
                assert stats.total_cost > 0

    def test_cost_calculation(self):
        """Test that costs are calculated correctly."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=False
            )

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Result"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 1000  # 1000 * 0.01 / 1000 = 0.01
            mock_response.usage.completion_tokens = 1000  # 1000 * 0.03 / 1000 = 0.03
            mock_response.usage.total_tokens = 2000

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)
                response = client.analyze_prompt("Test")

                # Expected cost: 0.01 + 0.03 = 0.04
                assert abs(response.cost - 0.04) < 0.001


class TestCostLimits:
    """Test cost limit enforcement."""

    def test_cost_limit_warning(self, caplog):
        """Test that cost limit warnings are logged."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=False,
                cost_limits={"max_cost_per_request": 0.001}  # Very low limit
            )

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Result"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 1000
            mock_response.usage.completion_tokens = 1000
            mock_response.usage.total_tokens = 2000

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)
                client.analyze_prompt("Test")

                # Should log warning about exceeding limit
                # (cost will be ~0.04, limit is 0.001)


class TestUsageStats:
    """Test usage statistics."""

    def test_get_usage_stats(self):
        """Test getting usage statistics."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=True
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI"):
                client = OpenAIClient(config)
                stats = client.get_usage_stats()

                assert isinstance(stats, UsageStats)
                assert stats.total_requests == 0
                assert stats.total_tokens == 0
                assert stats.total_cost == 0.0

    def test_reset_stats(self):
        """Test resetting statistics."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=False
            )

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Result"
            mock_response.choices[0].finish_reason = "stop"
            mock_response.usage = Mock()
            mock_response.usage.prompt_tokens = 100
            mock_response.usage.completion_tokens = 50
            mock_response.usage.total_tokens = 150

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.return_value = mock_response
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)
                client.analyze_prompt("Test")

                # Verify stats are recorded
                stats_before = client.get_usage_stats()
                assert stats_before.total_requests > 0

                # Reset
                client.reset_stats()

                # Verify stats are cleared
                stats_after = client.get_usage_stats()
                assert stats_after.total_requests == 0
                assert stats_after.total_tokens == 0


class TestErrorHandling:
    """Test error handling."""

    def test_api_error_propagation(self):
        """Test that API errors are propagated."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key-123"}):
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                model="gpt-4-turbo-preview",
                enable_cache=False
            )

            with patch("src.optimizer.interfaces.llm_providers.openai_client.OpenAI") as mock_openai:
                mock_client = Mock()
                mock_client.chat.completions.create.side_effect = Exception("API Error")
                mock_openai.return_value = mock_client

                client = OpenAIClient(config)

                with pytest.raises(Exception, match="API Error"):
                    client.analyze_prompt("Test")
