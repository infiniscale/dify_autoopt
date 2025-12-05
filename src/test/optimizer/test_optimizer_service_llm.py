"""
Test Suite for OptimizerService LLM Integration.

Tests the LLM configuration and LLM-driven optimization features
in OptimizerService, including:
- LLM client creation from config
- LLM strategy support
- LLM usage statistics
- Backward compatibility
- Graceful degradation

Date: 2025-11-18
Author: backend-developer
"""

import os
from typing import Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
from loguru import logger

from src.config.models import WorkflowCatalog
from src.optimizer import (
    LLMConfig,
    LLMProvider,
    OptimizerService,
    optimize_workflow,
    OptimizationConfig,
    OptimizationStrategy,
)
from src.optimizer.interfaces.llm_client import LLMClient, LLMResponse, UsageStats
from src.optimizer.interfaces.llm_providers import OpenAIClient


@pytest.fixture
def mock_catalog():
    """Create mock workflow catalog."""
    catalog = MagicMock(spec=WorkflowCatalog)
    return catalog


@pytest.fixture
def mock_llm_client():
    """Create mock LLM client."""
    client = MagicMock(spec=LLMClient)

    # Mock optimize_prompt response
    client.optimize_prompt.return_value = LLMResponse(
        content="Optimized prompt text",
        tokens_used=100,
        cost=0.001,
        model="gpt-4-turbo-preview",
        provider="openai",
        latency_ms=200.0,
        cached=False,
        metadata={}
    )

    # Mock get_usage_stats
    client.get_usage_stats.return_value = UsageStats(
        total_requests=5,
        total_tokens=500,
        total_cost=0.005,
        cache_hits=2,
        cache_misses=3,
        average_latency_ms=150.0
    )

    return client


class TestOptimizerServiceLLMInit:
    """Test OptimizerService initialization with LLM configuration."""

    def test_init_with_stub_provider(self, mock_catalog):
        """Test initialization with STUB provider (rule-based)."""
        llm_config = LLMConfig(provider=LLMProvider.STUB)

        service = OptimizerService(catalog=mock_catalog, llm_config=llm_config)

        assert service._llm_config == llm_config
        assert service._llm_client is not None
        # Should be StubLLMClient
        from src.optimizer.interfaces.llm_client import StubLLMClient
        assert isinstance(service._llm_client, StubLLMClient)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.optimizer.optimizer_service.OpenAIClient")
    def test_init_with_openai_provider(self, mock_openai_class, mock_catalog):
        """Test initialization with OpenAI provider."""
        mock_client = MagicMock(spec=OpenAIClient)
        mock_openai_class.return_value = mock_client

        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo-preview",
            api_key_env="OPENAI_API_KEY"
        )

        service = OptimizerService(catalog=mock_catalog, llm_config=llm_config)

        assert service._llm_config == llm_config
        assert service._llm_client == mock_client
        mock_openai_class.assert_called_once_with(llm_config)

    def test_init_with_anthropic_provider_fallback(self, mock_catalog):
        """Test initialization with Anthropic provider falls back to STUB."""
        llm_config = LLMConfig(provider=LLMProvider.ANTHROPIC)

        service = OptimizerService(catalog=mock_catalog, llm_config=llm_config)

        # Should fallback to StubLLMClient
        from src.optimizer.interfaces.llm_client import StubLLMClient
        assert isinstance(service._llm_client, StubLLMClient)

    def test_init_with_local_provider_fallback(self, mock_catalog):
        """Test initialization with LOCAL provider falls back to STUB."""
        llm_config = LLMConfig(
            provider=LLMProvider.LOCAL,
            base_url="http://localhost:11434/v1"
        )

        service = OptimizerService(catalog=mock_catalog, llm_config=llm_config)

        # Should fallback to StubLLMClient
        from src.optimizer.interfaces.llm_client import StubLLMClient
        assert isinstance(service._llm_client, StubLLMClient)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    @patch("src.optimizer.optimizer_service.OpenAIClient")
    def test_init_openai_failure_fallback(self, mock_openai_class, mock_catalog):
        """Test OpenAI initialization failure falls back to STUB."""
        # Make OpenAI init raise exception
        mock_openai_class.side_effect = Exception("API connection failed")

        llm_config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY"
        )

        service = OptimizerService(catalog=mock_catalog, llm_config=llm_config)

        # Should fallback to StubLLMClient
        from src.optimizer.interfaces.llm_client import StubLLMClient
        assert isinstance(service._llm_client, StubLLMClient)

    def test_backward_compatibility_explicit_client(self, mock_catalog, mock_llm_client):
        """Test backward compatibility with explicit llm_client parameter."""
        service = OptimizerService(
            catalog=mock_catalog,
            llm_client=mock_llm_client
        )

        assert service._llm_client == mock_llm_client
        assert service._llm_config is None

    def test_backward_compatibility_no_llm(self, mock_catalog):
        """Test backward compatibility with no LLM parameters."""
        service = OptimizerService(catalog=mock_catalog)

        # Should default to StubLLMClient
        from src.optimizer.interfaces.llm_client import StubLLMClient
        assert isinstance(service._llm_client, StubLLMClient)
        assert service._llm_config is None

    def test_priority_explicit_client_over_config(self, mock_catalog, mock_llm_client):
        """Test that explicit client parameter takes priority over config."""
        llm_config = LLMConfig(provider=LLMProvider.STUB)

        service = OptimizerService(
            catalog=mock_catalog,
            llm_client=mock_llm_client,
            llm_config=llm_config
        )

        # Explicit client should be used
        assert service._llm_client == mock_llm_client
        assert service._llm_config is None


class TestOptimizerServiceLLMStats:
    """Test LLM usage statistics methods."""

    def test_get_llm_stats_with_stub_client(self, mock_catalog):
        """Test get_llm_stats returns None for StubLLMClient."""
        service = OptimizerService(catalog=mock_catalog)

        stats = service.get_llm_stats()

        assert stats is None

    def test_get_llm_stats_with_real_client(self, mock_catalog, mock_llm_client):
        """Test get_llm_stats returns statistics for real LLM client."""
        service = OptimizerService(
            catalog=mock_catalog,
            llm_client=mock_llm_client
        )

        stats = service.get_llm_stats()

        assert stats is not None
        assert stats["total_requests"] == 5
        assert stats["total_tokens"] == 500
        assert stats["total_cost"] == 0.005
        assert stats["cache_hits"] == 2
        assert stats["cache_misses"] == 3
        assert stats["cache_hit_rate"] == 0.4  # 2/5
        assert stats["average_latency_ms"] == 150.0

    def test_get_llm_stats_zero_requests(self, mock_catalog, mock_llm_client):
        """Test get_llm_stats handles zero requests correctly."""
        mock_llm_client.get_usage_stats.return_value = UsageStats(
            total_requests=0,
            total_tokens=0,
            total_cost=0.0,
            cache_hits=0,
            cache_misses=0,
            average_latency_ms=0.0
        )

        service = OptimizerService(
            catalog=mock_catalog,
            llm_client=mock_llm_client
        )

        stats = service.get_llm_stats()

        assert stats is not None
        assert stats["cache_hit_rate"] == 0.0  # No division by zero

    def test_get_llm_stats_error_handling(self, mock_catalog, mock_llm_client):
        """Test get_llm_stats handles errors gracefully."""
        mock_llm_client.get_usage_stats.side_effect = Exception("Stats error")

        service = OptimizerService(
            catalog=mock_catalog,
            llm_client=mock_llm_client
        )

        stats = service.get_llm_stats()

        assert stats is None

    def test_reset_llm_stats_with_stub_client(self, mock_catalog):
        """Test reset_llm_stats is no-op for StubLLMClient."""
        service = OptimizerService(catalog=mock_catalog)

        # Should not raise exception
        service.reset_llm_stats()

    def test_reset_llm_stats_with_real_client(self, mock_catalog, mock_llm_client):
        """Test reset_llm_stats calls client's reset_stats."""
        service = OptimizerService(
            catalog=mock_catalog,
            llm_client=mock_llm_client
        )

        service.reset_llm_stats()

        mock_llm_client.reset_stats.assert_called_once()

    def test_reset_llm_stats_error_handling(self, mock_catalog, mock_llm_client):
        """Test reset_llm_stats handles errors gracefully."""
        mock_llm_client.reset_stats.side_effect = Exception("Reset error")

        service = OptimizerService(
            catalog=mock_catalog,
            llm_client=mock_llm_client
        )

        # Should not raise exception
        service.reset_llm_stats()


class TestOptimizeWorkflowConvenienceFunction:
    """Test optimize_workflow convenience function with LLM support."""

    @patch("src.optimizer.OptimizerService")
    def test_optimize_workflow_with_llm_config(self, mock_service_class, mock_catalog):
        """Test optimize_workflow passes llm_config correctly."""
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.run_optimization_cycle.return_value = []

        llm_config = LLMConfig(provider=LLMProvider.STUB)

        result = optimize_workflow(
            workflow_id="wf_001",
            catalog=mock_catalog,
            llm_config=llm_config
        )

        # Verify OptimizerService was created with llm_config
        mock_service_class.assert_called_once()
        call_kwargs = mock_service_class.call_args[1]
        assert call_kwargs["llm_config"] == llm_config
        assert call_kwargs["catalog"] == mock_catalog

    @patch("src.optimizer.OptimizerService")
    def test_optimize_workflow_backward_compatibility(self, mock_service_class, mock_catalog):
        """Test optimize_workflow works without LLM parameters."""
        mock_service = MagicMock()
        mock_service_class.return_value = mock_service
        mock_service.run_optimization_cycle.return_value = []

        result = optimize_workflow(
            workflow_id="wf_001",
            catalog=mock_catalog,
            strategy="clarity_focus"
        )

        # Should work without llm_config
        mock_service_class.assert_called_once()
        mock_service.run_optimization_cycle.assert_called_once()


class TestLLMStrategySupport:
    """Test LLM strategy support in optimization cycle."""

    def test_llm_strategy_with_stub_client_fallback(self, mock_catalog):
        """Test LLM strategy falls back to rule-based when no LLM client."""
        # This is an integration test that verifies the engine handles fallback
        service = OptimizerService(catalog=mock_catalog)

        # Verify StubLLMClient was created
        from src.optimizer.interfaces.llm_client import StubLLMClient
        assert isinstance(service._llm_client, StubLLMClient)

        # The engine should handle LLM strategy fallback internally
        # (tested in optimization_engine tests)


def test_llm_integration_end_to_end():
    """Integration test for LLM configuration flow."""
    # Create LLM config
    llm_config = LLMConfig(provider=LLMProvider.STUB)

    # Create service
    catalog = MagicMock(spec=WorkflowCatalog)
    service = OptimizerService(catalog=catalog, llm_config=llm_config)

    # Verify client was created
    assert service._llm_client is not None
    assert service._llm_config == llm_config

    # Stats should return None for STUB
    stats = service.get_llm_stats()
    assert stats is None

    # Reset should work
    service.reset_llm_stats()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
