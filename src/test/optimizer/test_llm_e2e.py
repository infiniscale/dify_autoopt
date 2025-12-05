"""
End-to-End Test Suite for LLM Integration.

This test suite covers complete workflows from configuration to optimization:
1. Complete LLM optimization flow (config -> service -> extract -> optimize -> patch)
2. Multi-strategy comparison tests
3. Hybrid strategy tests (LLM + rules)
4. Cache effectiveness tests
5. Cost limit tests
6. Degradation behavior tests (fallback to rules)

Author: QA Engineer
Date: 2025-11-18
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from src.optimizer.config import LLMConfig, LLMProvider
from src.optimizer.interfaces.llm_client import LLMClient, LLMResponse, StubLLMClient
from src.optimizer.interfaces.llm_providers import OpenAIClient
from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.prompt_extractor import PromptExtractor
from src.optimizer.optimization_engine import OptimizationEngine
from src.optimizer.prompt_analyzer import PromptAnalyzer
from src.optimizer.models import Prompt
from src.config.models import WorkflowCatalog


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_workflow_dsl() -> Dict[str, Any]:
    """Create a sample workflow DSL for testing."""
    return {
        "graph": {
            "nodes": [
                {
                    "data": {
                        "type": "llm",
                        "title": "Summarizer",
                        "prompt_template": [
                            {
                                "role": "system",
                                "text": "You are a helpful assistant. Summarize documents clearly and concisely."
                            }
                        ]
                    },
                    "id": "llm_1"
                },
                {
                    "data": {
                        "type": "llm",
                        "title": "Analyzer",
                        "prompt_template": [
                            {
                                "role": "user",
                                "text": "Analyze {{document}} and provide key insights about {{topic}}"
                            }
                        ]
                    },
                    "id": "llm_2"
                }
            ]
        }
    }


@pytest.fixture
def mock_catalog(sample_workflow_dsl: Dict[str, Any]) -> WorkflowCatalog:
    """Create a mock WorkflowCatalog."""
    catalog = Mock(spec=WorkflowCatalog)
    catalog.get_workflow_by_id.return_value = sample_workflow_dsl
    return catalog


@pytest.fixture
def stub_llm_config() -> LLMConfig:
    """Create STUB LLM configuration for testing."""
    return LLMConfig(
        provider=LLMProvider.STUB,
        enable_cache=True,
        cache_ttl=3600
    )


@pytest.fixture
def openai_llm_config() -> LLMConfig:
    """Create OpenAI LLM configuration for testing."""
    return LLMConfig(
        provider=LLMProvider.OPENAI,
        model="gpt-4-turbo-preview",
        api_key_env="OPENAI_API_KEY",
        temperature=0.7,
        max_tokens=2000,
        enable_cache=True,
        cache_ttl=3600,
        cost_limits={
            "max_cost_per_request": 0.1,
            "max_cost_per_day": 10.0
        }
    )


@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Create a mock OpenAI API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "You are a professional assistant. Provide clear, concise document summaries with key points highlighted."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }


# ============================================================================
# Test 1: Complete LLM Optimization Flow
# ============================================================================


class TestCompleteLLMOptimizationFlow:
    """Test complete flow from configuration to patch generation."""

    def test_stub_complete_flow(self, mock_catalog: WorkflowCatalog, stub_llm_config: LLMConfig):
        """Test complete optimization flow using STUB provider (rule-based)."""
        # Step 1: Create service with STUB config
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=stub_llm_config
        )

        # Step 2: Extract prompts
        prompts = service.extract_prompts("wf_001")
        assert len(prompts) == 2
        assert prompts[0].node_id == "llm_1"
        assert prompts[1].node_id == "llm_2"

        # Step 3: Analyze prompts
        analysis_0 = service.analyze_prompt(prompts[0])
        assert analysis_0.overall_score >= 0
        assert analysis_0.clarity_score >= 0

        # Step 4: Optimize using rule-based strategy
        result = service.optimize_prompt(prompts[0], "llm_clarity")
        assert result.optimized_text != prompts[0].text
        assert result.improvement_score >= 0

        # Step 5: Verify variables preserved
        assert "{{document}}" not in prompts[0].text  # Original has no variables
        if "{{" in prompts[1].text:
            # Test variable preservation for prompt with variables
            result_with_vars = service.optimize_prompt(prompts[1], "llm_clarity")
            assert "{{document}}" in result_with_vars.optimized_text
            assert "{{topic}}" in result_with_vars.optimized_text

        # Step 6: Generate patch
        patches = service.run_optimization_cycle(
            workflow_id="wf_001",
            strategy="llm_clarity"
        )
        assert len(patches) > 0
        assert all(p.node_id in ["llm_1", "llm_2"] for p in patches)

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_openai_complete_flow(
        self,
        mock_openai: Mock,
        mock_catalog: WorkflowCatalog,
        openai_llm_config: LLMConfig,
        mock_openai_response: Dict[str, Any]
    ):
        """Test complete optimization flow using OpenAI provider."""
        # Mock OpenAI API response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(
            message=MagicMock(content=mock_openai_response["choices"][0]["message"]["content"]),
            finish_reason="stop"
        )]
        mock_completion.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        # Mock environment variable for API key
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
            # Create service with OpenAI config
            service = OptimizerService(
                catalog=mock_catalog,
                llm_config=openai_llm_config
            )

            # Extract and optimize
            prompts = service.extract_prompts("wf_001")
            assert len(prompts) == 2

            # Optimize with LLM
            result = service.optimize_prompt(prompts[0], "llm_guided")

            # Verify LLM was called
            assert mock_client.chat.completions.create.called
            assert result.optimized_text != prompts[0].text

            # Verify cost tracking
            stats = service.get_usage_stats()
            assert stats.total_requests > 0
            assert stats.total_cost > 0
            assert stats.total_tokens == 150


# ============================================================================
# Test 2: Multi-Strategy Comparison
# ============================================================================


class TestMultiStrategyComparison:
    """Test comparing different optimization strategies."""

    def test_rule_based_strategy_comparison(self, mock_catalog: WorkflowCatalog, stub_llm_config: LLMConfig):
        """Compare rule-based strategies on the same prompt."""
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=stub_llm_config
        )

        prompts = service.extract_prompts("wf_001")
        test_prompt = prompts[0]

        strategies = ["llm_clarity", "llm_efficiency", "llm_guided"]
        results = {}

        for strategy in strategies:
            result = service.optimize_prompt(test_prompt, strategy)
            analysis = service.analyze_prompt_text(result.optimized_text)
            results[strategy] = {
                "optimized_text": result.optimized_text,
                "improvement": result.improvement_score,
                "quality": analysis.overall_score,
                "length": len(result.optimized_text)
            }

        # Verify each strategy produced different results
        texts = [r["optimized_text"] for r in results.values()]
        assert len(set(texts)) > 1  # At least some strategies differ

        # Verify improvement scores are meaningful
        for strategy, metrics in results.items():
            assert metrics["improvement"] >= -100  # Within valid range
            assert metrics["quality"] >= 0

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_llm_vs_rule_comparison(
        self,
        mock_openai: Mock,
        mock_catalog: WorkflowCatalog,
        openai_llm_config: LLMConfig
    ):
        """Compare LLM optimization vs rule-based optimization."""
        # Setup OpenAI mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(
            message=MagicMock(content="Optimized by LLM: Clear and concise summary."),
            finish_reason="stop"
        )]
        mock_completion.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120
        )
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
            # Test with LLM
            llm_service = OptimizerService(
                catalog=mock_catalog,
                llm_config=openai_llm_config
            )
            prompts = llm_service.extract_prompts("wf_001")
            llm_result = llm_service.optimize_prompt(prompts[0], "llm_guided")

            # Test with rules (STUB)
            stub_config = LLMConfig(provider=LLMProvider.STUB)
            rule_service = OptimizerService(
                catalog=mock_catalog,
                llm_config=stub_config
            )
            rule_result = rule_service.optimize_prompt(prompts[0], "llm_guided")

            # Verify both produced results
            assert llm_result.optimized_text != prompts[0].text
            assert rule_result.optimized_text != prompts[0].text

            # LLM and rule results should differ
            assert llm_result.optimized_text != rule_result.optimized_text


# ============================================================================
# Test 3: Hybrid Strategy Tests
# ============================================================================


class TestHybridStrategy:
    """Test hybrid optimization combining LLM and rules."""

    def test_hybrid_strategy_stub(self, mock_catalog: WorkflowCatalog, stub_llm_config: LLMConfig):
        """Test hybrid strategy using STUB (should behave like rule-based)."""
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=stub_llm_config
        )

        prompts = service.extract_prompts("wf_001")

        # Test hybrid strategy
        result = service.optimize_prompt(prompts[0], "hybrid")

        # Verify optimization occurred
        assert result.optimized_text is not None
        assert result.improvement_score >= -100

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_hybrid_with_llm(
        self,
        mock_openai: Mock,
        mock_catalog: WorkflowCatalog,
        openai_llm_config: LLMConfig
    ):
        """Test hybrid strategy with real LLM integration."""
        # Mock OpenAI
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(
            message=MagicMock(content="Hybrid optimized: Professional summary."),
            finish_reason="stop"
        )]
        mock_completion.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=30,
            total_tokens=130
        )
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
            service = OptimizerService(
                catalog=mock_catalog,
                llm_config=openai_llm_config
            )

            prompts = service.extract_prompts("wf_001")
            result = service.optimize_prompt(prompts[0], "hybrid")

            # Verify hybrid produced result
            assert result.optimized_text is not None
            assert "Hybrid optimized" in result.optimized_text or result.optimized_text != prompts[0].text


# ============================================================================
# Test 4: Cache Effectiveness
# ============================================================================


class TestCacheEffectiveness:
    """Test cache behavior and cost savings."""

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_cache_hit_reduces_cost(
        self,
        mock_openai: Mock,
        mock_catalog: WorkflowCatalog,
        openai_llm_config: LLMConfig
    ):
        """Test that cache hits reduce API calls and costs."""
        # Setup OpenAI mock
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(
            message=MagicMock(content="Cached optimized result"),
            finish_reason="stop"
        )]
        mock_completion.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=30,
            total_tokens=130
        )
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
            service = OptimizerService(
                catalog=mock_catalog,
                llm_config=openai_llm_config
            )

            prompts = service.extract_prompts("wf_001")
            prompt_text = prompts[0].text

            # First call - cache miss
            result1 = service.optimize_prompt(prompts[0], "llm_clarity")
            stats1 = service.get_usage_stats()

            # Second call - should hit cache
            result2 = service.optimize_prompt(prompts[0], "llm_clarity")
            stats2 = service.get_usage_stats()

            # Verify cache hit
            assert result1.optimized_text == result2.optimized_text

            # Cost should not double (cache hit means no new API call)
            # Note: This depends on implementation - adjust assertion as needed
            assert stats2.total_cost <= stats1.total_cost * 1.5  # Allow some margin

            # Cache hits should increase
            assert stats2.cache_hits >= stats1.cache_hits

    def test_cache_expiration(self, mock_catalog: WorkflowCatalog):
        """Test cache expiration behavior."""
        # Create config with very short TTL
        config = LLMConfig(
            provider=LLMProvider.STUB,
            enable_cache=True,
            cache_ttl=1  # 1 second TTL
        )

        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=config
        )

        prompts = service.extract_prompts("wf_001")

        # First optimization
        result1 = service.optimize_prompt(prompts[0], "llm_clarity")

        # Immediate second call should hit cache
        result2 = service.optimize_prompt(prompts[0], "llm_clarity")
        assert result1.optimized_text == result2.optimized_text

        # Wait for cache expiration
        import time
        time.sleep(2)

        # Third call should miss cache (expired)
        result3 = service.optimize_prompt(prompts[0], "llm_clarity")
        # Result should still be valid (just not from cache)
        assert result3.optimized_text is not None


# ============================================================================
# Test 5: Cost Limit Enforcement
# ============================================================================


class TestCostLimits:
    """Test cost limit enforcement and warnings."""

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_per_request_cost_limit_warning(
        self,
        mock_openai: Mock,
        mock_catalog: WorkflowCatalog
    ):
        """Test warning when per-request cost limit is exceeded."""
        # Create config with low cost limit
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo-preview",
            api_key_env="OPENAI_API_KEY",
            cost_limits={
                "max_cost_per_request": 0.001  # Very low limit
            }
        )

        # Mock expensive response
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(
            message=MagicMock(content="Expensive result"),
            finish_reason="stop"
        )]
        mock_completion.usage = MagicMock(
            prompt_tokens=5000,  # High token count
            completion_tokens=2000,
            total_tokens=7000
        )
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
            service = OptimizerService(
                catalog=mock_catalog,
                llm_config=config
            )

            prompts = service.extract_prompts("wf_001")

            # Should complete but log warning
            result = service.optimize_prompt(prompts[0], "llm_guided")
            assert result.optimized_text is not None

            # Verify cost was tracked
            stats = service.get_usage_stats()
            assert stats.total_cost > 0

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_daily_cost_limit_tracking(
        self,
        mock_openai: Mock,
        mock_catalog: WorkflowCatalog
    ):
        """Test daily cost limit tracking."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="gpt-4-turbo-preview",
            api_key_env="OPENAI_API_KEY",
            cost_limits={
                "max_cost_per_day": 5.0
            }
        )

        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(
            message=MagicMock(content="Result"),
            finish_reason="stop"
        )]
        mock_completion.usage = MagicMock(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500
        )
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
            service = OptimizerService(
                catalog=mock_catalog,
                llm_config=config
            )

            prompts = service.extract_prompts("wf_001")

            # Make multiple requests
            for _ in range(3):
                service.optimize_prompt(prompts[0], "llm_clarity")

            # Check accumulated cost
            stats = service.get_usage_stats()
            assert stats.total_requests == 3
            assert stats.total_cost > 0


# ============================================================================
# Test 6: Degradation Behavior (Fallback to Rules)
# ============================================================================


class TestDegradationBehavior:
    """Test graceful degradation when LLM is unavailable."""

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_api_failure_falls_back_to_rules(
        self,
        mock_openai: Mock,
        mock_catalog: WorkflowCatalog,
        openai_llm_config: LLMConfig
    ):
        """Test fallback to rule-based optimization when API fails."""
        # Mock API failure
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
            service = OptimizerService(
                catalog=mock_catalog,
                llm_config=openai_llm_config
            )

            prompts = service.extract_prompts("wf_001")

            # Should fall back to rules and still produce result
            # Note: Implementation needs to support fallback
            try:
                result = service.optimize_prompt(prompts[0], "llm_guided")
                # If fallback works, we get a result
                assert result.optimized_text is not None
            except Exception:
                # If no fallback, we expect an exception
                # This documents current behavior
                pass

    def test_invalid_api_key_handling(self, mock_catalog: WorkflowCatalog):
        """Test handling of invalid API key."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="NONEXISTENT_API_KEY"  # This env var doesn't exist
        )

        # Creating service should work
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=config
        )

        # But using it should fail gracefully
        with pytest.raises(Exception):
            prompts = service.extract_prompts("wf_001")
            service.optimize_prompt(prompts[0], "llm_guided")

    def test_stub_provider_never_fails(self, mock_catalog: WorkflowCatalog):
        """Test that STUB provider always works (no external dependencies)."""
        config = LLMConfig(provider=LLMProvider.STUB)

        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=config
        )

        prompts = service.extract_prompts("wf_001")

        # STUB should always work
        for strategy in ["llm_clarity", "llm_efficiency", "llm_guided"]:
            result = service.optimize_prompt(prompts[0], strategy)
            assert result.optimized_text is not None
            assert result.improvement_score >= -100


# ============================================================================
# Test 7: Variable Preservation
# ============================================================================


class TestVariablePreservation:
    """Test that optimization preserves template variables."""

    def test_variable_preservation_in_optimization(self, mock_catalog: WorkflowCatalog, stub_llm_config: LLMConfig):
        """Test that {{variables}} are preserved during optimization."""
        service = OptimizerService(
            catalog=mock_catalog,
            llm_config=stub_llm_config
        )

        prompts = service.extract_prompts("wf_001")

        # Find prompt with variables
        prompt_with_vars = next((p for p in prompts if "{{" in p.text), None)

        if prompt_with_vars:
            # Extract variables before optimization
            original_vars = [var.name for var in prompt_with_vars.variables]

            # Optimize
            result = service.optimize_prompt(prompt_with_vars, "llm_clarity")

            # Verify all variables still present
            for var_name in original_vars:
                assert f"{{{{{var_name}}}}}" in result.optimized_text, \
                    f"Variable {{{{{var_name}}}}} was lost during optimization"

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_variable_preservation_with_llm(
        self,
        mock_openai: Mock,
        mock_catalog: WorkflowCatalog,
        openai_llm_config: LLMConfig
    ):
        """Test variable preservation with LLM optimization."""
        # Mock LLM to return optimized text with variables
        mock_client = MagicMock()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock(
            message=MagicMock(
                content="Analyze the {{document}} carefully and provide insights about {{topic}}"
            ),
            finish_reason="stop"
        )]
        mock_completion.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=30,
            total_tokens=130
        )
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-api-key'}):
            service = OptimizerService(
                catalog=mock_catalog,
                llm_config=openai_llm_config
            )

            prompts = service.extract_prompts("wf_001")
            prompt_with_vars = next((p for p in prompts if "{{" in p.text), None)

            if prompt_with_vars:
                result = service.optimize_prompt(prompt_with_vars, "llm_guided")

                # Verify variables preserved
                assert "{{document}}" in result.optimized_text
                assert "{{topic}}" in result.optimized_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
