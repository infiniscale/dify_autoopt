"""
Error Handling Test Suite for LLM Integration.

This test suite validates error handling and exception scenarios:
1. API error handling (auth, rate limit, timeout, network)
2. Configuration error handling
3. Degradation scenarios (fallback logic)
4. Data validation errors

Author: QA Engineer
Date: 2025-11-18
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from src.optimizer.config import LLMConfig, LLMProvider
from src.optimizer.interfaces.llm_providers import OpenAIClient
from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.exceptions import (
    OptimizerError,
    InvalidStrategyError,
    OptimizationFailedError,
)
from src.config.models import WorkflowCatalog
from openai import AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def sample_workflow_dsl():
    """Sample workflow for testing."""
    return {
        "graph": {
            "nodes": [
                {
                    "data": {
                        "type": "llm",
                        "title": "Test",
                        "prompt_template": [
                            {"role": "system", "text": "Test prompt"}
                        ]
                    },
                    "id": "llm_1"
                }
            ]
        }
    }


@pytest.fixture
def mock_catalog(sample_workflow_dsl):
    """Mock workflow catalog."""
    catalog = Mock(spec=WorkflowCatalog)
    catalog.get_workflow_by_id.return_value = sample_workflow_dsl
    return catalog


# ============================================================================
# Test 1: API Error Handling
# ============================================================================


class TestAPIErrorHandling:
    """Test handling of various API errors."""

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_authentication_failure(self, mock_openai, mock_catalog):
        """Test handling of authentication errors."""
        # Mock authentication error
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = AuthenticationError(
            "Invalid API key",
            response=MagicMock(status_code=401),
            body=None
        )
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY"
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'invalid-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            # Should raise error (or handle gracefully)
            with pytest.raises((AuthenticationError, Exception)):
                service.optimize_prompt(prompts[0], "llm_guided")

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_rate_limit_handling(self, mock_openai, mock_catalog):
        """Test handling of rate limit errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limit exceeded",
            response=MagicMock(status_code=429),
            body=None
        )
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            max_retries=0  # Disable retries for faster test
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            with pytest.raises((RateLimitError, Exception)):
                service.optimize_prompt(prompts[0], "llm_guided")

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_timeout_handling(self, mock_openai, mock_catalog):
        """Test handling of timeout errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APITimeoutError(
            "Request timeout"
        )
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            timeout=1,
            max_retries=0
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            with pytest.raises((APITimeoutError, Exception)):
                service.optimize_prompt(prompts[0], "llm_guided")

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_network_error_handling(self, mock_openai, mock_catalog):
        """Test handling of network connection errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIConnectionError(
            "Connection failed"
        )
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            max_retries=0
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            with pytest.raises((APIConnectionError, Exception)):
                service.optimize_prompt(prompts[0], "llm_guided")

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_generic_api_error(self, mock_openai, mock_catalog):
        """Test handling of generic API errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception(
            "Unexpected API error"
        )
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY"
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            with pytest.raises(Exception):
                service.optimize_prompt(prompts[0], "llm_guided")


# ============================================================================
# Test 2: Configuration Error Handling
# ============================================================================


class TestConfigurationErrors:
    """Test handling of configuration errors."""

    def test_invalid_api_key_env_variable(self, mock_catalog):
        """Test handling when API key environment variable is missing."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="NONEXISTENT_KEY"
        )

        # Should fail when trying to validate or use config
        with pytest.raises(ValueError, match="API key not found"):
            config.get_api_key()

    def test_missing_configuration(self):
        """Test handling of missing required configuration."""
        # LOCAL provider requires base_url
        config = LLMConfig(
            provider=LLMProvider.LOCAL,
            base_url=None
        )

        with pytest.raises(ValueError, match="base_url is required"):
            config.validate_config()

    def test_invalid_provider(self):
        """Test handling of invalid provider value."""
        # Pydantic should catch this at creation time
        with pytest.raises((ValueError, TypeError)):
            LLMConfig(provider="invalid_provider")

    def test_invalid_temperature(self):
        """Test validation of temperature parameter."""
        with pytest.raises(ValueError, match="Temperature"):
            LLMConfig(temperature=3.0)  # > 2.0

        with pytest.raises(ValueError, match="Temperature"):
            LLMConfig(temperature=-0.5)  # < 0.0

    def test_invalid_max_tokens(self):
        """Test validation of max_tokens parameter."""
        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig(max_tokens=0)  # Must be positive

        with pytest.raises(ValueError, match="max_tokens"):
            LLMConfig(max_tokens=200000)  # Too large

    def test_invalid_cache_ttl(self):
        """Test validation of cache_ttl parameter."""
        with pytest.raises(ValueError, match="cache_ttl"):
            LLMConfig(cache_ttl=0)

        with pytest.raises(ValueError, match="cache_ttl"):
            LLMConfig(cache_ttl=-100)

    def test_invalid_cost_limits(self):
        """Test validation of cost_limits parameter."""
        with pytest.raises(ValueError, match="Cost limit"):
            LLMConfig(cost_limits={"max_cost_per_request": -1.0})

        with pytest.raises(ValueError, match="Cost limit"):
            LLMConfig(cost_limits={"max_cost_per_request": "invalid"})

    def test_invalid_cost_limit_keys(self):
        """Test validation of cost_limits keys."""
        config = LLMConfig(
            provider=LLMProvider.STUB,
            cost_limits={"invalid_key": 10.0}
        )

        with pytest.raises(ValueError, match="Invalid cost limit keys"):
            config.validate_config()

    def test_empty_model_name(self):
        """Test validation of empty model name."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model="",
            api_key_env="OPENAI_API_KEY"
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            with pytest.raises(ValueError, match="Model name cannot be empty"):
                config.validate_config()

    def test_wrong_provider_for_client(self):
        """Test creating OpenAI client with wrong provider config."""
        config = LLMConfig(provider=LLMProvider.STUB)

        with pytest.raises(ValueError, match="requires provider=OPENAI"):
            OpenAIClient(config)


# ============================================================================
# Test 3: Degradation Scenarios
# ============================================================================


class TestDegradationScenarios:
    """Test graceful degradation and fallback behavior."""

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_fallback_on_api_failure(self, mock_openai, mock_catalog):
        """Test system behavior when API consistently fails."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API down")
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            max_retries=0
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            # Should fail (no automatic fallback in current implementation)
            with pytest.raises(Exception):
                service.optimize_prompt(prompts[0], "llm_guided")

    def test_stub_provider_always_works(self, mock_catalog):
        """Test that STUB provider works even without network."""
        config = LLMConfig(provider=LLMProvider.STUB)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")

        # Should always succeed
        result = service.optimize_prompt(prompts[0], "llm_clarity")
        assert result.optimized_text is not None

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_partial_response_handling(self, mock_openai, mock_catalog):
        """Test handling of incomplete/partial API responses."""
        mock_client = MagicMock()

        # Mock response with missing fields
        mock_completion = MagicMock()
        mock_completion.choices = []  # Empty choices
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY"
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            # Should handle gracefully
            with pytest.raises((IndexError, AttributeError, Exception)):
                service.optimize_prompt(prompts[0], "llm_guided")

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_malformed_json_response(self, mock_openai, mock_catalog):
        """Test handling of malformed JSON in analysis responses."""
        mock_client = MagicMock()
        mock_completion = MagicMock()

        # Return invalid JSON
        mock_completion.choices = [MagicMock(
            message=MagicMock(content="This is not valid JSON at all"),
            finish_reason="stop"
        )]
        mock_completion.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        mock_client.chat.completions.create.return_value = mock_completion
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY"
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            # analyze_prompt expects JSON, should handle error
            # Note: Current implementation may not validate JSON
            result = service.analyze_prompt_text(prompts[0].text)
            # Should either parse successfully or handle error


# ============================================================================
# Test 4: Data Validation
# ============================================================================


class TestDataValidation:
    """Test data validation and edge cases."""

    def test_empty_prompt_handling(self, mock_catalog):
        """Test handling of empty prompt text."""
        config = LLMConfig(provider=LLMProvider.STUB)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        # Create prompt with empty text
        from src.optimizer.models import Prompt

        empty_prompt = Prompt(
            id="test",
            workflow_id="wf",
            node_id="node",
            node_type="llm",
            text="",  # Empty
            role="system",
            variables=[],
            context={},
            extracted_at=datetime.now()
        )

        # Should handle gracefully
        result = service.optimize_prompt(empty_prompt, "llm_clarity")
        # Result may be empty or have minimal content
        assert result is not None

    def test_very_long_prompt_handling(self, mock_catalog):
        """Test handling of very long prompts."""
        config = LLMConfig(
            provider=LLMProvider.STUB,
            max_tokens=100  # Small limit
        )
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        from src.optimizer.models import Prompt

        # Create very long prompt
        long_text = "This is a test prompt. " * 1000  # Very long

        long_prompt = Prompt(
            id="test",
            workflow_id="wf",
            node_id="node",
            node_type="llm",
            text=long_text,
            role="system",
            variables=[],
            context={},
            extracted_at=datetime.now()
        )

        # Should handle without errors
        result = service.optimize_prompt(long_prompt, "llm_clarity")
        assert result.optimized_text is not None

    def test_special_characters_in_prompt(self, mock_catalog):
        """Test handling of special characters."""
        config = LLMConfig(provider=LLMProvider.STUB)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        from src.optimizer.models import Prompt

        # Prompt with special characters
        special_text = "Test with special chars: <>&\"'\\n\\t{{var}} 中文 emoji 🎉"

        special_prompt = Prompt(
            id="test",
            workflow_id="wf",
            node_id="node",
            node_type="llm",
            text=special_text,
            role="system",
            variables=[],
            context={},
            extracted_at=datetime.now()
        )

        # Should handle without errors
        result = service.optimize_prompt(special_prompt, "llm_clarity")
        assert result.optimized_text is not None

    def test_invalid_strategy_name(self, mock_catalog):
        """Test error handling for invalid strategy names."""
        config = LLMConfig(provider=LLMProvider.STUB)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")

        # Invalid strategy
        with pytest.raises((InvalidStrategyError, Exception)):
            service.optimize_prompt(prompts[0], "invalid_strategy_name")

    def test_null_prompt_handling(self, mock_catalog):
        """Test handling of None/null prompt."""
        config = LLMConfig(provider=LLMProvider.STUB)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        # Should raise appropriate error
        with pytest.raises((AttributeError, TypeError)):
            service.optimize_prompt(None, "llm_clarity")

    def test_variable_extraction_error(self, mock_catalog):
        """Test handling when variable extraction fails."""
        config = LLMConfig(provider=LLMProvider.STUB)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        from src.optimizer.models import Prompt

        # Malformed variable syntax
        malformed_prompt = Prompt(
            id="test",
            workflow_id="wf",
            node_id="node",
            node_type="llm",
            text="Test with malformed {{var} or {var}} variables",
            role="system",
            variables=[],
            context={},
            extracted_at=datetime.now()
        )

        # Should handle gracefully
        result = service.optimize_prompt(malformed_prompt, "llm_clarity")
        assert result.optimized_text is not None


# ============================================================================
# Test 5: Edge Cases and Boundary Conditions
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_temperature(self, mock_catalog):
        """Test with temperature = 0 (deterministic)."""
        config = LLMConfig(
            provider=LLMProvider.STUB,
            temperature=0.0
        )
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")
        result = service.optimize_prompt(prompts[0], "llm_clarity")

        assert result.optimized_text is not None

    def test_maximum_temperature(self, mock_catalog):
        """Test with temperature = 2.0 (maximum)."""
        config = LLMConfig(
            provider=LLMProvider.STUB,
            temperature=2.0
        )
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")
        result = service.optimize_prompt(prompts[0], "llm_clarity")

        assert result.optimized_text is not None

    def test_minimum_max_tokens(self, mock_catalog):
        """Test with minimum max_tokens."""
        config = LLMConfig(
            provider=LLMProvider.STUB,
            max_tokens=1
        )
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")
        result = service.optimize_prompt(prompts[0], "llm_clarity")

        assert result.optimized_text is not None

    def test_cache_disabled(self, mock_catalog):
        """Test operation with cache disabled."""
        config = LLMConfig(
            provider=LLMProvider.STUB,
            enable_cache=False
        )
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")

        # Multiple calls should not benefit from cache
        result1 = service.optimize_prompt(prompts[0], "llm_clarity")
        result2 = service.optimize_prompt(prompts[0], "llm_clarity")

        # Results should still be valid
        assert result1.optimized_text is not None
        assert result2.optimized_text is not None

    def test_zero_cache_ttl_behavior(self):
        """Test cache behavior with TTL = 0 (should reject)."""
        with pytest.raises(ValueError, match="cache_ttl"):
            LLMConfig(cache_ttl=0)

    def test_zero_max_retries(self, mock_catalog):
        """Test with max_retries = 0 (no retries)."""
        config = LLMConfig(
            provider=LLMProvider.STUB,
            max_retries=0
        )
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")
        result = service.optimize_prompt(prompts[0], "llm_clarity")

        assert result.optimized_text is not None

    def test_maximum_max_retries(self, mock_catalog):
        """Test with maximum max_retries."""
        config = LLMConfig(
            provider=LLMProvider.STUB,
            max_retries=10
        )
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")
        result = service.optimize_prompt(prompts[0], "llm_clarity")

        assert result.optimized_text is not None


# ============================================================================
# Test 6: Error Recovery
# ============================================================================


class TestErrorRecovery:
    """Test error recovery and resilience."""

    @patch('src.optimizer.interfaces.llm_providers.openai_client.OpenAI')
    def test_recovery_after_transient_failure(self, mock_openai, mock_catalog):
        """Test that system recovers after transient failures."""
        mock_client = MagicMock()

        # First call fails, second succeeds
        call_count = [0]

        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise Exception("Transient error")
            else:
                mock_resp = MagicMock()
                mock_resp.choices = [MagicMock(
                    message=MagicMock(content="Success"),
                    finish_reason="stop"
                )]
                mock_resp.usage = MagicMock(
                    prompt_tokens=100,
                    completion_tokens=50,
                    total_tokens=150
                )
                return mock_resp

        mock_client.chat.completions.create.side_effect = side_effect
        mock_openai.return_value = mock_client

        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY"
        )

        with patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'}):
            service = OptimizerService(catalog=mock_catalog, llm_config=config)
            prompts = service.extract_prompts("wf_001")

            # First call fails
            with pytest.raises(Exception):
                service.optimize_prompt(prompts[0], "llm_guided")

            # Second call should succeed
            result = service.optimize_prompt(prompts[0], "llm_guided")
            assert result.optimized_text == "Success"

    def test_stats_tracking_after_errors(self, mock_catalog):
        """Test that statistics are correctly tracked even after errors."""
        config = LLMConfig(provider=LLMProvider.STUB)
        service = OptimizerService(catalog=mock_catalog, llm_config=config)

        prompts = service.extract_prompts("wf_001")

        # Successful optimization
        service.optimize_prompt(prompts[0], "llm_clarity")

        # Attempt with invalid strategy (will fail)
        try:
            service.optimize_prompt(prompts[0], "invalid_strategy")
        except:
            pass

        # Stats should still be valid
        stats = service.get_usage_stats()
        assert stats.total_requests >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
