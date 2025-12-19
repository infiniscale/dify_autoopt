"""
Unit Tests for LLM Configuration Module.

This test suite validates:
1. LLMConfig model creation and validation
2. Configuration loading from dictionaries
3. Configuration loading from environment variables
4. API key security (SecretStr usage)
5. Parameter validation (temperature, max_tokens, etc.)
6. Default configuration
7. Configuration display (with masked API keys)

Author: Backend Developer
Date: 2025-11-18
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pytest
import yaml

from src.optimizer.config import LLMConfig, LLMProvider, LLMConfigLoader


class TestLLMConfig:
    """Test suite for LLMConfig model."""

    def test_default_config_creation(self):
        """Test creating LLMConfig with default values."""
        config = LLMConfig()

        assert config.provider == LLMProvider.STUB
        assert config.model == "gpt-4-turbo-preview"
        assert config.temperature == 0.7
        assert config.max_tokens == 2000
        assert config.enable_cache is True
        assert config.cache_ttl == 86400
        assert config.max_retries == 3
        assert config.timeout == 60

    def test_openai_config_creation(self):
        """Test creating OpenAI configuration."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="OPENAI_API_KEY",
            model="gpt-4-turbo-preview",
            temperature=0.5,
            max_tokens=1500
        )

        assert config.provider == LLMProvider.OPENAI
        assert config.api_key_env == "OPENAI_API_KEY"
        assert config.model == "gpt-4-turbo-preview"
        assert config.temperature == 0.5
        assert config.max_tokens == 1500

    def test_anthropic_config_creation(self):
        """Test creating Anthropic configuration."""
        config = LLMConfig(
            provider=LLMProvider.ANTHROPIC,
            api_key_env="ANTHROPIC_API_KEY",
            model="claude-3-opus-20240229",
            temperature=0.8
        )

        assert config.provider == LLMProvider.ANTHROPIC
        assert config.api_key_env == "ANTHROPIC_API_KEY"
        assert config.model == "claude-3-opus-20240229"
        assert config.temperature == 0.8

    def test_local_config_creation(self):
        """Test creating local LLM configuration."""
        config = LLMConfig(
            provider=LLMProvider.LOCAL,
            api_key_env="",
            model="llama2",
            base_url="http://localhost:11434/v1"
        )

        assert config.provider == LLMProvider.LOCAL
        assert config.model == "llama2"
        assert config.base_url == "http://localhost:11434/v1"

    def test_temperature_validation_valid(self):
        """Test temperature validation with valid values."""
        valid_temperatures = [0.0, 0.5, 1.0, 1.5, 2.0]

        for temp in valid_temperatures:
            config = LLMConfig(temperature=temp)
            assert config.temperature == temp

    def test_temperature_validation_invalid(self):
        """Test temperature validation with invalid values."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig(temperature=-0.1)

        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig(temperature=2.1)

    def test_max_tokens_validation_valid(self):
        """Test max_tokens validation with valid values."""
        valid_tokens = [1, 100, 1000, 4000, 128000]

        for tokens in valid_tokens:
            config = LLMConfig(max_tokens=tokens)
            assert config.max_tokens == tokens

    def test_max_tokens_validation_invalid(self):
        """Test max_tokens validation with invalid values."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig(max_tokens=0)

        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig(max_tokens=-100)

        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig(max_tokens=200000)

    def test_cache_ttl_validation(self):
        """Test cache TTL validation."""
        # Valid TTL
        config = LLMConfig(cache_ttl=3600)
        assert config.cache_ttl == 3600

        # Invalid TTL (zero or negative)
        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig(cache_ttl=-100)

        with pytest.raises(Exception):  # Pydantic ValidationError
            LLMConfig(cache_ttl=0)

    def test_cost_limits_validation_valid(self):
        """Test cost limits validation with valid values."""
        cost_limits = {
            "max_cost_per_request": 0.10,
            "max_cost_per_day": 100.0,
            "max_cost_per_month": 1000.0
        }

        config = LLMConfig(cost_limits=cost_limits)
        assert config.cost_limits == cost_limits

    def test_cost_limits_validation_invalid(self):
        """Test cost limits validation with invalid values."""
        # Negative cost limit
        with pytest.raises(ValueError, match="must be a non-negative number"):
            LLMConfig(cost_limits={"max_cost_per_request": -0.10})

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValueError):
            LLMConfig(provider=LLMProvider.STUB, invalid_field="value")

    def test_get_api_key_stub_provider(self):
        """Test getting API key for STUB provider (should return empty string)."""
        config = LLMConfig(provider=LLMProvider.STUB)
        api_key = config.get_api_key()

        assert api_key == ""

    def test_get_api_key_from_env(self):
        """Test getting API key from environment variable."""
        # Set environment variable
        os.environ["TEST_OPENAI_KEY"] = "sk-test-key-12345"

        try:
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key_env="TEST_OPENAI_KEY"
            )
            api_key = config.get_api_key()

            assert api_key == "sk-test-key-12345"
        finally:
            # Clean up
            del os.environ["TEST_OPENAI_KEY"]

    def test_get_api_key_missing_env(self):
        """Test getting API key when environment variable is not set."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="NONEXISTENT_API_KEY"
        )

        with pytest.raises(ValueError, match="API key not found|Environment variable"):
            config.get_api_key()

    def test_validate_config_stub_provider(self):
        """Test configuration validation for STUB provider."""
        config = LLMConfig(provider=LLMProvider.STUB)
        is_valid = config.validate_config()

        assert is_valid is True

    def test_validate_config_openai_valid(self):
        """Test configuration validation for OpenAI with valid API key."""
        os.environ["TEST_OPENAI_KEY"] = "sk-test-key-12345"

        try:
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key_env="TEST_OPENAI_KEY",
                model="gpt-4-turbo-preview"
            )
            is_valid = config.validate_config()

            assert is_valid is True
        finally:
            del os.environ["TEST_OPENAI_KEY"]

    def test_validate_config_missing_api_key(self):
        """Test configuration validation with missing API key."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key_env="NONEXISTENT_API_KEY"
        )

        with pytest.raises(ValueError, match="Configuration validation failed|API key"):
            config.validate_config()

    def test_validate_config_empty_model(self):
        """Test configuration validation with empty model name."""
        os.environ["TEST_OPENAI_KEY"] = "sk-test-key-12345"

        try:
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key_env="TEST_OPENAI_KEY",
                model=""
            )

            with pytest.raises(ValueError, match="Model name cannot be empty"):
                config.validate_config()
        finally:
            del os.environ["TEST_OPENAI_KEY"]

    def test_validate_config_local_missing_base_url(self):
        """Test configuration validation for LOCAL provider without base_url."""
        # Note: LOCAL provider doesn't require API key, but needs base_url
        # This test needs the provider to have a way to skip API key validation
        os.environ["DUMMY_LOCAL_KEY"] = ""  # Empty key for local

        try:
            config = LLMConfig(
                provider=LLMProvider.LOCAL,
                api_key_env="DUMMY_LOCAL_KEY",
                model="llama2"
            )

            with pytest.raises(ValueError, match="base_url is required for LOCAL provider"):
                config.validate_config()
        finally:
            del os.environ["DUMMY_LOCAL_KEY"]

    def test_validate_config_invalid_cost_limit_keys(self):
        """Test configuration validation with invalid cost limit keys."""
        os.environ["TEST_OPENAI_KEY"] = "sk-test-key-12345"

        try:
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key_env="TEST_OPENAI_KEY",
                cost_limits={"invalid_key": 100.0}
            )

            with pytest.raises(ValueError, match="Invalid cost limit keys"):
                config.validate_config()
        finally:
            del os.environ["TEST_OPENAI_KEY"]

    def test_get_display_config_api_key_masked(self):
        """Test that API key is masked in display configuration."""
        os.environ["TEST_OPENAI_KEY"] = "sk-test-key-12345"

        try:
            config = LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key_env="TEST_OPENAI_KEY"
            )
            display_config = config.get_display_config()

            assert display_config["api_key"] == "***MASKED***"
            assert display_config["provider"] == "openai"  # String, not enum
            assert display_config["api_key_env"] == "TEST_OPENAI_KEY"
        finally:
            del os.environ["TEST_OPENAI_KEY"]

    def test_get_display_config_stub_provider(self):
        """Test display configuration for STUB provider."""
        config = LLMConfig(provider=LLMProvider.STUB)
        display_config = config.get_display_config()

        assert display_config["api_key"] == "N/A"
        assert display_config["provider"] == "stub"


class TestLLMConfigLoader:
    """Test suite for LLMConfigLoader."""

    def test_from_dict_basic(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
            "temperature": 0.8,
            "max_tokens": 1500
        }

        config = LLMConfigLoader.from_dict(config_dict)

        assert config.provider == LLMProvider.OPENAI
        assert config.model == "gpt-4-turbo-preview"
        assert config.temperature == 0.8
        assert config.max_tokens == 1500

    def test_from_dict_with_cost_limits(self):
        """Test loading configuration with cost limits."""
        config_dict = {
            "provider": "anthropic",
            "model": "claude-3-opus-20240229",
            "cost_limits": {
                "max_cost_per_request": 0.15,
                "max_cost_per_day": 150.0
            }
        }

        config = LLMConfigLoader.from_dict(config_dict)

        assert config.provider == LLMProvider.ANTHROPIC
        assert config.cost_limits["max_cost_per_request"] == 0.15
        assert config.cost_limits["max_cost_per_day"] == 150.0

    def test_from_dict_invalid_provider(self):
        """Test loading configuration with invalid provider."""
        config_dict = {
            "provider": "invalid_provider"
        }

        with pytest.raises(ValueError, match="Invalid provider"):
            LLMConfigLoader.from_dict(config_dict)

    def test_from_env_basic(self):
        """Test loading configuration from environment variables."""
        # Set environment variables
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = "gpt-3.5-turbo"
        os.environ["LLM_TEMPERATURE"] = "0.5"
        os.environ["LLM_MAX_TOKENS"] = "1000"

        try:
            config = LLMConfigLoader.from_env()

            assert config.provider == LLMProvider.OPENAI
            assert config.model == "gpt-3.5-turbo"
            assert config.temperature == 0.5
            assert config.max_tokens == 1000
        finally:
            # Clean up
            del os.environ["LLM_PROVIDER"]
            del os.environ["LLM_MODEL"]
            del os.environ["LLM_TEMPERATURE"]
            del os.environ["LLM_MAX_TOKENS"]

    def test_from_env_with_cache_settings(self):
        """Test loading cache settings from environment."""
        os.environ["LLM_ENABLE_CACHE"] = "true"
        os.environ["LLM_CACHE_TTL"] = "3600"

        try:
            config = LLMConfigLoader.from_env()

            assert config.enable_cache is True
            assert config.cache_ttl == 3600
        finally:
            del os.environ["LLM_ENABLE_CACHE"]
            del os.environ["LLM_CACHE_TTL"]

    def test_from_env_cache_boolean_parsing(self):
        """Test parsing boolean values for enable_cache."""
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("false", False),
            ("FALSE", False),
            ("0", False),
            ("no", False)
        ]

        for env_value, expected in test_cases:
            os.environ["LLM_ENABLE_CACHE"] = env_value

            try:
                config = LLMConfigLoader.from_env()
                assert config.enable_cache is expected
            finally:
                del os.environ["LLM_ENABLE_CACHE"]

    def test_from_env_with_cost_limits(self):
        """Test loading cost limits from environment."""
        os.environ["LLM_MAX_COST_PER_REQUEST"] = "0.20"
        os.environ["LLM_MAX_COST_PER_DAY"] = "50.0"

        try:
            config = LLMConfigLoader.from_env()

            assert config.cost_limits is not None
            assert config.cost_limits["max_cost_per_request"] == 0.20
            assert config.cost_limits["max_cost_per_day"] == 50.0
        finally:
            del os.environ["LLM_MAX_COST_PER_REQUEST"]
            del os.environ["LLM_MAX_COST_PER_DAY"]

    def test_from_yaml_valid_file(self):
        """Test loading configuration from YAML file."""
        yaml_content = """
llm:
  provider: openai
  api_key_env: OPENAI_API_KEY
  model: gpt-4-turbo-preview
  temperature: 0.7
  max_tokens: 2000
  enable_cache: true
  cache_ttl: 86400
  max_retries: 3
  timeout: 60
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = LLMConfigLoader.from_yaml(temp_path)

            assert config.provider == LLMProvider.OPENAI
            assert config.model == "gpt-4-turbo-preview"
            assert config.temperature == 0.7
            assert config.max_tokens == 2000
        finally:
            os.unlink(temp_path)

    def test_from_yaml_file_not_found(self):
        """Test loading from non-existent YAML file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            LLMConfigLoader.from_yaml("nonexistent_file.yaml")

    def test_from_yaml_empty_file(self):
        """Test loading from empty YAML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Configuration file is empty"):
                LLMConfigLoader.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)

    def test_from_yaml_missing_llm_section(self):
        """Test loading YAML without 'llm' section."""
        yaml_content = """
some_other_config:
  value: 123
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="Missing 'llm' section"):
                LLMConfigLoader.from_yaml(temp_path)
        finally:
            os.unlink(temp_path)

    def test_default_config(self):
        """Test getting default configuration."""
        config = LLMConfigLoader.default()

        assert config.provider == LLMProvider.STUB
        assert config.model == "stub-model"
        assert config.enable_cache is False
        assert config.max_retries == 0
        assert config.timeout == 1  # Minimum valid value

    def test_auto_load_from_env(self):
        """Test auto_load prioritizes environment variables."""
        os.environ["LLM_PROVIDER"] = "anthropic"
        os.environ["LLM_MODEL"] = "claude-3-sonnet-20240229"

        try:
            config = LLMConfigLoader.auto_load()

            assert config.provider == LLMProvider.ANTHROPIC
            assert config.model == "claude-3-sonnet-20240229"
        finally:
            del os.environ["LLM_PROVIDER"]
            del os.environ["LLM_MODEL"]

    def test_auto_load_from_yaml(self):
        """Test auto_load loads from YAML when env vars not set."""
        yaml_content = """
llm:
  provider: local
  model: llama2
  base_url: http://localhost:11434/v1
"""

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = LLMConfigLoader.auto_load(temp_path)

            assert config.provider == LLMProvider.LOCAL
            assert config.model == "llama2"
        finally:
            os.unlink(temp_path)

    def test_auto_load_fallback_to_default(self):
        """Test auto_load falls back to default when no config found."""
        # Ensure no env vars set
        env_vars_to_clear = ["LLM_PROVIDER", "LLM_MODEL"]
        saved_env = {}

        for var in env_vars_to_clear:
            if var in os.environ:
                saved_env[var] = os.environ[var]
                del os.environ[var]

        try:
            config = LLMConfigLoader.auto_load("nonexistent.yaml")

            assert config.provider == LLMProvider.STUB
            assert config.model == "stub-model"
        finally:
            # Restore environment
            for var, value in saved_env.items():
                os.environ[var] = value


class TestLLMConfigIntegration:
    """Integration tests for LLM configuration."""

    def test_openai_config_end_to_end(self):
        """Test complete OpenAI configuration workflow."""
        # Set up environment
        os.environ["OPENAI_API_KEY"] = "sk-test-openai-key"
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_MODEL"] = "gpt-4-turbo-preview"

        try:
            # Load from environment
            config = LLMConfigLoader.from_env()

            # Validate
            assert config.validate_config() is True

            # Get API key
            api_key = config.get_api_key()
            assert api_key == "sk-test-openai-key"

            # Get display config (masked)
            display = config.get_display_config()
            assert display["api_key"] == "***MASKED***"
        finally:
            del os.environ["OPENAI_API_KEY"]
            del os.environ["LLM_PROVIDER"]
            del os.environ["LLM_MODEL"]

    def test_anthropic_config_end_to_end(self):
        """Test complete Anthropic configuration workflow."""
        config_dict = {
            "provider": "anthropic",
            "api_key_env": "ANTHROPIC_API_KEY",
            "model": "claude-3-opus-20240229",
            "temperature": 0.8,
            "max_tokens": 4000,
            "cost_limits": {
                "max_cost_per_request": 0.15,
                "max_cost_per_day": 150.0
            }
        }

        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key"

        try:
            config = LLMConfigLoader.from_dict(config_dict)

            assert config.provider == LLMProvider.ANTHROPIC
            assert config.validate_config() is True
            assert config.get_api_key() == "sk-ant-test-key"
        finally:
            del os.environ["ANTHROPIC_API_KEY"]
