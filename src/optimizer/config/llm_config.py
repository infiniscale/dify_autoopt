"""
LLM Configuration Module for Optimizer.

This module defines the configuration models for LLM integration, supporting
multiple providers (OpenAI, Anthropic, Local, Stub) with unified configuration schema.

Security:
- API keys are protected using Pydantic's SecretStr
- API keys are loaded from environment variables, never hardcoded
- Configuration validation ensures secure and valid parameters

Author: Backend Developer
Date: 2025-11-18
"""

import os
from enum import Enum
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, SecretStr, field_validator, ConfigDict


class LLMProvider(str, Enum):
    """Supported LLM providers.

    Attributes:
        OPENAI: OpenAI GPT models (GPT-4, GPT-3.5)
        ANTHROPIC: Anthropic Claude models
        LOCAL: Local LLM models (Ollama, vLLM, etc.)
        STUB: Stub client using rule-based optimization (default, no API calls)
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    STUB = "stub"


class LLMConfig(BaseModel):
    """LLM client configuration with comprehensive settings.

    Provides complete configuration for LLM-driven prompt optimization,
    including caching, cost control, and retry mechanisms.

    Note: Core fields overlap with ModelEvaluator (src/config/models/common.py).
    This is intentional due to different use cases:
    - ModelEvaluator: Simple LLM calls for test scoring
    - LLMConfig: Complex LLM optimization with caching, cost control

    Future: May extract common base class if both modules need full LLM features.
    See: config/README.md "LLM配置说明" for refactoring plan.

    This model provides comprehensive configuration for LLM integration with:
    - Multiple provider support (OpenAI, Anthropic, Local, Stub)
    - Security features (SecretStr for API keys)
    - Cost control (rate limiting, token limits, cost caps)
    - Performance optimization (caching, retry logic)
    - Validation (parameter ranges, required fields)

    Example:
        >>> config = LLMConfig(
        ...     provider=LLMProvider.OPENAI,
        ...     api_key_env="OPENAI_API_KEY",
        ...     model="gpt-4-turbo-preview"
        ... )
        >>> api_key = config.get_api_key()
        >>> is_valid = config.validate_config()

    Security Notes:
        - API keys MUST be stored in environment variables
        - Use api_key_env to reference the environment variable name
        - Never hardcode API keys in configuration files
    """

    # Provider Configuration
    provider: LLMProvider = Field(
        default=LLMProvider.STUB,
        description="LLM provider to use (openai, anthropic, local, stub)"
    )

    # API Key Configuration (Security)
    api_key_env: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable name containing the API key (e.g., 'OPENAI_API_KEY')"
    )

    # Model Selection
    model: str = Field(
        default="gpt-4-turbo-preview",
        description="Model name (e.g., 'gpt-4-turbo-preview', 'claude-3-opus-20240229')"
    )

    # API Configuration
    base_url: Optional[str] = Field(
        default=None,
        description="Custom API base URL (for local models, e.g., 'http://localhost:11434/v1')"
    )

    # Generation Parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = deterministic, 2.0 = highly random)"
    )

    max_tokens: int = Field(
        default=2000,
        gt=0,
        le=128000,
        description="Maximum tokens to generate in response"
    )

    # Caching Configuration
    enable_cache: bool = Field(
        default=True,
        description="Enable response caching to reduce API calls and costs"
    )

    cache_ttl: int = Field(
        default=86400,
        gt=0,
        description="Cache TTL in seconds (default: 24 hours)"
    )

    # Retry Configuration
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts for failed API calls"
    )

    timeout: int = Field(
        default=60,
        gt=0,
        le=600,
        description="Request timeout in seconds"
    )

    # Cost Control
    cost_limits: Optional[Dict[str, float]] = Field(
        default=None,
        description="Cost limits (e.g., {'max_cost_per_request': 0.1, 'max_cost_per_day': 100.0})"
    )

    # Provider-Specific Settings
    provider_settings: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific configuration options"
    )

    # Model Configuration
    model_config = ConfigDict(
        extra="forbid",  # Prevent extra fields to catch configuration errors
        use_enum_values=True
    )

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is within acceptable range.

        Args:
            v: Temperature value to validate

        Returns:
            Validated temperature value

        Raises:
            ValueError: If temperature is out of range [0.0, 2.0]
        """
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, v: int) -> int:
        """Validate max_tokens is positive and within reasonable limits.

        Args:
            v: Max tokens value to validate

        Returns:
            Validated max tokens value

        Raises:
            ValueError: If max_tokens is invalid
        """
        if v <= 0:
            raise ValueError(f"max_tokens must be positive, got {v}")
        if v > 128000:
            raise ValueError(f"max_tokens must be <= 128000, got {v}")
        return v

    @field_validator("cache_ttl")
    @classmethod
    def validate_cache_ttl(cls, v: int) -> int:
        """Validate cache TTL is positive.

        Args:
            v: Cache TTL value to validate

        Returns:
            Validated cache TTL value

        Raises:
            ValueError: If cache_ttl is not positive
        """
        if v <= 0:
            raise ValueError(f"cache_ttl must be positive, got {v}")
        return v

    @field_validator("cost_limits")
    @classmethod
    def validate_cost_limits(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Validate cost limits if provided.

        Args:
            v: Cost limits dictionary to validate

        Returns:
            Validated cost limits or None

        Raises:
            ValueError: If cost limits contain invalid values
        """
        if v is None:
            return v

        for key, value in v.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Cost limit '{key}' must be a non-negative number, got {value}")

        return v

    def get_api_key(self) -> str:
        """Get API key from environment variable.

        This method reads the API key from the environment variable specified
        in api_key_env. For STUB provider, returns an empty string.

        Returns:
            API key string

        Raises:
            ValueError: If API key environment variable is not set (for non-STUB providers)

        Example:
            >>> config = LLMConfig(provider=LLMProvider.OPENAI)
            >>> api_key = config.get_api_key()
        """
        # STUB provider doesn't need API key
        if self.provider == LLMProvider.STUB:
            return ""

        # Get API key from environment
        api_key = os.getenv(self.api_key_env)

        if not api_key:
            raise ValueError(
                f"API key not found: Environment variable '{self.api_key_env}' is not set. "
                f"Please set it before using {self.provider} provider."
            )

        return api_key

    def validate_config(self) -> bool:
        """Validate complete configuration integrity.

        This method performs comprehensive validation:
        - Checks API key availability for non-STUB providers
        - Validates provider-specific requirements
        - Ensures configuration consistency

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config = LLMConfig(provider=LLMProvider.OPENAI)
            >>> is_valid = config.validate_config()
        """
        # For STUB provider, no API key needed
        if self.provider == LLMProvider.STUB:
            return True

        # For LOCAL provider, allow empty API key
        if self.provider == LLMProvider.LOCAL:
            # Just validate base_url is present
            if not self.base_url:
                raise ValueError("base_url is required for LOCAL provider")
            return True

        # Check API key availability for cloud providers
        try:
            api_key = self.get_api_key()
            if not api_key:
                raise ValueError(f"API key is empty for provider {self.provider}")
        except ValueError as e:
            raise ValueError(f"Configuration validation failed: {e}")

        # Validate model name is not empty
        if not self.model or not self.model.strip():
            raise ValueError("Model name cannot be empty")

        # Validate cost limits if provided
        if self.cost_limits:
            allowed_keys = {
                "max_cost_per_request",
                "max_cost_per_day",
                "max_cost_per_month"
            }
            invalid_keys = set(self.cost_limits.keys()) - allowed_keys
            if invalid_keys:
                raise ValueError(
                    f"Invalid cost limit keys: {invalid_keys}. "
                    f"Allowed keys: {allowed_keys}"
                )

        return True

    def get_display_config(self) -> Dict[str, Any]:
        """Get configuration for display/logging (with API key masked).

        This method returns a safe representation of the configuration
        suitable for logging or display, with sensitive information masked.

        Returns:
            Dictionary with masked sensitive information

        Example:
            >>> config = LLMConfig(provider=LLMProvider.OPENAI)
            >>> display_config = config.get_display_config()
            >>> print(display_config)
        """
        return {
            "provider": self.provider,  # Returns string value due to use_enum_values
            "model": self.model,
            "api_key_env": self.api_key_env,
            "api_key": "***MASKED***" if self.provider != LLMProvider.STUB else "N/A",
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_cache": self.enable_cache,
            "cache_ttl": self.cache_ttl,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "cost_limits": self.cost_limits,
        }
