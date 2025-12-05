"""
LLM Configuration Loader Module.

This module provides utilities to load LLM configuration from multiple sources:
- YAML configuration files
- Environment variables
- Python dictionaries
- Default configurations

The loader follows a priority order:
1. Environment variables (highest priority)
2. YAML configuration file
3. Dictionary configuration
4. Default values (lowest priority)

Security:
- API keys are always loaded from environment variables
- Configuration files reference environment variable names, not actual keys
- Validation ensures secure configuration

Author: Backend Developer
Date: 2025-11-18
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from loguru import logger

from .llm_config import LLMConfig, LLMProvider


class LLMConfigLoader:
    """Load LLM configuration from various sources.

    This class provides static methods to load LLM configuration from:
    - YAML files (recommended for production)
    - Environment variables (for containerized deployments)
    - Python dictionaries (for programmatic configuration)
    - Default configurations (for development/testing)

    The loader supports configuration merging with environment variables
    taking precedence over file-based configuration.

    Example:
        >>> # Load from YAML file
        >>> config = LLMConfigLoader.from_yaml("config/llm.yaml")
        >>>
        >>> # Load from environment variables
        >>> config = LLMConfigLoader.from_env()
        >>>
        >>> # Load from dictionary
        >>> config = LLMConfigLoader.from_dict({"provider": "openai"})
        >>>
        >>> # Get default configuration
        >>> config = LLMConfigLoader.default()
    """

    @staticmethod
    def from_yaml(config_path: str) -> LLMConfig:
        """Load LLM configuration from YAML file.

        This method loads configuration from a YAML file and merges it with
        environment variables (env vars take precedence).

        Args:
            config_path: Path to YAML configuration file

        Returns:
            LLMConfig instance with loaded configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If configuration is invalid
            yaml.YAMLError: If YAML parsing fails

        Example:
            >>> config = LLMConfigLoader.from_yaml("config/llm.yaml")
            >>> print(config.provider)
        """
        config_file = Path(config_path)

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found: {config_path}\n"
                f"Please create a configuration file or use from_env() or default()."
            )

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f)

            if not yaml_data:
                raise ValueError(f"Configuration file is empty: {config_path}")

            # Extract LLM configuration section
            llm_config = yaml_data.get("llm", {})
            if not llm_config:
                raise ValueError(
                    f"Missing 'llm' section in configuration file: {config_path}\n"
                    f"Expected format:\n"
                    f"llm:\n"
                    f"  provider: openai\n"
                    f"  model: gpt-4-turbo-preview\n"
                    f"  ..."
                )

            logger.info(f"Loaded LLM configuration from YAML: {config_path}")

            # Create config from dictionary (will merge with env vars)
            return LLMConfigLoader.from_dict(llm_config)

        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse YAML file {config_path}: {e}")

    @staticmethod
    def from_dict(config_dict: Dict[str, Any]) -> LLMConfig:
        """Load LLM configuration from dictionary.

        This method creates an LLMConfig from a dictionary, merging with
        environment variables where applicable.

        Environment variables override dictionary values:
        - LLM_PROVIDER overrides provider
        - LLM_MODEL overrides model
        - LLM_BASE_URL overrides base_url
        - LLM_TEMPERATURE overrides temperature
        - LLM_MAX_TOKENS overrides max_tokens
        - LLM_ENABLE_CACHE overrides enable_cache
        - LLM_MAX_RETRIES overrides max_retries
        - LLM_TIMEOUT overrides timeout

        Args:
            config_dict: Dictionary containing configuration

        Returns:
            LLMConfig instance

        Raises:
            ValueError: If configuration is invalid

        Example:
            >>> config_dict = {
            ...     "provider": "openai",
            ...     "model": "gpt-4-turbo-preview",
            ...     "temperature": 0.7
            ... }
            >>> config = LLMConfigLoader.from_dict(config_dict)
        """
        # Merge with environment variables (env vars take precedence)
        provider_str = os.getenv("LLM_PROVIDER", config_dict.get("provider", "stub"))
        model = os.getenv("LLM_MODEL", config_dict.get("model", "gpt-4-turbo-preview"))
        api_key_env = os.getenv("LLM_API_KEY_ENV", config_dict.get("api_key_env", "OPENAI_API_KEY"))
        base_url = os.getenv("LLM_BASE_URL", config_dict.get("base_url"))
        temperature_str = os.getenv("LLM_TEMPERATURE", config_dict.get("temperature", 0.7))
        max_tokens_str = os.getenv("LLM_MAX_TOKENS", config_dict.get("max_tokens", 2000))
        enable_cache_str = os.getenv("LLM_ENABLE_CACHE", config_dict.get("enable_cache", True))
        cache_ttl_str = os.getenv("LLM_CACHE_TTL", config_dict.get("cache_ttl", 86400))
        max_retries_str = os.getenv("LLM_MAX_RETRIES", config_dict.get("max_retries", 3))
        timeout_str = os.getenv("LLM_TIMEOUT", config_dict.get("timeout", 60))

        # Convert string values to appropriate types
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            raise ValueError(
                f"Invalid provider: '{provider_str}'. "
                f"Must be one of: {[p.value for p in LLMProvider]}"
            )

        temperature = float(temperature_str)
        max_tokens = int(max_tokens_str)
        cache_ttl = int(cache_ttl_str)
        max_retries = int(max_retries_str)
        timeout = int(timeout_str)

        # Handle boolean enable_cache
        if isinstance(enable_cache_str, str):
            enable_cache = enable_cache_str.lower() in ("true", "1", "yes", "on")
        else:
            enable_cache = bool(enable_cache_str)

        # Get cost limits
        cost_limits = config_dict.get("cost_limits")

        # Get provider settings
        provider_settings = config_dict.get("provider_settings", {})

        # Create configuration
        config = LLMConfig(
            provider=provider,
            api_key_env=api_key_env,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl,
            max_retries=max_retries,
            timeout=timeout,
            cost_limits=cost_limits,
            provider_settings=provider_settings
        )

        logger.info(f"Created LLM configuration: provider={config.provider}, model={config.model}")
        return config

    @staticmethod
    def from_env() -> LLMConfig:
        """Load LLM configuration from environment variables.

        This method creates an LLMConfig using only environment variables.
        Useful for containerized deployments (Docker, Kubernetes).

        Environment Variables:
            LLM_PROVIDER: Provider name (openai, anthropic, local, stub)
            LLM_MODEL: Model name (e.g., gpt-4-turbo-preview)
            LLM_API_KEY_ENV: Environment variable name for API key
            LLM_BASE_URL: API base URL (for local models)
            LLM_TEMPERATURE: Sampling temperature (0.0-2.0)
            LLM_MAX_TOKENS: Maximum tokens to generate
            LLM_ENABLE_CACHE: Enable caching (true/false)
            LLM_CACHE_TTL: Cache TTL in seconds
            LLM_MAX_RETRIES: Maximum retry attempts
            LLM_TIMEOUT: Request timeout in seconds

        Returns:
            LLMConfig instance from environment variables

        Example:
            >>> # Set environment variables first
            >>> os.environ["LLM_PROVIDER"] = "openai"
            >>> os.environ["LLM_MODEL"] = "gpt-4-turbo-preview"
            >>> config = LLMConfigLoader.from_env()
        """
        provider_str = os.getenv("LLM_PROVIDER", "stub")
        model = os.getenv("LLM_MODEL", "gpt-4-turbo-preview")
        api_key_env = os.getenv("LLM_API_KEY_ENV", "OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "2000"))
        enable_cache_str = os.getenv("LLM_ENABLE_CACHE", "true")
        cache_ttl = int(os.getenv("LLM_CACHE_TTL", "86400"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        timeout = int(os.getenv("LLM_TIMEOUT", "60"))

        # Parse provider
        try:
            provider = LLMProvider(provider_str)
        except ValueError:
            raise ValueError(
                f"Invalid LLM_PROVIDER: '{provider_str}'. "
                f"Must be one of: {[p.value for p in LLMProvider]}"
            )

        # Parse enable_cache
        enable_cache = enable_cache_str.lower() in ("true", "1", "yes", "on")

        # Parse cost limits from env if provided
        cost_limits = None
        max_cost_per_request = os.getenv("LLM_MAX_COST_PER_REQUEST")
        max_cost_per_day = os.getenv("LLM_MAX_COST_PER_DAY")
        if max_cost_per_request or max_cost_per_day:
            cost_limits = {}
            if max_cost_per_request:
                cost_limits["max_cost_per_request"] = float(max_cost_per_request)
            if max_cost_per_day:
                cost_limits["max_cost_per_day"] = float(max_cost_per_day)

        config = LLMConfig(
            provider=provider,
            api_key_env=api_key_env,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_cache=enable_cache,
            cache_ttl=cache_ttl,
            max_retries=max_retries,
            timeout=timeout,
            cost_limits=cost_limits
        )

        logger.info(f"Loaded LLM configuration from environment: provider={config.provider}")
        return config

    @staticmethod
    def default() -> LLMConfig:
        """Return default LLM configuration (StubClient).

        This configuration uses the StubClient which implements rule-based
        optimization without making any LLM API calls. Suitable for:
        - Development without API keys
        - Testing without API costs
        - Fallback when LLM is unavailable

        Returns:
            LLMConfig with STUB provider (no API calls)

        Example:
            >>> config = LLMConfigLoader.default()
            >>> print(config.provider)  # LLMProvider.STUB
        """
        config = LLMConfig(
            provider=LLMProvider.STUB,
            api_key_env="",  # Not needed for STUB
            model="stub-model",
            temperature=0.7,
            max_tokens=2000,
            enable_cache=False,  # No caching needed for rules
            cache_ttl=1,  # Minimum valid TTL (won't be used)
            max_retries=0,  # No retries needed for rules
            timeout=1  # Minimum valid timeout (won't be used)
        )

        logger.info("Using default LLM configuration (STUB provider - rule-based optimization)")
        return config

    @staticmethod
    def auto_load(config_path: Optional[str] = None) -> LLMConfig:
        """Automatically load configuration with fallback chain.

        This method tries to load configuration in the following order:
        1. Environment variables (if LLM_PROVIDER is set)
        2. YAML file (if provided and exists)
        3. Default YAML file (config/llm.yaml if exists)
        4. Default configuration (STUB provider)

        Args:
            config_path: Optional path to YAML configuration file

        Returns:
            LLMConfig loaded from the first available source

        Example:
            >>> # Tries env vars, then config/llm.yaml, then defaults
            >>> config = LLMConfigLoader.auto_load()
            >>>
            >>> # Tries env vars, then custom path, then defaults
            >>> config = LLMConfigLoader.auto_load("custom/llm.yaml")
        """
        # 1. Try environment variables first
        if os.getenv("LLM_PROVIDER"):
            logger.info("Loading LLM configuration from environment variables")
            try:
                return LLMConfigLoader.from_env()
            except Exception as e:
                logger.warning(f"Failed to load from environment: {e}")

        # 2. Try provided YAML path
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                logger.info(f"Loading LLM configuration from: {config_path}")
                try:
                    return LLMConfigLoader.from_yaml(config_path)
                except Exception as e:
                    logger.warning(f"Failed to load from {config_path}: {e}")

        # 3. Try default YAML path
        default_config_path = Path("config/llm.yaml")
        if default_config_path.exists():
            logger.info(f"Loading LLM configuration from: {default_config_path}")
            try:
                return LLMConfigLoader.from_yaml(str(default_config_path))
            except Exception as e:
                logger.warning(f"Failed to load from {default_config_path}: {e}")

        # 4. Fall back to default configuration
        logger.info("No configuration found, using default (STUB provider)")
        return LLMConfigLoader.default()
