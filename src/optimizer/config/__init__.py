"""
LLM Configuration Module.

This module provides configuration management for LLM integration in the optimizer.
It supports multiple LLM providers (OpenAI, Anthropic, Local, Stub) with unified
configuration schema, security features, and flexible loading options.

Public Classes:
    LLMConfig: Configuration model with validation and security features
    LLMProvider: Enum of supported LLM providers
    LLMConfigLoader: Utilities to load configuration from various sources

Example Usage:
    >>> from src.optimizer.config import LLMConfig, LLMProvider, LLMConfigLoader
    >>>
    >>> # Load from YAML file
    >>> config = LLMConfigLoader.from_yaml("config/llm.yaml")
    >>>
    >>> # Load from environment variables
    >>> config = LLMConfigLoader.from_env()
    >>>
    >>> # Create programmatically
    >>> config = LLMConfig(
    ...     provider=LLMProvider.OPENAI,
    ...     model="gpt-4-turbo-preview"
    ... )
    >>>
    >>> # Auto-load with fallback
    >>> config = LLMConfigLoader.auto_load()

Author: Backend Developer
Date: 2025-11-18
"""

from .llm_config import LLMConfig, LLMProvider
from .llm_config_loader import LLMConfigLoader

__all__ = [
    "LLMConfig",
    "LLMProvider",
    "LLMConfigLoader",
]
