"""
YAML Configuration Module - Common Models

Date: 2025-11-13
Author: Rebirthli
Description: Shared Pydantic models used across multiple configuration files
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class RateLimit(BaseModel):
    """Rate limiting configuration"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    per_minute: int = Field(..., ge=1, description="Requests per minute")
    burst: int = Field(5, ge=1, description="Burst size for rate limiting")


class ModelEvaluator(BaseModel):
    """High-level model configuration for evaluation.

    Used by executor module for test result scoring with LLMs.

    TODO: Consider extracting common base class with LLMConfig
    when executor LLM evaluation is fully implemented.

    Current status: Placeholder (executor not using LLM evaluation yet)

    Field overlap with LLMConfig (src/optimizer/config/llm_config.py):
    - provider, model_name, temperature, max_tokens (100% overlap)

    Future refactoring:
    - Create src/config/models/llm_base_config.py
    - ModelEvaluator and LLMConfig both inherit from LLMBaseConfig
    - Eliminates field duplication while maintaining separate concerns

    See: config/README.md "LLM配置说明" for details
    """
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    provider: str = Field(..., description="Model provider (e.g., 'openai', 'anthropic')")
    model_name: str = Field(..., description="Model name (e.g., 'gpt-4', 'claude-3')")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(512, ge=1, description="Maximum tokens for generation")
