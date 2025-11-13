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
    """High-level model configuration for evaluation"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    provider: str = Field(..., description="Model provider (e.g., 'openai', 'anthropic')")
    model_name: str = Field(..., description="Model name (e.g., 'gpt-4', 'claude-3')")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(512, ge=1, description="Maximum tokens for generation")
