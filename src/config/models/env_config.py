"""
YAML Configuration Module - Environment Configuration Models

Date: 2025-11-13
Author: Rebirthli
Description: Pydantic models for env_config.yaml
"""

from typing import Any, Dict, List, Optional
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .common import RateLimit, ModelEvaluator


class AuthConfig(BaseModel):
    """Authentication configuration"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    primary_token: SecretStr = Field(..., description="Primary API token")
    fallback_tokens: List[SecretStr] = Field(default_factory=list, description="Fallback tokens for rotation")


class DifyConfig(BaseModel):
    """Dify platform configuration"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    base_url: str = Field(..., description="Dify base URL")
    tenant_id: Optional[str] = Field(None, description="Tenant ID (optional)")
    auth: AuthConfig = Field(..., description="Authentication configuration")
    rate_limits: RateLimit = Field(..., description="Rate limiting configuration")

    @field_validator('base_url')
    @classmethod
    def validate_url(cls, value: str) -> str:
        if not value.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL format: {value}")
        return value


class EnvConfig(BaseModel):
    """Root environment configuration model"""
    model_config = ConfigDict(extra='forbid', validate_assignment=True)

    meta: Dict[str, Any] = Field(..., description="Metadata (version, environment, updated_by)")
    dify: DifyConfig = Field(..., description="Dify platform configuration")
    model_evaluator: ModelEvaluator = Field(..., description="High-level model for evaluation")
    io_paths: Dict[str, Path] = Field(..., description="I/O paths configuration")
    logging: Dict[str, Any] = Field(..., description="Logging configuration")
    defaults: Dict[str, Any] = Field(default_factory=dict, description="Default values")

    @field_validator('io_paths', mode='before')
    @classmethod
    def validate_paths(cls, value: Any) -> Dict[str, Path]:
        """Convert string paths to Path objects"""
        if isinstance(value, dict):
            return {k: Path(path) if isinstance(path, str) else path for k, path in value.items()}
        return value
