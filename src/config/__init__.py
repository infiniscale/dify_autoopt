"""
YAML Configuration Module

Provides models, loaders, and validators for YAML-based configuration management.
Components for execution (test generation, manifest building) have been moved to:
- src/optimizer/ - Prompt optimization functionality
- src/executor/ - Test case generation and execution

This module now focuses on:
- Configuration data models
- YAML file loading and parsing
- Cross-configuration validation

Note: YamlModuleFacade has been moved out of __init__ to avoid circular imports.
Import it directly: from src.config.facades.yaml_module_facade import YamlModuleFacade
"""

from .models import (
    # Common models
    RateLimit,
    ModelEvaluator,
    # Environment config models
    EnvConfig,
    DifyConfig,
    AuthConfig,
    # Workflow catalog models
    WorkflowCatalog,
    WorkflowEntry,
    NodeMeta,
    # Test plan models
    TestPlan,
    WorkflowPlanEntry,
    Dataset,
    InputParameter,
    PromptPatch,
    PromptSelector,
    PromptStrategy,
    PromptTemplate,
    ExecutionPolicy,
    # Run manifest models
    RunManifest,
    TestCase,
)

from .loaders import ConfigLoader, ConfigValidator, FileSystemReader

__all__ = [
    # Models
    'RateLimit',
    'ModelEvaluator',
    'EnvConfig',
    'DifyConfig',
    'AuthConfig',
    'WorkflowCatalog',
    'WorkflowEntry',
    'NodeMeta',
    'TestPlan',
    'WorkflowPlanEntry',
    'Dataset',
    'InputParameter',
    'PromptPatch',
    'PromptSelector',
    'PromptStrategy',
    'PromptTemplate',
    'ExecutionPolicy',
    'RunManifest',
    'TestCase',
    # Loaders
    'ConfigLoader',
    'ConfigValidator',
    'FileSystemReader',
]

__version__ = "1.0.0"
