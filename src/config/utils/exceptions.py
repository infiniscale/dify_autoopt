"""
YAML Configuration Module - Custom Exceptions

Date: 2025-11-13
Author: Rebirthli
Description: Custom exception classes for the YAML configuration module
"""


class YamlModuleError(Exception):
    """Base exception for YAML module errors"""
    pass


class ConfigurationError(YamlModuleError):
    """Configuration file error (schema validation, format, etc.)"""
    pass


class ConfigReferenceError(YamlModuleError):
    """Cross-file reference error (workflow not found, dataset missing, etc.)"""
    pass


class PatchTargetMissing(YamlModuleError):
    """Prompt patch target node not found in DSL"""
    pass


class TemplateRenderError(YamlModuleError):
    """Template rendering error in prompt patch"""
    pass


class CaseGenerationError(YamlModuleError):
    """Test case generation error"""
    pass


class DSLParseError(YamlModuleError):
    """DSL YAML parsing error"""
    pass


class ManifestError(YamlModuleError):
    """RunManifest building error"""
    pass
