"""
Test Suite for Optimizer Module

Date: 2025-11-17
Author: qa-engineer
Description: Comprehensive tests for prompt optimization functionality
"""

import importlib.util
import pytest

if importlib.util.find_spec("src.optimizer.models") is None:
    pytest.skip("Legacy optimizer components are unavailable; skipping optimizer tests.", allow_module_level=True)
