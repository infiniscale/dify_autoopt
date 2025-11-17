"""
Test Fixtures for Optimizer Module

Date: 2025-11-17
Author: qa-engineer
Description: Shared pytest fixtures for optimizer module tests
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pytest
from typing import Dict, Any

from src.optimizer.models import (
    Prompt,
    PromptAnalysis,
    PromptIssue,
    PromptSuggestion,
    OptimizationResult,
    PromptVersion,
    OptimizationConfig,
    OptimizationStrategy,
    IssueSeverity,
    IssueType,
    SuggestionType,
)
from src.optimizer.prompt_extractor import PromptExtractor
from src.optimizer.prompt_analyzer import PromptAnalyzer
from src.optimizer.optimization_engine import OptimizationEngine
from src.optimizer.version_manager import VersionManager
from src.optimizer.optimizer_service import OptimizerService
from src.optimizer.interfaces.storage import InMemoryStorage
from src.optimizer.interfaces.llm_client import StubLLMClient


# =============================================================================
# Time and ID Fixtures
# =============================================================================


@pytest.fixture
def fixed_time():
    """Fixed timestamp for deterministic testing."""
    return datetime(2025, 11, 17, 12, 0, 0)


@pytest.fixture
def fixed_id():
    """Fixed ID for deterministic testing."""
    return "test_prompt_001"


# =============================================================================
# Prompt Fixtures
# =============================================================================


@pytest.fixture
def sample_prompt() -> Prompt:
    """Sample prompt for testing."""
    return Prompt(
        id="wf_001_llm_1",
        workflow_id="wf_001",
        node_id="llm_1",
        node_type="llm",
        text="You are a helpful assistant. Summarize the following document: {{document}}",
        role="system",
        variables=["document"],
        context={"label": "Summarizer", "temperature": 0.7},
        extracted_at=datetime(2025, 11, 17, 12, 0, 0),
    )


@pytest.fixture
def sample_prompt_short() -> Prompt:
    """Short prompt for testing edge cases."""
    return Prompt(
        id="wf_001_llm_2",
        workflow_id="wf_001",
        node_id="llm_2",
        node_type="llm",
        text="Tell me about it",
        role="user",
        variables=[],
        context={},
    )


@pytest.fixture
def sample_prompt_long() -> Prompt:
    """Long prompt for testing edge cases."""
    return Prompt(
        id="wf_001_llm_3",
        workflow_id="wf_001",
        node_id="llm_3",
        node_type="llm",
        text="This is a very " * 300 + "long prompt with repetitive content.",
        role="system",
        variables=[],
        context={},
    )


@pytest.fixture
def sample_prompt_vague() -> Prompt:
    """Prompt with vague language for testing issue detection."""
    return Prompt(
        id="wf_001_llm_4",
        workflow_id="wf_001",
        node_id="llm_4",
        node_type="llm",
        text="Maybe you could do some stuff with the things, kind of sort them or whatever.",
        role="user",
        variables=[],
        context={},
    )


@pytest.fixture
def sample_prompt_well_structured() -> Prompt:
    """Well-structured prompt with good formatting."""
    return Prompt(
        id="wf_001_llm_5",
        workflow_id="wf_001",
        node_id="llm_5",
        node_type="llm",
        text="""# Task

Summarize the following document in 3-5 bullet points.

## Requirements

1. Focus on key insights
2. Use clear and concise language
3. Preserve factual accuracy

## Document

{{document}}

## Expected Output

- Bullet point 1
- Bullet point 2
- Bullet point 3""",
        role="system",
        variables=["document"],
        context={"label": "Summarizer"},
    )


@pytest.fixture
def sample_prompt_with_multiple_vars() -> Prompt:
    """Prompt with multiple variables."""
    return Prompt(
        id="wf_002_llm_1",
        workflow_id="wf_002",
        node_id="llm_1",
        node_type="llm",
        text="Hello {{user_name}}, your order {{order_id}} is ready. Details: {{order_details}}",
        role="assistant",
        variables=["user_name", "order_id", "order_details"],
        context={},
    )


# =============================================================================
# Workflow DSL Fixtures
# =============================================================================


@pytest.fixture
def sample_workflow_dict() -> Dict[str, Any]:
    """Sample workflow DSL dictionary."""
    return {
        "id": "test_workflow_001",
        "name": "Test Workflow",
        "graph": {
            "nodes": [
                {
                    "id": "llm_node_1",
                    "type": "llm",
                    "title": "Summarizer",
                    "data": {
                        "prompt_template": {
                            "messages": [
                                {
                                    "role": "system",
                                    "text": "You are a helpful assistant. {{context}}",
                                }
                            ]
                        },
                        "model": {"provider": "openai", "name": "gpt-4"},
                        "temperature": 0.7,
                        "max_tokens": 1000,
                    },
                    "position": {"x": 100, "y": 200},
                },
                {
                    "id": "llm_node_2",
                    "type": "llm",
                    "title": "Analyzer",
                    "data": {
                        "prompt_template": {
                            "messages": [
                                {
                                    "role": "user",
                                    "text": "Analyze this: {{input_data}}",
                                }
                            ]
                        },
                        "model": {"provider": "anthropic", "name": "claude-3"},
                        "temperature": 0.5,
                    },
                },
                {
                    "id": "code_node_1",
                    "type": "code",
                    "data": {"code": "print('hello')"},
                },
            ],
            "edges": [
                {"source": "llm_node_1", "target": "llm_node_2"},
                {"source": "llm_node_2", "target": "code_node_1"},
            ],
        },
    }


@pytest.fixture
def sample_workflow_dict_empty_nodes() -> Dict[str, Any]:
    """Workflow DSL with no LLM nodes."""
    return {
        "id": "workflow_no_llm",
        "graph": {
            "nodes": [
                {"id": "code_1", "type": "code", "data": {}},
                {"id": "tool_1", "type": "tool", "data": {}},
            ]
        },
    }


@pytest.fixture
def sample_workflow_dict_nested() -> Dict[str, Any]:
    """Workflow DSL with nested structure."""
    return {
        "app": {
            "workflow": {
                "graph": {
                    "nodes": [
                        {
                            "id": "nested_llm",
                            "type": "llm",
                            "data": {"text": "Nested prompt {{var}}"},
                        }
                    ]
                }
            }
        }
    }


@pytest.fixture
def sample_workflow_dict_direct_nodes() -> Dict[str, Any]:
    """Workflow DSL with direct nodes list."""
    return {
        "id": "direct_workflow",
        "nodes": [
            {
                "id": "direct_llm",
                "type": "llm",
                "data": {"prompt": "Direct prompt {{input}}"},
            }
        ],
    }


# =============================================================================
# Analysis Fixtures
# =============================================================================


@pytest.fixture
def sample_analysis() -> PromptAnalysis:
    """Sample analysis result."""
    return PromptAnalysis(
        prompt_id="wf_001_llm_1",
        overall_score=75.0,
        clarity_score=80.0,
        efficiency_score=68.0,
        issues=[
            PromptIssue(
                severity=IssueSeverity.WARNING,
                type=IssueType.VAGUE_LANGUAGE,
                description="Contains vague terms",
                location="Throughout prompt",
                suggestion="Replace vague language",
            )
        ],
        suggestions=[
            PromptSuggestion(
                type=SuggestionType.CLARIFY_INSTRUCTIONS,
                description="Improve clarity",
                priority=8,
            )
        ],
        metadata={
            "character_count": 120,
            "word_count": 20,
            "sentence_count": 2,
            "estimated_tokens": 30.0,
        },
    )


@pytest.fixture
def sample_analysis_low_score() -> PromptAnalysis:
    """Analysis with low score."""
    return PromptAnalysis(
        prompt_id="wf_001_llm_4",
        overall_score=45.0,
        clarity_score=40.0,
        efficiency_score=52.0,
        issues=[
            PromptIssue(
                severity=IssueSeverity.WARNING,
                type=IssueType.VAGUE_LANGUAGE,
                description="Vague",
            ),
            PromptIssue(
                severity=IssueSeverity.WARNING,
                type=IssueType.AMBIGUOUS_INSTRUCTIONS,
                description="Ambiguous",
            ),
        ],
        suggestions=[],
        metadata={},
    )


@pytest.fixture
def sample_analysis_high_score() -> PromptAnalysis:
    """Analysis with high score."""
    return PromptAnalysis(
        prompt_id="wf_001_llm_5",
        overall_score=92.0,
        clarity_score=95.0,
        efficiency_score=88.0,
        issues=[],
        suggestions=[],
        metadata={},
    )


# =============================================================================
# Optimization Result Fixtures
# =============================================================================


@pytest.fixture
def sample_optimization_result() -> OptimizationResult:
    """Sample optimization result."""
    return OptimizationResult(
        prompt_id="wf_001_llm_1",
        original_prompt="Summarize the document",
        optimized_prompt="## Task\n\nSummarize the document in 3-5 bullet points focusing on key insights.",
        strategy=OptimizationStrategy.CLARITY_FOCUS,
        improvement_score=12.5,
        confidence=0.85,
        changes=["Added section headers", "Added specific output format"],
        metadata={
            "original_score": 65.0,
            "optimized_score": 77.5,
        },
    )


# =============================================================================
# Version Fixtures
# =============================================================================


@pytest.fixture
def sample_version(sample_prompt, sample_analysis) -> PromptVersion:
    """Sample prompt version."""
    return PromptVersion(
        prompt_id="wf_001_llm_1",
        version="1.0.0",
        prompt=sample_prompt,
        analysis=sample_analysis,
        optimization_result=None,
        parent_version=None,
        created_at=datetime(2025, 11, 17, 12, 0, 0),
        metadata={"author": "baseline"},
    )


# =============================================================================
# Configuration Fixtures
# =============================================================================


@pytest.fixture
def sample_optimization_config() -> OptimizationConfig:
    """Sample optimization configuration."""
    return OptimizationConfig(
        strategies=[OptimizationStrategy.CLARITY_FOCUS],
        min_confidence=0.6,
        max_iterations=3,
        analysis_rules={},
        metadata={},
    )


# =============================================================================
# Component Fixtures
# =============================================================================


@pytest.fixture
def extractor() -> PromptExtractor:
    """PromptExtractor instance."""
    return PromptExtractor()


@pytest.fixture
def stub_llm_client() -> StubLLMClient:
    """StubLLMClient instance."""
    return StubLLMClient()


@pytest.fixture
def analyzer(stub_llm_client) -> PromptAnalyzer:
    """PromptAnalyzer instance with StubLLMClient."""
    return PromptAnalyzer(llm_client=stub_llm_client)


@pytest.fixture
def engine(analyzer) -> OptimizationEngine:
    """OptimizationEngine instance."""
    return OptimizationEngine(analyzer=analyzer)


@pytest.fixture
def in_memory_storage() -> InMemoryStorage:
    """InMemoryStorage instance."""
    return InMemoryStorage()


@pytest.fixture
def version_manager(in_memory_storage) -> VersionManager:
    """VersionManager with InMemoryStorage."""
    return VersionManager(storage=in_memory_storage)


@pytest.fixture
def optimizer_service() -> OptimizerService:
    """OptimizerService instance for testing (no catalog)."""
    return OptimizerService(
        catalog=None,
        llm_client=StubLLMClient(),
        storage=InMemoryStorage(),
    )


# =============================================================================
# Mock Catalog Fixture
# =============================================================================


@pytest.fixture
def mock_catalog(tmp_path):
    """Mock WorkflowCatalog for testing."""
    from unittest.mock import MagicMock, PropertyMock
    from src.config.models import WorkflowCatalog, WorkflowEntry

    # Create mock DSL file
    dsl_content = """
id: test_workflow_001
name: Test Workflow
graph:
  nodes:
    - id: llm_node_1
      type: llm
      data:
        prompt_template:
          messages:
            - role: system
              text: "You are helpful. {{context}}"
        model:
          provider: openai
          name: gpt-4
    - id: llm_node_2
      type: llm
      data:
        text: "Analyze: {{input}}"
"""
    dsl_file = tmp_path / "test_workflow.yaml"
    dsl_file.write_text(dsl_content)

    # Create mock workflow entry
    mock_entry = MagicMock(spec=WorkflowEntry)
    mock_entry.id = "test_workflow_001"
    type(mock_entry).dsl_path_resolved = PropertyMock(return_value=dsl_file)

    # Create mock catalog
    mock_catalog = MagicMock(spec=WorkflowCatalog)
    mock_catalog.get_workflow.return_value = mock_entry

    return mock_catalog
