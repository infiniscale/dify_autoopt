"""
Tests for Test-Driven Optimization Scoring Rules

Date: 2025-11-19
Author: backend-architect
Description: Test ScoringRules with test execution results integration
"""

import pytest
from src.optimizer.scoring_rules import ScoringRules
from src.optimizer.models import (
    PromptAnalysis,
    PromptIssue,
    IssueSeverity,
    IssueType,
    TestExecutionReport,
    ErrorDistribution
)


class TestScoringRulesTestDriven:
    """Test ScoringRules with test execution results."""

    @pytest.fixture
    def default_rules(self):
        """Provide default ScoringRules instance."""
        return ScoringRules()

    @pytest.fixture
    def high_quality_analysis(self):
        """Provide high-quality prompt analysis."""
        return PromptAnalysis(
            prompt_id="test_prompt",
            overall_score=90.0,
            clarity_score=90.0,
            efficiency_score=90.0,
            issues=[],
            suggestions=[]
        )

    @pytest.fixture
    def low_quality_analysis(self):
        """Provide low-quality prompt analysis."""
        return PromptAnalysis(
            prompt_id="test_prompt",
            overall_score=60.0,
            clarity_score=60.0,
            efficiency_score=60.0,
            issues=[
                PromptIssue(
                    severity=IssueSeverity.CRITICAL,
                    type=IssueType.VAGUE_LANGUAGE,
                    description="Vague language detected"
                )
            ],
            suggestions=[]
        )

    @pytest.fixture
    def good_test_results(self):
        """Provide good test execution results."""
        return TestExecutionReport(
            workflow_id="wf_001",
            run_id="run_001",
            total_tests=100,
            successful_tests=95,
            failed_tests=5,
            success_rate=0.95,
            avg_response_time_ms=1000.0,
            total_tokens=10000,
            avg_tokens_per_request=100.0,
            total_cost=1.0,
            cost_per_request=0.01
        )

    @pytest.fixture
    def poor_test_results_low_success(self):
        """Provide poor test results with low success rate."""
        return TestExecutionReport(
            workflow_id="wf_001",
            run_id="run_001",
            total_tests=100,
            successful_tests=70,
            failed_tests=30,
            success_rate=0.70,  # Below default 0.8 threshold
            avg_response_time_ms=1000.0,
            total_tokens=10000,
            avg_tokens_per_request=100.0,
            total_cost=1.0,
            cost_per_request=0.01
        )

    @pytest.fixture
    def poor_test_results_high_latency(self):
        """Provide poor test results with high latency."""
        return TestExecutionReport(
            workflow_id="wf_001",
            run_id="run_001",
            total_tests=100,
            successful_tests=95,
            failed_tests=5,
            success_rate=0.95,
            avg_response_time_ms=8000.0,  # Above default 5000ms threshold
            total_tokens=10000,
            avg_tokens_per_request=100.0,
            total_cost=1.0,
            cost_per_request=0.01
        )

    @pytest.fixture
    def poor_test_results_high_cost(self):
        """Provide poor test results with high cost."""
        return TestExecutionReport(
            workflow_id="wf_001",
            run_id="run_001",
            total_tests=100,
            successful_tests=95,
            failed_tests=5,
            success_rate=0.95,
            avg_response_time_ms=1000.0,
            total_tokens=10000,
            avg_tokens_per_request=100.0,
            total_cost=15.0,
            cost_per_request=0.15  # Above default 0.1 threshold
        )

    @pytest.fixture
    def poor_test_results_timeouts(self):
        """Provide poor test results with excessive timeouts."""
        return TestExecutionReport(
            workflow_id="wf_001",
            run_id="run_001",
            total_tests=100,
            successful_tests=90,
            failed_tests=10,
            success_rate=0.90,
            avg_response_time_ms=2000.0,
            total_tokens=10000,
            avg_tokens_per_request=100.0,
            total_cost=1.0,
            cost_per_request=0.01,
            error_distribution=ErrorDistribution(
                timeout_errors=10,  # 10% timeout rate, above 5% threshold
                api_errors=0,
                validation_errors=0,
                llm_errors=0,
                total_errors=10
            )
        )

    # ==================== Static Analysis Tests (Unchanged Behavior) ====================

    def test_should_optimize_low_static_score(self, default_rules, low_quality_analysis):
        """Should optimize when static analysis score is low (existing behavior)."""
        assert default_rules.should_optimize(low_quality_analysis) is True

    def test_should_optimize_critical_issues(self, default_rules, high_quality_analysis):
        """Should optimize when critical issues present (existing behavior)."""
        analysis_with_critical = PromptAnalysis(
            prompt_id="test",
            overall_score=90.0,
            clarity_score=90.0,
            efficiency_score=90.0,
            issues=[
                PromptIssue(
                    severity=IssueSeverity.CRITICAL,
                    type=IssueType.AMBIGUOUS_INSTRUCTIONS,
                    description="Critical issue"
                )
            ],
            suggestions=[]
        )

        assert default_rules.should_optimize(analysis_with_critical) is True

    def test_should_not_optimize_high_static_score_no_issues(
        self, default_rules, high_quality_analysis
    ):
        """Should not optimize when static analysis is good (existing behavior)."""
        assert default_rules.should_optimize(high_quality_analysis) is False

    # ==================== Test-Based Optimization Tests (New Behavior) ====================

    def test_should_optimize_low_success_rate(
        self, default_rules, high_quality_analysis, poor_test_results_low_success
    ):
        """Should optimize when test success rate is low (NEW)."""
        result = default_rules.should_optimize(
            high_quality_analysis,
            test_results=poor_test_results_low_success
        )

        assert result is True

    def test_should_optimize_high_latency(
        self, default_rules, high_quality_analysis, poor_test_results_high_latency
    ):
        """Should optimize when average latency is high (NEW)."""
        result = default_rules.should_optimize(
            high_quality_analysis,
            test_results=poor_test_results_high_latency
        )

        assert result is True

    def test_should_optimize_high_cost(
        self, default_rules, high_quality_analysis, poor_test_results_high_cost
    ):
        """Should optimize when cost per request is high (NEW)."""
        result = default_rules.should_optimize(
            high_quality_analysis,
            test_results=poor_test_results_high_cost
        )

        assert result is True

    def test_should_optimize_excessive_timeouts(
        self, default_rules, high_quality_analysis, poor_test_results_timeouts
    ):
        """Should optimize when timeout error rate is excessive (NEW)."""
        result = default_rules.should_optimize(
            high_quality_analysis,
            test_results=poor_test_results_timeouts
        )

        assert result is True

    def test_should_not_optimize_all_metrics_good(
        self, default_rules, high_quality_analysis, good_test_results
    ):
        """Should not optimize when all metrics are good (NEW)."""
        result = default_rules.should_optimize(
            high_quality_analysis,
            test_results=good_test_results
        )

        assert result is False

    def test_should_optimize_custom_thresholds(self, high_quality_analysis, good_test_results):
        """Should respect custom thresholds (NEW)."""
        strict_rules = ScoringRules(
            min_success_rate=0.99,  # Very strict
            max_acceptable_latency_ms=500.0,  # Very strict
            max_cost_per_request=0.005  # Very strict
        )

        # good_test_results: success_rate=0.95, latency=1000ms, cost=0.01
        # All of these should trigger optimization with strict thresholds
        assert strict_rules.should_optimize(
            high_quality_analysis,
            test_results=good_test_results
        ) is True

    # ==================== Backward Compatibility Tests ====================

    def test_should_optimize_none_test_results(
        self, default_rules, high_quality_analysis
    ):
        """Should work with None test_results (backward compatible)."""
        result = default_rules.should_optimize(
            high_quality_analysis,
            test_results=None
        )

        assert result is False  # High quality, no test data

    def test_should_optimize_legacy_baseline_metrics(
        self, default_rules, high_quality_analysis
    ):
        """Should support legacy baseline_metrics dict (backward compatible)."""
        legacy_metrics = {
            "success_rate": 0.75,  # Below 0.8 threshold
            "total_tests": 100,  # Important: specify total_tests to avoid rounding
            "avg_response_time": 2.0  # seconds
        }

        result = default_rules.should_optimize(
            high_quality_analysis,
            baseline_metrics=legacy_metrics
        )

        assert result is True  # Low success rate

    def test_should_optimize_legacy_baseline_metrics_good(
        self, default_rules, high_quality_analysis
    ):
        """Should not optimize with good legacy metrics (backward compatible)."""
        legacy_metrics = {
            "success_rate": 0.95,
            "total_tests": 100,  # Important: specify total_tests
            "avg_response_time": 1.0
        }

        result = default_rules.should_optimize(
            high_quality_analysis,
            baseline_metrics=legacy_metrics
        )

        assert result is False

    def test_should_optimize_test_results_takes_precedence(
        self, default_rules, high_quality_analysis, good_test_results
    ):
        """test_results should take precedence over baseline_metrics."""
        legacy_metrics = {
            "success_rate": 0.5  # Should trigger optimization
        }

        # But test_results is good, so should not optimize
        result = default_rules.should_optimize(
            high_quality_analysis,
            test_results=good_test_results,
            baseline_metrics=legacy_metrics
        )

        assert result is False  # test_results takes precedence

    # ==================== Strategy Selection Tests ====================

    def test_select_strategy_clarity_focus_low_clarity(self, default_rules):
        """Should select clarity_focus when clarity score is low."""
        analysis = PromptAnalysis(
            prompt_id="test",
            overall_score=70.0,
            clarity_score=60.0,  # Low
            efficiency_score=80.0,  # High
            issues=[],
            suggestions=[]
        )

        strategy = default_rules.select_strategy(analysis)
        assert strategy == "clarity_focus"

    def test_select_strategy_efficiency_focus_low_efficiency(self, default_rules):
        """Should select efficiency_focus when efficiency score is low."""
        analysis = PromptAnalysis(
            prompt_id="test",
            overall_score=70.0,
            clarity_score=80.0,  # High
            efficiency_score=60.0,  # Low
            issues=[],
            suggestions=[]
        )

        strategy = default_rules.select_strategy(analysis)
        assert strategy == "efficiency_focus"

    def test_select_strategy_structure_focus_both_low(self, default_rules):
        """Should select structure_focus when both scores are low."""
        analysis = PromptAnalysis(
            prompt_id="test",
            overall_score=60.0,
            clarity_score=65.0,  # Low
            efficiency_score=65.0,  # Low
            issues=[],
            suggestions=[]
        )

        strategy = default_rules.select_strategy(analysis)
        assert strategy == "structure_focus"

    def test_select_strategy_clarity_focus_high_timeouts(
        self, default_rules, poor_test_results_timeouts
    ):
        """Should select clarity_focus when timeout rate is high (NEW)."""
        analysis = PromptAnalysis(
            prompt_id="test",
            overall_score=75.0,
            clarity_score=75.0,
            efficiency_score=75.0,
            issues=[],
            suggestions=[]
        )

        strategy = default_rules.select_strategy(analysis, test_results=poor_test_results_timeouts)
        assert strategy == "clarity_focus"

    def test_select_strategy_efficiency_focus_low_efficiency_score(
        self, default_rules, good_test_results
    ):
        """Should select efficiency_focus when efficiency score is low with test data (NEW)."""
        analysis = PromptAnalysis(
            prompt_id="test",
            overall_score=70.0,
            clarity_score=80.0,
            efficiency_score=65.0,  # Low, below low_score_threshold (70.0)
            issues=[],
            suggestions=[]
        )

        strategy = default_rules.select_strategy(analysis, test_results=good_test_results)
        assert strategy == "efficiency_focus"

    def test_select_strategy_backward_compatible(self, default_rules):
        """Should work without test_results (backward compatible)."""
        analysis = PromptAnalysis(
            prompt_id="test",
            overall_score=85.0,
            clarity_score=85.0,
            efficiency_score=85.0,
            issues=[],
            suggestions=[]
        )

        strategy = default_rules.select_strategy(analysis)
        assert strategy == "clarity_focus"  # Default

    # ==================== Helper Method Tests ====================

    def test_convert_legacy_baseline_metrics_valid(self, default_rules):
        """Should convert valid legacy metrics to TestExecutionReport."""
        legacy = {
            "success_rate": 0.85,
            "workflow_id": "wf_001",
            "run_id": "run_001",
            "total_tests": 100,
            "avg_response_time": 1.5,  # seconds
            "total_tokens": 10000,
            "avg_tokens": 100.0,
            "total_cost": 2.0,
            "avg_cost": 0.02
        }

        report = default_rules._convert_legacy_baseline_metrics(legacy)

        assert report is not None
        assert report.success_rate == 0.85
        assert report.workflow_id == "wf_001"
        assert report.run_id == "run_001"
        assert report.total_tests == 100
        assert report.avg_response_time_ms == 1500.0  # Converted to ms
        assert report.total_tokens == 10000
        assert report.avg_tokens_per_request == 100.0
        assert report.total_cost == 2.0
        assert report.cost_per_request == 0.02

    def test_convert_legacy_baseline_metrics_minimal(self, default_rules):
        """Should convert minimal legacy metrics (only success_rate)."""
        legacy = {"success_rate": 0.90}

        report = default_rules._convert_legacy_baseline_metrics(legacy)

        assert report is not None
        # With total_tests=1, success_rate=0.9 rounds to 1 successful test
        # So actual_success_rate = 1/1 = 1.0
        assert report.success_rate == 1.0  # Recalculated from counts
        assert report.workflow_id == "unknown"
        assert report.run_id == "legacy"
        assert report.total_tests == 1
        assert report.avg_response_time_ms == 0.0

    def test_convert_legacy_baseline_metrics_none(self, default_rules):
        """Should return None for None input."""
        assert default_rules._convert_legacy_baseline_metrics(None) is None

    def test_convert_legacy_baseline_metrics_empty(self, default_rules):
        """Should return None for empty dict."""
        assert default_rules._convert_legacy_baseline_metrics({}) is None

    def test_convert_legacy_baseline_metrics_no_success_rate(self, default_rules):
        """Should return None when success_rate missing."""
        legacy = {"total_tests": 100, "avg_response_time": 1.0}

        assert default_rules._convert_legacy_baseline_metrics(legacy) is None

    # ==================== Config Override Tests ====================

    def test_should_optimize_config_override_threshold(self, high_quality_analysis):
        """Should use config's score_threshold to override default."""
        from dataclasses import dataclass

        @dataclass
        class MockConfig:
            score_threshold: float = 95.0  # Very high threshold

        rules = ScoringRules(optimization_threshold=80.0)
        config = MockConfig()

        # Analysis score is 90.0, which is:
        # - Above default threshold (80.0) -> would NOT optimize
        # - Below config threshold (95.0) -> SHOULD optimize
        result = rules.should_optimize(high_quality_analysis, config=config)

        assert result is True

    # ==================== Version Bump Type Tests (Existing) ====================

    def test_version_bump_type_major(self, default_rules):
        """Should return 'major' for large improvements."""
        assert default_rules.version_bump_type(20.0) == "major"

    def test_version_bump_type_minor(self, default_rules):
        """Should return 'minor' for medium improvements."""
        assert default_rules.version_bump_type(8.0) == "minor"

    def test_version_bump_type_patch(self, default_rules):
        """Should return 'patch' for small improvements."""
        assert default_rules.version_bump_type(2.0) == "patch"

    # ==================== Is High Quality Tests (Existing) ====================

    def test_is_high_quality_true(self, default_rules):
        """Should return True for high quality optimization."""
        assert default_rules.is_high_quality(confidence=0.85, improvement=8.0) is True

    def test_is_high_quality_low_confidence(self, default_rules):
        """Should return False for low confidence."""
        assert default_rules.is_high_quality(confidence=0.5, improvement=8.0) is False

    def test_is_high_quality_low_improvement(self, default_rules):
        """Should return False for low improvement."""
        assert default_rules.is_high_quality(confidence=0.85, improvement=2.0) is False
