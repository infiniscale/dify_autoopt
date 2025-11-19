"""
Optimizer Module - Configurable Scoring Rules

Date: 2025-11-18
Author: backend-developer
Description: Configurable scoring thresholds and rules for optimization decisions.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .models import PromptAnalysis, TestExecutionReport


@dataclass
class ScoringRules:
    """Configurable scoring thresholds and rules.

    Attributes:
        # Static Analysis Thresholds
        optimization_threshold: Overall score below which to optimize
        critical_issue_threshold: Number of critical issues to trigger optimization
        clarity_efficiency_gap: Score gap to prefer one strategy over another
        low_score_threshold: Score considered "low"
        min_confidence: Minimum confidence to accept optimization
        high_confidence: Confidence considered "high quality"
        major_version_min_improvement: Min improvement for major version bump
        minor_version_min_improvement: Min improvement for minor version bump

        # Test-Based Optimization Triggers (Phase 1)
        min_success_rate: Minimum success rate to avoid optimization (0.0-1.0)
        max_acceptable_latency_ms: Maximum acceptable average latency (milliseconds)
        max_cost_per_request: Maximum acceptable cost per request (USD)
        max_timeout_error_rate: Maximum acceptable timeout error rate (0.0-1.0)
    """

    # Static analysis triggers (EXISTING)
    optimization_threshold: float = 80.0
    critical_issue_threshold: int = 1

    # Strategy selection (EXISTING)
    clarity_efficiency_gap: float = 10.0
    low_score_threshold: float = 70.0

    # Confidence thresholds (EXISTING)
    min_confidence: float = 0.6
    high_confidence: float = 0.8

    # Version management (EXISTING)
    major_version_min_improvement: float = 15.0
    minor_version_min_improvement: float = 5.0

    # Test-based optimization triggers (NEW - Phase 1)
    min_success_rate: float = 0.8
    max_acceptable_latency_ms: float = 5000.0
    max_cost_per_request: float = 0.1
    max_timeout_error_rate: float = 0.05

    @classmethod
    def from_config(cls, config_dict: Dict[str, Any]) -> "ScoringRules":
        """Load from configuration dict.

        Args:
            config_dict: Configuration dictionary

        Returns:
            ScoringRules instance
        """
        # Filter to only valid fields
        valid_fields = {
            k: v
            for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        }
        return cls(**valid_fields)

    def should_optimize(
        self,
        analysis: PromptAnalysis,
        test_results: Optional[TestExecutionReport] = None,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        config: Optional[Any] = None,
    ) -> bool:
        """Determine if optimization is needed based on static analysis AND test results.

        Decision Logic:
            1. Static Analysis Criteria (unchanged):
               - Low overall score (< optimization_threshold)
               - Critical issues present

            2. Test-Based Criteria (NEW - Phase 1):
               - Low success rate (< min_success_rate)
               - High latency (> max_acceptable_latency_ms)
               - High cost (> max_cost_per_request)
               - Excessive timeout errors (> max_timeout_error_rate)

        Args:
            analysis: Prompt analysis result (static)
            test_results: Test execution report (optional, runtime metrics)
            baseline_metrics: DEPRECATED - Legacy dict format, use test_results instead
            config: Optional config to override score_threshold

        Returns:
            True if optimization recommended

        Backward Compatibility:
            - When test_results=None and baseline_metrics=None, behaves identically to old version
            - baseline_metrics dict is auto-converted to TestExecutionReport if provided
            - Existing code continues to work without changes

        Example:
            >>> from src.optimizer.models import TestExecutionReport
            >>> rules = ScoringRules()
            >>> test_report = TestExecutionReport(
            ...     workflow_id="wf_001",
            ...     run_id="run_001",
            ...     total_tests=100,
            ...     successful_tests=95,
            ...     failed_tests=5,
            ...     success_rate=0.95,
            ...     avg_response_time_ms=1000.0,
            ...     total_tokens=10000,
            ...     avg_tokens_per_request=100.0,
            ...     total_cost=1.0,
            ...     cost_per_request=0.01
            ... )
            >>> should_opt = rules.should_optimize(analysis, test_results=test_report)
        """
        # Backward compatibility: Convert legacy baseline_metrics if test_results not provided
        if test_results is None and baseline_metrics:
            test_results = self._convert_legacy_baseline_metrics(baseline_metrics)

        # Use config's score_threshold if provided
        score_threshold = (
            config.score_threshold if config else self.optimization_threshold
        )

        # ========== Static Analysis Criteria (UNCHANGED) ==========

        # 1. Low overall score
        if analysis.overall_score < score_threshold:
            return True

        # 2. Critical issues present
        critical_issues = [
            i
            for i in analysis.issues
            if hasattr(i.severity, "value") and i.severity.value == "critical"
        ]
        if len(critical_issues) >= self.critical_issue_threshold:
            return True

        # ========== Test-Based Criteria (NEW - Phase 1) ==========

        if test_results:
            # 3. Low success rate
            if test_results.success_rate < self.min_success_rate:
                return True

            # 4. High average latency
            if test_results.avg_response_time_ms > self.max_acceptable_latency_ms:
                return True

            # 5. High cost per request
            if test_results.cost_per_request > self.max_cost_per_request:
                return True

            # 6. Excessive timeout errors
            timeout_rate = test_results.get_timeout_error_rate()
            if timeout_rate > self.max_timeout_error_rate:
                return True

        return False

    def select_strategy(
        self,
        analysis: PromptAnalysis,
        test_results: Optional[TestExecutionReport] = None,
        baseline_metrics: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Select best optimization strategy based on analysis and test results.

        Strategy Selection Logic:
            - Clarity focus: Low clarity score OR high timeout errors (suggests unclear prompts)
            - Efficiency focus: Low efficiency score OR high token usage
            - Structure focus: Both scores low OR mixed error patterns

        Args:
            analysis: Prompt analysis
            test_results: Test execution report (optional)
            baseline_metrics: DEPRECATED - Legacy dict format, use test_results instead

        Returns:
            Strategy name

        Example:
            >>> strategy = rules.select_strategy(analysis, test_results)
            >>> assert strategy in ["clarity_focus", "efficiency_focus", "structure_focus"]
        """
        # Backward compatibility: Convert legacy baseline_metrics if test_results not provided
        if test_results is None and baseline_metrics:
            test_results = self._convert_legacy_baseline_metrics(baseline_metrics)

        clarity = analysis.clarity_score
        efficiency = analysis.efficiency_score

        # Factor in test results if available (NEW - Phase 1)
        if test_results:
            # High timeout errors suggest clarity issues
            if test_results.has_timeout_errors():
                timeout_rate = test_results.get_timeout_error_rate()
                if timeout_rate > 0.1:  # More than 10% timeouts
                    return "clarity_focus"

            # High token usage with low efficiency score
            # (Tokens significantly above what's expected for complexity)
            if efficiency < self.low_score_threshold:
                # Could add more sophisticated token analysis here
                return "efficiency_focus"

        # Original logic (unchanged)
        # Clarity significantly lower
        if clarity < efficiency - self.clarity_efficiency_gap:
            return "clarity_focus"

        # Efficiency significantly lower
        if efficiency < clarity - self.clarity_efficiency_gap:
            return "efficiency_focus"

        # Both scores low
        if clarity < self.low_score_threshold and efficiency < self.low_score_threshold:
            return "structure_focus"

        # Default
        return "clarity_focus"

    def version_bump_type(self, improvement_score: float) -> str:
        """Determine version bump type based on improvement.

        Args:
            improvement_score: Score improvement (optimized - original)

        Returns:
            "major", "minor", or "patch"
        """
        if improvement_score >= self.major_version_min_improvement:
            return "major"
        elif improvement_score >= self.minor_version_min_improvement:
            return "minor"
        else:
            return "patch"

    def is_high_quality(self, confidence: float, improvement: float) -> bool:
        """Check if optimization result is high quality.

        Args:
            confidence: Confidence score
            improvement: Improvement score

        Returns:
            True if considered high quality
        """
        return (
            confidence >= self.high_confidence
            and improvement >= self.minor_version_min_improvement
        )

    def _convert_legacy_baseline_metrics(
        self, baseline_metrics: Optional[Dict[str, Any]]
    ) -> Optional[TestExecutionReport]:
        """Convert legacy baseline_metrics dict to TestExecutionReport.

        This provides backward compatibility for existing code using the old API.
        The method creates a minimal TestExecutionReport from legacy dict data.

        Args:
            baseline_metrics: Legacy dict with optional keys:
                - success_rate (required)
                - workflow_id, run_id (optional)
                - total_tests (optional, defaults to 1)
                - avg_response_time (optional, in seconds)
                - total_tokens, avg_tokens (optional)
                - total_cost, avg_cost (optional)

        Returns:
            TestExecutionReport if valid data provided, None otherwise

        Example:
            >>> legacy = {"success_rate": 0.75, "avg_response_time": 2.0}
            >>> report = rules._convert_legacy_baseline_metrics(legacy)
            >>> assert report.success_rate == 0.75
            >>> assert report.avg_response_time_ms == 2000.0
        """
        if not baseline_metrics or "success_rate" not in baseline_metrics:
            return None

        total_tests = baseline_metrics.get("total_tests", 1)
        success_rate = baseline_metrics["success_rate"]

        # Calculate test counts from success_rate
        # Use round() to ensure success_rate matches the ratio
        successful_tests = round(total_tests * success_rate)
        failed_tests = total_tests - successful_tests

        # Recalculate success_rate to match test counts (avoid floating point errors)
        actual_success_rate = successful_tests / total_tests if total_tests > 0 else 0.0

        # Convert response time from seconds to ms (if provided)
        avg_response_time_ms = baseline_metrics.get("avg_response_time", 0.0) * 1000

        return TestExecutionReport(
            workflow_id=baseline_metrics.get("workflow_id", "unknown"),
            run_id=baseline_metrics.get("run_id", "legacy"),
            total_tests=total_tests,
            successful_tests=successful_tests,
            failed_tests=failed_tests,
            success_rate=actual_success_rate,  # Use calculated rate
            avg_response_time_ms=avg_response_time_ms,
            total_tokens=baseline_metrics.get("total_tokens", 0),
            avg_tokens_per_request=baseline_metrics.get("avg_tokens", 0.0),
            total_cost=baseline_metrics.get("total_cost", 0.0),
            cost_per_request=baseline_metrics.get("avg_cost", 0.0),
        )
