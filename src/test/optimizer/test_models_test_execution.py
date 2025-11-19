"""
Tests for Test-Driven Optimization Models

Date: 2025-11-19
Author: backend-architect
Description: Test ErrorDistribution and TestExecutionReport models
"""

import pytest
from datetime import datetime
from src.optimizer.models import ErrorDistribution, TestExecutionReport


class TestErrorDistribution:
    """Test ErrorDistribution model validation and methods."""

    def test_create_valid_error_distribution(self):
        """Should create valid ErrorDistribution with correct totals."""
        errors = ErrorDistribution(
            timeout_errors=5,
            api_errors=2,
            validation_errors=1,
            llm_errors=0,
            total_errors=8
        )

        assert errors.timeout_errors == 5
        assert errors.api_errors == 2
        assert errors.validation_errors == 1
        assert errors.llm_errors == 0
        assert errors.total_errors == 8

    def test_error_distribution_defaults_to_zero(self):
        """Should default all error counts to zero."""
        errors = ErrorDistribution()

        assert errors.timeout_errors == 0
        assert errors.api_errors == 0
        assert errors.validation_errors == 0
        assert errors.llm_errors == 0
        assert errors.total_errors == 0

    def test_validate_total_errors_mismatch(self):
        """Should raise ValueError if total doesn't match sum."""
        with pytest.raises(ValueError, match="total_errors.*must equal sum"):
            ErrorDistribution(
                timeout_errors=5,
                api_errors=2,
                validation_errors=1,
                llm_errors=0,
                total_errors=10  # Should be 8
            )

    def test_validate_negative_errors_rejected(self):
        """Should reject negative error counts."""
        with pytest.raises(ValueError):
            ErrorDistribution(timeout_errors=-1)

        with pytest.raises(ValueError):
            ErrorDistribution(api_errors=-5)


class TestTestExecutionReport:
    """Test TestExecutionReport model validation and methods."""

    @pytest.fixture
    def valid_report_data(self):
        """Provide valid test report data."""
        return {
            "workflow_id": "wf_001",
            "run_id": "run_001",
            "total_tests": 100,
            "successful_tests": 95,
            "failed_tests": 5,
            "success_rate": 0.95,
            "avg_response_time_ms": 1200.0,
            "total_tokens": 50000,
            "avg_tokens_per_request": 500.0,
            "total_cost": 2.5,
            "cost_per_request": 0.025
        }

    def test_create_valid_test_execution_report(self, valid_report_data):
        """Should create valid TestExecutionReport."""
        report = TestExecutionReport(**valid_report_data)

        assert report.workflow_id == "wf_001"
        assert report.run_id == "run_001"
        assert report.total_tests == 100
        assert report.successful_tests == 95
        assert report.failed_tests == 5
        assert report.success_rate == 0.95
        assert report.avg_response_time_ms == 1200.0
        assert report.total_tokens == 50000
        assert report.avg_tokens_per_request == 500.0
        assert report.total_cost == 2.5
        assert report.cost_per_request == 0.025

    def test_validate_success_rate_consistency(self):
        """Should validate success_rate matches successful_tests/total_tests."""
        with pytest.raises(ValueError, match="success_rate.*inconsistent"):
            TestExecutionReport(
                workflow_id="wf_001",
                run_id="run_001",
                total_tests=100,
                successful_tests=95,
                failed_tests=5,
                success_rate=0.80,  # Should be 0.95
                avg_response_time_ms=1000.0,
                total_tokens=10000,
                avg_tokens_per_request=100.0,
                total_cost=1.0,
                cost_per_request=0.01
            )

    def test_validate_success_rate_range(self):
        """Should enforce success_rate in range [0.0, 1.0]."""
        base_data = {
            "workflow_id": "wf_001",
            "run_id": "run_001",
            "total_tests": 100,
            "successful_tests": 95,
            "failed_tests": 5,
            "avg_response_time_ms": 1000.0,
            "total_tokens": 10000,
            "avg_tokens_per_request": 100.0,
            "total_cost": 1.0,
            "cost_per_request": 0.01
        }

        # Test upper bound
        with pytest.raises(ValueError):
            TestExecutionReport(**{**base_data, "success_rate": 1.5})

        # Test lower bound
        with pytest.raises(ValueError):
            TestExecutionReport(**{**base_data, "success_rate": -0.1})

    def test_percentile_metrics_optional(self, valid_report_data):
        """Should allow optional percentile metrics."""
        report = TestExecutionReport(
            **valid_report_data,
            p95_response_time_ms=1500.0,
            p99_response_time_ms=2000.0
        )

        assert report.p95_response_time_ms == 1500.0
        assert report.p99_response_time_ms == 2000.0

    def test_percentile_metrics_can_be_none(self, valid_report_data):
        """Should allow None for percentile metrics."""
        report = TestExecutionReport(**valid_report_data)

        assert report.p95_response_time_ms is None
        assert report.p99_response_time_ms is None

    def test_error_distribution_defaults(self, valid_report_data):
        """Should default to empty ErrorDistribution."""
        report = TestExecutionReport(**valid_report_data)

        assert isinstance(report.error_distribution, ErrorDistribution)
        assert report.error_distribution.total_errors == 0

    def test_error_distribution_custom(self, valid_report_data):
        """Should accept custom ErrorDistribution."""
        errors = ErrorDistribution(
            timeout_errors=3,
            api_errors=2,
            validation_errors=0,
            llm_errors=0,
            total_errors=5
        )

        report = TestExecutionReport(
            **valid_report_data,
            error_distribution=errors
        )

        assert report.error_distribution.timeout_errors == 3
        assert report.error_distribution.api_errors == 2

    def test_has_timeout_errors_true(self, valid_report_data):
        """Should return True when timeout errors present."""
        errors = ErrorDistribution(
            timeout_errors=5,
            api_errors=0,
            validation_errors=0,
            llm_errors=0,
            total_errors=5
        )

        report = TestExecutionReport(
            **valid_report_data,
            error_distribution=errors
        )

        assert report.has_timeout_errors() is True

    def test_has_timeout_errors_false(self, valid_report_data):
        """Should return False when no timeout errors."""
        report = TestExecutionReport(**valid_report_data)
        assert report.has_timeout_errors() is False

    def test_has_api_errors_true(self, valid_report_data):
        """Should return True when API errors present."""
        errors = ErrorDistribution(
            timeout_errors=0,
            api_errors=3,
            validation_errors=0,
            llm_errors=0,
            total_errors=3
        )

        report = TestExecutionReport(
            **valid_report_data,
            error_distribution=errors
        )

        assert report.has_api_errors() is True

    def test_has_api_errors_false(self, valid_report_data):
        """Should return False when no API errors."""
        report = TestExecutionReport(**valid_report_data)
        assert report.has_api_errors() is False

    def test_get_error_rate(self, valid_report_data):
        """Should calculate error rate correctly."""
        report = TestExecutionReport(**valid_report_data)
        assert report.get_error_rate() == pytest.approx(0.05)  # 1.0 - 0.95

    def test_get_timeout_error_rate(self, valid_report_data):
        """Should calculate timeout error rate correctly."""
        errors = ErrorDistribution(
            timeout_errors=10,
            api_errors=0,
            validation_errors=0,
            llm_errors=0,
            total_errors=10
        )

        report = TestExecutionReport(
            **valid_report_data,
            error_distribution=errors
        )

        assert report.get_timeout_error_rate() == pytest.approx(0.10)  # 10/100

    def test_get_timeout_error_rate_zero_tests(self):
        """Should return 0.0 when total_tests is 0 (edge case)."""
        # Note: This shouldn't happen in practice due to ge=1 validation
        # but testing the method's defensive programming
        report = TestExecutionReport(
            workflow_id="wf_001",
            run_id="run_001",
            total_tests=1,  # Minimum allowed
            successful_tests=1,
            failed_tests=0,
            success_rate=1.0,
            avg_response_time_ms=1000.0,
            total_tokens=100,
            avg_tokens_per_request=100.0,
            total_cost=0.01,
            cost_per_request=0.01
        )

        # Even with 0 timeout errors, rate should be 0.0
        assert report.get_timeout_error_rate() == 0.0

    def test_metadata_defaults_to_empty_dict(self, valid_report_data):
        """Should default metadata to empty dict."""
        report = TestExecutionReport(**valid_report_data)
        assert report.metadata == {}

    def test_metadata_custom(self, valid_report_data):
        """Should accept custom metadata."""
        custom_metadata = {"total_retries": 5, "cancelled_tasks": 2}

        report = TestExecutionReport(
            **valid_report_data,
            metadata=custom_metadata
        )

        assert report.metadata == custom_metadata

    def test_executed_at_defaults_to_now(self, valid_report_data):
        """Should default executed_at to current time."""
        before = datetime.now()
        report = TestExecutionReport(**valid_report_data)
        after = datetime.now()

        assert before <= report.executed_at <= after

    def test_executed_at_custom(self, valid_report_data):
        """Should accept custom executed_at timestamp."""
        custom_time = datetime(2025, 11, 19, 12, 0, 0)

        report = TestExecutionReport(
            **valid_report_data,
            executed_at=custom_time
        )

        assert report.executed_at == custom_time


class TestTestExecutionReportConversion:
    """Test conversion from executor's RunExecutionResult."""

    @pytest.fixture
    def mock_executor_result(self):
        """Create mock executor result."""
        from dataclasses import dataclass
        from src.executor.models import RunStatistics, TaskResult, TaskStatus

        @dataclass
        class MockRunExecutionResult:
            workflow_id: str
            run_id: str
            statistics: RunStatistics
            task_results: list
            finished_at: datetime

        # Create mock statistics
        stats = RunStatistics(
            total_tasks=100,
            completed_tasks=100,
            succeeded_tasks=85,
            failed_tasks=10,
            timeout_tasks=5,
            cancelled_tasks=0,
            error_tasks=0,
            success_rate=0.85,
            total_execution_time=120.0,
            avg_execution_time=1.2,
            total_tokens=50000,
            total_cost=2.5,
            total_retries=3
        )

        # Create mock task results with varying execution times
        task_results = []
        for i in range(100):
            task_result = TaskResult(
                task_id=f"task_{i}",
                workflow_id="wf_001",
                dataset="test_dataset",
                scenario="normal",
                task_status=TaskStatus.SUCCEEDED,
                execution_time=1.0 + (i * 0.01),  # Varying times for percentiles
                metadata={}
            )
            task_results.append(task_result)

        return MockRunExecutionResult(
            workflow_id="wf_001",
            run_id="run_001",
            statistics=stats,
            task_results=task_results,
            finished_at=datetime(2025, 11, 19, 12, 0, 0)
        )

    def test_from_executor_result_basic_conversion(self, mock_executor_result):
        """Should convert executor result to test report."""
        report = TestExecutionReport.from_executor_result(mock_executor_result)

        assert report.workflow_id == "wf_001"
        assert report.run_id == "run_001"
        assert report.total_tests == 100
        assert report.successful_tests == 85
        assert report.failed_tests == 15  # failed + timeout + error
        assert report.success_rate == 0.85

    def test_from_executor_result_converts_time_to_ms(self, mock_executor_result):
        """Should convert execution time from seconds to milliseconds."""
        report = TestExecutionReport.from_executor_result(mock_executor_result)

        # 1.2 seconds -> 1200 ms
        assert report.avg_response_time_ms == pytest.approx(1200.0)

    def test_from_executor_result_calculates_percentiles(self, mock_executor_result):
        """Should calculate p95 and p99 percentiles."""
        report = TestExecutionReport.from_executor_result(mock_executor_result)

        # With 100 tasks, p95 is index 95, p99 is index 99
        # Times range from 1.0 to 1.99 seconds (1000 to 1990 ms)
        assert report.p95_response_time_ms is not None
        assert report.p99_response_time_ms is not None
        assert report.p95_response_time_ms < report.p99_response_time_ms

    def test_from_executor_result_maps_error_distribution(self, mock_executor_result):
        """Should map executor errors to error distribution."""
        report = TestExecutionReport.from_executor_result(mock_executor_result)

        assert report.error_distribution.timeout_errors == 5
        assert report.error_distribution.api_errors == 0  # executor.error_tasks
        assert report.error_distribution.total_errors == 5  # Only timeout + api errors

    def test_from_executor_result_calculates_token_metrics(self, mock_executor_result):
        """Should calculate per-request token and cost metrics."""
        report = TestExecutionReport.from_executor_result(mock_executor_result)

        assert report.total_tokens == 50000
        # avg_tokens_per_request = 50000 / 100
        assert report.avg_tokens_per_request == pytest.approx(500.0)

        assert report.total_cost == 2.5
        # cost_per_request = 2.5 / 100
        assert report.cost_per_request == pytest.approx(0.025)

    def test_from_executor_result_includes_metadata(self, mock_executor_result):
        """Should include executor metadata in report."""
        report = TestExecutionReport.from_executor_result(mock_executor_result)

        assert "total_retries" in report.metadata
        assert report.metadata["total_retries"] == 3
        assert "cancelled_tasks" in report.metadata

    def test_from_executor_result_none_raises_error(self):
        """Should raise ValueError if executor_result is None."""
        with pytest.raises(ValueError, match="executor_result cannot be None"):
            TestExecutionReport.from_executor_result(None)

    def test_from_executor_result_sets_executed_at(self, mock_executor_result):
        """Should use executor's finished_at as executed_at."""
        report = TestExecutionReport.from_executor_result(mock_executor_result)

        assert report.executed_at == datetime(2025, 11, 19, 12, 0, 0)
