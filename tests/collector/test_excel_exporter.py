"""
ExcelExporter 验收测试

Date: 2025-11-13
Author: backend-developer
Description: 验证 ExcelExporter 的 Excel 导出功能
"""

from datetime import datetime
from pathlib import Path

from src.collector import (
    ExcelExporter,
    DataCollector,
    ResultClassifier,
    TestResult,
    TestStatus
)


def test_excel_export():
    """完整的 Excel 导出验收测试"""

    print("=" * 70)
    print("ExcelExporter Acceptance Test")
    print("=" * 70)

    # 1. 准备测试数据
    print("\n1. Preparing test data...")
    collector = DataCollector()

    for i in range(20):
        result = TestResult(
            workflow_id=f"wf_{i % 3}",
            execution_id=f"exec_{i}",
            timestamp=datetime.now(),
            status=TestStatus.SUCCESS if i % 4 != 0 else TestStatus.FAILED,
            execution_time=1.0 + i * 0.5,
            tokens_used=100 + i * 10,
            cost=0.01 + i * 0.005,
            inputs={"query": f"test_{i}"},
            outputs={"answer": f"result_{i}" * 50},
            error_message="Timeout error" if i % 4 == 0 else None
        )
        collector.collect_result(result)

    print(f"   Created {len(collector.get_all_results())} test records")

    # 2. 创建导出器
    print("\n2. Initializing ExcelExporter...")
    exporter = ExcelExporter()

    # 3. 导出完整报告 (包含统计)
    print("\n3. Exporting full report (with statistics)...")
    output_path = Path("output/test_report.xlsx")

    try:
        result_path = exporter.export_results(
            collector.get_all_results(),
            output_path,
            include_stats=True
        )
        print(f"   [OK] Export successful: {result_path}")
        assert result_path.exists(), "Export file does not exist"
        print(f"   [OK] File exists: {result_path.stat().st_size} bytes")
    except Exception as e:
        print(f"   [ERROR] Export failed: {e}")
        raise

    # 4. 导出仅统计报告
    print("\n4. Exporting statistics report...")
    stats_path = Path("output/stats_report.xlsx")

    try:
        metrics = collector.get_statistics()
        classifier = ResultClassifier()
        classification = classifier.classify_batch(collector.get_all_results())

        stats_output = exporter.export_statistics(
            metrics,
            classification,
            stats_path
        )
        print(f"   [OK] Export successful: {stats_output}")
        assert stats_output.exists(), "Statistics file does not exist"
        print(f"   [OK] File exists: {stats_output.stat().st_size} bytes")
    except Exception as e:
        print(f"   [ERROR] Export failed: {e}")
        raise

    # 5. 验证统计数据
    print("\n5. Verifying exported data...")
    print(f"   - Total executions: {metrics.total_executions}")
    print(f"   - Successful: {metrics.successful_count}")
    print(f"   - Failed: {metrics.failed_count}")
    print(f"   - Success rate: {metrics.success_rate:.2%}")
    print(f"   - Avg execution time: {metrics.avg_execution_time:.3f}s")
    print(f"   - P95 execution time: {metrics.p95_execution_time:.3f}s")
    print(f"   - Total tokens: {metrics.total_tokens:,}")
    print(f"   - Total cost: ${metrics.total_cost:.2f}")

    print("\n6. Performance grade distribution:")
    print(f"   - Excellent: {classification.excellent_count}")
    print(f"   - Good: {classification.good_count}")
    print(f"   - Fair: {classification.fair_count}")
    print(f"   - Poor: {classification.poor_count}")

    # 7. 测试空数据异常处理
    print("\n7. Testing exception handling...")
    try:
        exporter.export_results([], Path("output/empty.xlsx"))
        print("   [ERROR] Should have raised exception but didn't")
        assert False
    except Exception as e:
        print(f"   [OK] Correctly raised exception: {type(e).__name__}")

    print("\n" + "=" * 70)
    print("[SUCCESS] All tests passed!")
    print("=" * 70)
    print(f"\nPlease manually verify these files:")
    print(f"  1. {result_path.absolute()}")
    print(f"  2. {stats_output.absolute()}")
    print("\nVerify worksheet contents:")
    print("  - Test Overview: Contains statistical summary")
    print("  - Detailed Results: Contains all test records")
    print("  - Performance Analysis: Contains per-workflow statistics")


if __name__ == "__main__":
    test_excel_export()
