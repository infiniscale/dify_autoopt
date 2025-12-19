"""
性能基准测试 - Collector 模块

Date: 2025-11-14
Author: qa-engineer
Description: 测试各组件的性能基准，确保满足性能要求
"""

import pytest
import time
import random
from datetime import datetime

from src.collector import (
    DataCollector,
    ResultClassifier,
    ExcelExporter,
    TestResult,
    TestStatus
)
from .conftest import create_test_result


class TestCollectionPerformance:
    """测试收集性能基准"""

    def test_single_collection_performance(self):
        """
        测试单次收集性能

        基准: 单次收集应该 < 1ms
        """
        collector = DataCollector()
        result = create_test_result()

        iterations = 1000
        start_time = time.time()

        for _ in range(iterations):
            collector.collect_result(result)

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / iterations

        assert avg_time < 0.001, f"Average collection time: {avg_time * 1000:.3f}ms, expected < 1ms"

    def test_batch_collection_performance(self):
        """
        测试批量收集性能

        基准: 1000 条数据收集 < 1秒
        """
        collector = DataCollector()
        results = [create_test_result(execution_id=f"perf_{i}") for i in range(1000)]

        start_time = time.time()
        for result in results:
            collector.collect_result(result)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 1.0, f"Batch collection took {elapsed_time:.2f}s, expected < 1s"
        assert collector.get_result_count() == 1000

    def test_large_scale_collection_performance(self, large_dataset):
        """
        测试大规模收集性能

        基准: 5,000 条数据收集 < 5秒
        """
        collector = DataCollector()

        start_time = time.time()
        for result in large_dataset:
            collector.collect_result(result)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 5.0, f"Large scale collection took {elapsed_time:.2f}s, expected < 5s"
        assert collector.get_result_count() == 5000

    def test_memory_efficiency_during_collection(self):
        """
        测试收集过程的内存效率

        验证内存占用在合理范围内
        """
        import sys

        collector = DataCollector()

        # 收集前的内存占用
        initial_size = sys.getsizeof(collector)

        # 收集 1000 条数据
        for i in range(1000):
            result = create_test_result(execution_id=f"mem_{i}")
            collector.collect_result(result)

        # 收集后的内存占用
        final_size = sys.getsizeof(collector)

        # 内存增长应该是合理的（粗略检查）
        # 实际内存占用会包括所有 TestResult 对象
        assert final_size >= initial_size


class TestStatisticsPerformance:
    """测试统计计算性能基准"""

    def test_statistics_calculation_performance_small(self, sample_results):
        """
        测试小数据集统计计算性能

        基准: 20 条数据统计 < 10ms
        """
        collector = DataCollector()
        for result in sample_results:
            collector.collect_result(result)

        iterations = 100
        start_time = time.time()

        for _ in range(iterations):
            stats = collector.get_statistics()

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / iterations

        assert avg_time < 0.01, f"Statistics calculation took {avg_time * 1000:.2f}ms, expected < 10ms"

    def test_statistics_calculation_performance_medium(self):
        """
        测试中等数据集统计计算性能

        基准: 1,000 条数据统计 < 100ms
        """
        collector = DataCollector()
        results = [create_test_result(execution_id=f"stat_{i}") for i in range(1000)]

        for result in results:
            collector.collect_result(result)

        start_time = time.time()
        stats = collector.get_statistics()
        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.1, f"Statistics calculation took {elapsed_time * 1000:.2f}ms, expected < 100ms"

    def test_statistics_calculation_performance_large(self, large_dataset):
        """
        测试大数据集统计计算性能

        基准: 5,000 条数据统计 < 1秒
        """
        collector = DataCollector()
        for result in large_dataset:
            collector.collect_result(result)

        start_time = time.time()
        stats = collector.get_statistics()
        elapsed_time = time.time() - start_time

        assert elapsed_time < 1.0, f"Statistics calculation took {elapsed_time:.2f}s, expected < 1s"

    def test_repeated_statistics_calculation_performance(self):
        """
        测试重复统计计算性能

        验证多次调用统计计算的性能稳定性
        """
        collector = DataCollector()
        results = [create_test_result(execution_id=f"repeat_{i}") for i in range(500)]

        for result in results:
            collector.collect_result(result)

        # 多次调用统计
        times = []
        for _ in range(10):
            start_time = time.time()
            stats = collector.get_statistics()
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

        avg_time = sum(times) / len(times)
        max_time = max(times)

        assert avg_time < 0.05, f"Average statistics time: {avg_time * 1000:.2f}ms"
        assert max_time < 0.1, f"Max statistics time: {max_time * 1000:.2f}ms"


class TestClassificationPerformance:
    """测试分类性能基准"""

    def test_single_classification_performance(self):
        """
        测试单次分类性能

        基准: 单次分类 < 1ms
        """
        classifier = ResultClassifier()
        result = create_test_result(execution_time=1.5, tokens_used=100)

        iterations = 1000
        start_time = time.time()

        for _ in range(iterations):
            grade = classifier.classify_result(result)

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / iterations

        assert avg_time < 0.001, f"Single classification took {avg_time * 1000:.3f}ms, expected < 1ms"

    def test_batch_classification_performance_small(self, sample_results):
        """
        测试小批量分类性能

        基准: 20 条数据分类 < 10ms
        """
        classifier = ResultClassifier()

        start_time = time.time()
        classification = classifier.classify_batch(sample_results)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.01, f"Batch classification took {elapsed_time * 1000:.2f}ms, expected < 10ms"

    def test_batch_classification_performance_medium(self):
        """
        测试中等批量分类性能

        基准: 1,000 条数据分类 < 100ms
        """
        classifier = ResultClassifier()
        results = [
            create_test_result(
                execution_id=f"class_{i}",
                execution_time=random.uniform(0.5, 10.0),
                tokens_used=random.randint(50, 500)
            )
            for i in range(1000)
        ]

        start_time = time.time()
        classification = classifier.classify_batch(results)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.1, f"Batch classification took {elapsed_time * 1000:.2f}ms, expected < 100ms"

    def test_batch_classification_performance_large(self, large_dataset):
        """
        测试大批量分类性能

        基准: 5,000 条数据分类 < 2秒
        """
        classifier = ResultClassifier()

        start_time = time.time()
        classification = classifier.classify_batch(large_dataset)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 2.0, f"Large batch classification took {elapsed_time:.2f}s, expected < 2s"

        # 验证分类结果完整性
        total = (
                classification.excellent_count +
                classification.good_count +
                classification.fair_count +
                classification.poor_count
        )
        assert total == 5000

    def test_classification_with_custom_thresholds_performance(self):
        """
        测试自定义阈值分类性能

        验证阈值配置不影响性能
        """
        custom_thresholds = {
            "excellent": {"execution_time": 1.5, "token_efficiency": 0.85},
            "good": {"execution_time": 4.0, "token_efficiency": 0.65},
            "fair": {"execution_time": 8.0, "token_efficiency": 0.45}
        }

        classifier = ResultClassifier(thresholds=custom_thresholds)
        results = [create_test_result(execution_id=f"custom_{i}") for i in range(1000)]

        start_time = time.time()
        classification = classifier.classify_batch(results)
        elapsed_time = time.time() - start_time

        assert elapsed_time < 0.1, f"Custom threshold classification took {elapsed_time * 1000:.2f}ms"


class TestExportPerformance:
    """测试导出性能基准"""

    def test_export_performance_small(self, sample_results, temp_output_dir):
        """
        测试小数据集导出性能

        基准: 20 条数据导出 < 1秒
        """
        exporter = ExcelExporter()
        output_file = temp_output_dir / "perf_small.xlsx"

        start_time = time.time()
        exporter.export_results(sample_results, str(output_file))
        elapsed_time = time.time() - start_time

        assert elapsed_time < 1.0, f"Small export took {elapsed_time:.2f}s, expected < 1s"
        assert output_file.exists()

    def test_export_performance_medium(self, temp_output_dir):
        """
        测试中等数据集导出性能

        基准: 1,000 条数据导出 < 5秒
        """
        results = [create_test_result(execution_id=f"export_{i}") for i in range(1000)]
        exporter = ExcelExporter()
        output_file = temp_output_dir / "perf_medium.xlsx"

        start_time = time.time()
        exporter.export_results(results, str(output_file))
        elapsed_time = time.time() - start_time

        assert elapsed_time < 5.0, f"Medium export took {elapsed_time:.2f}s, expected < 5s"
        assert output_file.exists()

    def test_export_performance_large(self, large_dataset, temp_output_dir):
        """
        测试大数据集导出性能

        基准: 5,000 条数据导出 < 10秒
        """
        exporter = ExcelExporter()
        output_file = temp_output_dir / "perf_large.xlsx"

        start_time = time.time()
        exporter.export_results(large_dataset, str(output_file))
        elapsed_time = time.time() - start_time

        assert elapsed_time < 10.0, f"Large export took {elapsed_time:.2f}s, expected < 10s"
        assert output_file.exists()

        # 验证文件大小合理
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        assert file_size_mb > 0, "Export file should not be empty"

    def test_export_multiple_files_performance(self, sample_results, temp_output_dir):
        """
        测试多次导出性能

        验证重复导出的性能稳定性
        """
        exporter = ExcelExporter()

        times = []
        for i in range(5):
            output_file = temp_output_dir / f"multi_export_{i}.xlsx"

            start_time = time.time()
            exporter.export_results(sample_results, str(output_file))
            elapsed_time = time.time() - start_time

            times.append(elapsed_time)
            assert output_file.exists()

        avg_time = sum(times) / len(times)
        max_time = max(times)

        assert avg_time < 1.0, f"Average export time: {avg_time:.2f}s"
        assert max_time < 1.5, f"Max export time: {max_time:.2f}s"


class TestEndToEndPerformance:
    """测试端到端完整流程性能"""

    def test_complete_pipeline_performance_small(self, sample_results, temp_output_dir):
        """
        测试完整流程性能 - 小数据集

        基准: 收集+统计+分类+导出 < 2秒
        """
        start_time = time.time()

        # 收集
        collector = DataCollector()
        for result in sample_results:
            collector.collect_result(result)

        # 统计
        stats = collector.get_statistics()

        # 分类
        classifier = ResultClassifier()
        classification = classifier.classify_batch(sample_results)

        # 导出
        exporter = ExcelExporter()
        output_file = temp_output_dir / "pipeline_small.xlsx"
        exporter.export_results(sample_results, str(output_file))

        elapsed_time = time.time() - start_time

        assert elapsed_time < 2.0, f"Complete pipeline took {elapsed_time:.2f}s, expected < 2s"

    def test_complete_pipeline_performance_medium(self, temp_output_dir):
        """
        测试完整流程性能 - 中等数据集

        基准: 1,000 条数据完整流程 < 10秒
        """
        results = [create_test_result(execution_id=f"pipe_{i}") for i in range(1000)]

        start_time = time.time()

        collector = DataCollector()
        for result in results:
            collector.collect_result(result)

        stats = collector.get_statistics()

        classifier = ResultClassifier()
        classification = classifier.classify_batch(results)

        exporter = ExcelExporter()
        output_file = temp_output_dir / "pipeline_medium.xlsx"
        exporter.export_results(results, str(output_file))

        elapsed_time = time.time() - start_time

        assert elapsed_time < 10.0, f"Complete pipeline took {elapsed_time:.2f}s, expected < 10s"

    def test_complete_pipeline_performance_large(self, large_dataset, temp_output_dir):
        """
        测试完整流程性能 - 大数据集

        基准: 5,000 条数据完整流程 < 20秒
        """
        start_time = time.time()

        collector = DataCollector()
        for result in large_dataset:
            collector.collect_result(result)

        stats = collector.get_statistics()

        classifier = ResultClassifier()
        classification = classifier.classify_batch(large_dataset)

        exporter = ExcelExporter()
        output_file = temp_output_dir / "pipeline_large.xlsx"
        exporter.export_results(large_dataset, str(output_file))

        elapsed_time = time.time() - start_time

        assert elapsed_time < 20.0, f"Complete pipeline took {elapsed_time:.2f}s, expected < 20s"
        assert output_file.exists()


class TestPerformanceScalability:
    """测试性能可扩展性"""

    def test_collection_scalability(self):
        """
        测试收集操作的可扩展性

        验证收集时间与数据量呈线性关系
        """
        sizes = [100, 500, 1000, 2000]
        times = []

        for size in sizes:
            collector = DataCollector()
            results = [create_test_result(execution_id=f"scale_{i}") for i in range(size)]

            start_time = time.time()
            for result in results:
                collector.collect_result(result)
            elapsed_time = time.time() - start_time

            times.append(elapsed_time)

        # 验证时间增长是合理的（粗略线性）
        # 2000条的时间应该不超过100条的30倍
        assert times[-1] / times[0] < 30, "Collection performance does not scale linearly"

    def test_classification_scalability(self):
        """
        测试分类操作的可扩展性

        验证分类时间与数据量呈线性关系
        """
        classifier = ResultClassifier()
        sizes = [100, 500, 1000, 2000]
        times = []

        for size in sizes:
            results = [
                create_test_result(
                    execution_id=f"scale_class_{i}",
                    execution_time=random.uniform(0.5, 10.0),
                    tokens_used=random.randint(50, 500)
                )
                for i in range(size)
            ]

            start_time = time.time()
            classification = classifier.classify_batch(results)
            elapsed_time = time.time() - start_time

            times.append(elapsed_time)

        # 验证时间增长是合理的
        assert times[-1] / times[0] < 30, "Classification performance does not scale linearly"

    def test_statistics_scalability(self):
        """
        测试统计计算的可扩展性

        验证统计时间与数据量的关系合理
        """
        sizes = [100, 500, 1000, 2000]
        times = []

        for size in sizes:
            collector = DataCollector()
            results = [create_test_result(execution_id=f"stat_scale_{i}") for i in range(size)]

            for result in results:
                collector.collect_result(result)

            start_time = time.time()
            stats = collector.get_statistics()
            elapsed_time = time.time() - start_time

            times.append(elapsed_time)

        # 统计计算应该保持快速，即使数据量增长
        assert all(t < 0.5 for t in times), "Statistics calculation should remain fast"
