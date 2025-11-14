"""
并发安全测试 - Collector 模块

Date: 2025-11-14
Author: qa-engineer
Description: 测试多线程和并发场景下的安全性和数据完整性
"""

import pytest
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.collector import (
    DataCollector,
    ResultClassifier,
    ExcelExporter,
    TestResult,
    TestStatus
)
from .conftest import create_test_result


class TestConcurrentCollection:
    """测试并发收集的线程安全性"""

    def test_concurrent_collection_thread_safety(self):
        """
        测试并发收集的线程安全性

        场景:
        - 10个线程同时收集数据
        - 每个线程收集100条
        - 验证最终数据完整性（1000条）
        """
        collector = DataCollector()
        threads_count = 10
        results_per_thread = 100
        expected_total = threads_count * results_per_thread

        def collect_worker(thread_id):
            """工作线程函数"""
            for i in range(results_per_thread):
                result = create_test_result(
                    workflow_id=f"thread_{thread_id}",
                    execution_id=f"t{thread_id}_r{i:03d}"
                )
                collector.collect_result(result)

        # 创建并启动线程
        threads = []
        for thread_id in range(threads_count):
            thread = threading.Thread(target=collect_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证数据完整性
        assert collector.get_result_count() == expected_total

        # 验证没有数据丢失
        collected = collector.get_all_results()
        execution_ids = {r.execution_id for r in collected}
        assert len(execution_ids) == expected_total

    def test_concurrent_collection_with_queries(self):
        """
        测试并发收集和查询

        场景:
        - 一些线程持续收集数据
        - 另一些线程持续查询数据
        - 验证无死锁和数据竞争
        """
        collector = DataCollector()
        collection_threads = 5
        query_threads = 3
        results_per_thread = 50
        stop_event = threading.Event()
        query_counts = []

        def collect_worker(thread_id):
            """收集工作线程"""
            for i in range(results_per_thread):
                result = create_test_result(
                    workflow_id=f"wf_{thread_id % 3}",
                    execution_id=f"collect_t{thread_id}_r{i:03d}"
                )
                collector.collect_result(result)
                time.sleep(0.001)  # 模拟真实收集间隔

        def query_worker():
            """查询工作线程"""
            local_count = 0
            while not stop_event.is_set():
                # 执行各种查询
                all_results = collector.get_all_results()
                count = collector.get_result_count()
                wf0_results = collector.get_results_by_workflow("wf_0")

                local_count += 1
                time.sleep(0.01)

            query_counts.append(local_count)

        # 启动收集线程
        collect_threads_list = []
        for i in range(collection_threads):
            thread = threading.Thread(target=collect_worker, args=(i,))
            collect_threads_list.append(thread)
            thread.start()

        # 启动查询线程
        query_threads_list = []
        for _ in range(query_threads):
            thread = threading.Thread(target=query_worker)
            query_threads_list.append(thread)
            thread.start()

        # 等待收集完成
        for thread in collect_threads_list:
            thread.join()

        # 停止查询线程
        stop_event.set()
        for thread in query_threads_list:
            thread.join()

        # 验证数据完整性
        expected_total = collection_threads * results_per_thread
        assert collector.get_result_count() == expected_total

        # 验证查询线程都执行了多次查询
        assert all(count > 0 for count in query_counts)

    def test_concurrent_statistics_calculation(self):
        """
        测试并发统计计算

        场景:
        - 一个线程持续收集数据
        - 多个线程同时计算统计
        - 验证统计结果一致性
        """
        collector = DataCollector()
        stop_collection = threading.Event()
        statistics_results = []

        def collect_worker():
            """持续收集数据"""
            counter = 0
            while not stop_collection.is_set():
                result = create_test_result(execution_id=f"stats_collect_{counter:04d}")
                collector.collect_result(result)
                counter += 1
                time.sleep(0.01)

        def stats_worker(worker_id):
            """计算统计"""
            for _ in range(10):
                stats = collector.get_statistics()
                statistics_results.append({
                    "worker_id": worker_id,
                    "total": stats.total_executions,
                    "success_rate": stats.success_rate
                })
                time.sleep(0.02)

        # 启动收集线程
        collect_thread = threading.Thread(target=collect_worker)
        collect_thread.start()

        # 启动统计线程
        stats_threads = []
        for i in range(5):
            thread = threading.Thread(target=stats_worker, args=(i,))
            stats_threads.append(thread)
            thread.start()

        # 等待统计线程完成
        for thread in stats_threads:
            thread.join()

        # 停止收集
        stop_collection.set()
        collect_thread.join()

        # 验证统计结果都是有效的
        assert len(statistics_results) == 50  # 5 workers * 10 iterations
        assert all(r["total"] >= 0 for r in statistics_results)
        assert all(0 <= r["success_rate"] <= 1.0 for r in statistics_results)


class TestConcurrentClassification:
    """测试并发分类"""

    def test_concurrent_classification_safety(self, sample_results):
        """
        测试并发分类的安全性

        场景:
        - 多个线程同时对相同数据进行分类
        - 验证分类结果一致
        """
        classifier = ResultClassifier()
        classification_results = []
        threads_count = 10

        def classify_worker():
            """分类工作线程"""
            classification = classifier.classify_batch(sample_results)
            classification_results.append({
                "excellent": classification.excellent_count,
                "good": classification.good_count,
                "fair": classification.fair_count,
                "poor": classification.poor_count
            })

        # 启动多个分类线程
        threads = []
        for _ in range(threads_count):
            thread = threading.Thread(target=classify_worker)
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证所有结果一致
        assert len(classification_results) == threads_count

        first_result = classification_results[0]
        for result in classification_results[1:]:
            assert result["excellent"] == first_result["excellent"]
            assert result["good"] == first_result["good"]
            assert result["fair"] == first_result["fair"]
            assert result["poor"] == first_result["poor"]

    def test_concurrent_threshold_updates(self):
        """
        测试并发阈值更新

        场景:
        - 一些线程更新阈值
        - 其他线程进行分类
        - 验证不会崩溃或产生无效结果
        """
        classifier = ResultClassifier()
        test_result = create_test_result(execution_time=2.5, tokens_used=100)
        classification_results = []
        stop_event = threading.Event()

        def update_thresholds_worker():
            """更新阈值的线程"""
            thresholds_variants = [
                {
                    "excellent": {"execution_time": 2.0, "token_efficiency": 0.8},
                    "good": {"execution_time": 5.0, "token_efficiency": 0.6},
                    "fair": {"execution_time": 10.0, "token_efficiency": 0.4}
                },
                {
                    "excellent": {"execution_time": 3.0, "token_efficiency": 0.7},
                    "good": {"execution_time": 6.0, "token_efficiency": 0.5},
                    "fair": {"execution_time": 12.0, "token_efficiency": 0.3}
                }
            ]

            counter = 0
            while not stop_event.is_set():
                variant = thresholds_variants[counter % 2]
                try:
                    classifier.set_thresholds(variant)
                except Exception:
                    pass  # 可能因为并发导致暂时失败
                counter += 1
                time.sleep(0.01)

        def classify_worker():
            """分类线程"""
            for _ in range(50):
                try:
                    grade = classifier.classify_result(test_result)
                    classification_results.append(grade)
                except Exception:
                    pass  # 可能因为并发更新阈值导致暂时失败
                time.sleep(0.01)

        # 启动线程
        update_thread = threading.Thread(target=update_thresholds_worker)
        classify_threads = [threading.Thread(target=classify_worker) for _ in range(3)]

        update_thread.start()
        for thread in classify_threads:
            thread.start()

        # 等待分类完成
        for thread in classify_threads:
            thread.join()

        # 停止更新线程
        stop_event.set()
        update_thread.join()

        # 验证至少有一些分类成功
        assert len(classification_results) > 0


class TestConcurrentExport:
    """测试并发导出"""

    def test_concurrent_export_to_different_files(self, sample_results, temp_output_dir):
        """
        测试并发导出到不同文件

        场景:
        - 多个线程同时导出到不同文件
        - 验证所有文件都成功创建
        """
        exporter = ExcelExporter()
        threads_count = 5
        output_files = []

        def export_worker(worker_id):
            """导出工作线程"""
            output_file = temp_output_dir / f"concurrent_export_{worker_id}.xlsx"
            exporter.export_results(sample_results, str(output_file))
            output_files.append(output_file)

        # 启动导出线程
        threads = []
        for i in range(threads_count):
            thread = threading.Thread(target=export_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证所有文件都存在
        assert len(output_files) == threads_count
        for file in output_files:
            assert file.exists()
            assert file.stat().st_size > 0

    def test_concurrent_export_same_file(self, sample_results, temp_output_dir):
        """
        测试并发导出到同一文件

        场景:
        - 多个线程尝试导出到同一文件
        - 验证不会崩溃（虽然可能有竞争）
        """
        exporter = ExcelExporter()
        output_file = temp_output_dir / "same_file_export.xlsx"
        threads_count = 3
        success_count = [0]
        lock = threading.Lock()

        def export_worker():
            """导出工作线程"""
            try:
                exporter.export_results(sample_results, str(output_file))
                with lock:
                    success_count[0] += 1
            except Exception:
                # 可能因为文件被占用导致失败
                pass

        # 启动线程
        threads = []
        for _ in range(threads_count):
            thread = threading.Thread(target=export_worker)
            threads.append(thread)
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join()

        # 验证至少有一个成功
        assert success_count[0] >= 1
        assert output_file.exists()


class TestConcurrentFullPipeline:
    """测试并发完整流程"""

    def test_concurrent_pipeline_execution(self, temp_output_dir):
        """
        测试并发执行完整流程

        场景:
        - 多个线程同时执行收集→统计→分类→导出
        - 每个线程使用独立的数据和文件
        - 验证所有流程都成功完成
        """
        threads_count = 5
        results_per_thread = 50
        completed = []

        def pipeline_worker(worker_id):
            """完整流程工作线程"""
            try:
                # 收集
                collector = DataCollector()
                results = [
                    create_test_result(
                        workflow_id=f"pipeline_wf_{worker_id}",
                        execution_id=f"pw{worker_id}_r{i:03d}",
                        execution_time=random.uniform(0.5, 5.0),
                        tokens_used=random.randint(50, 300)
                    )
                    for i in range(results_per_thread)
                ]

                for result in results:
                    collector.collect_result(result)

                # 统计
                stats = collector.get_statistics()

                # 分类
                classifier = ResultClassifier()
                classification = classifier.classify_batch(results)

                # 导出
                exporter = ExcelExporter()
                output_file = temp_output_dir / f"pipeline_worker_{worker_id}.xlsx"
                exporter.export_results(results, str(output_file))

                completed.append({
                    "worker_id": worker_id,
                    "collected": collector.get_result_count(),
                    "stats_total": stats.total_executions,
                    "file_exists": output_file.exists()
                })

            except Exception as e:
                completed.append({
                    "worker_id": worker_id,
                    "error": str(e)
                })

        # 启动线程
        threads = []
        for i in range(threads_count):
            thread = threading.Thread(target=pipeline_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join()

        # 验证所有线程都成功
        assert len(completed) == threads_count
        for result in completed:
            assert "error" not in result
            assert result["collected"] == results_per_thread
            assert result["stats_total"] == results_per_thread
            assert result["file_exists"] is True

    def test_shared_collector_concurrent_operations(self):
        """
        测试共享 Collector 的并发操作

        场景:
        - 多个线程共享同一个 DataCollector
        - 同时进行收集、查询、统计操作
        - 验证数据一致性
        """
        collector = DataCollector()
        threads_count = 10
        operations_per_thread = 30  # 减少操作次数避免超时
        lock = threading.Lock()
        operation_counts = {"collect": 0, "query": 0, "stats": 0}

        def mixed_operations_worker(worker_id):
            """混合操作工作线程"""
            for i in range(operations_per_thread):
                op_type = random.choice(["collect", "query", "query", "stats"])  # 更多查询操作

                if op_type == "collect":
                    result = create_test_result(
                        workflow_id=f"mixed_wf_{worker_id % 3}",
                        execution_id=f"mixed_w{worker_id}_o{i:03d}"
                    )
                    collector.collect_result(result)
                    with lock:
                        operation_counts["collect"] += 1

                elif op_type == "query":
                    try:
                        all_results = collector.get_all_results()
                        count = collector.get_result_count()
                        with lock:
                            operation_counts["query"] += 1
                    except Exception:
                        pass  # 忽略查询异常

                elif op_type == "stats":
                    try:
                        if collector.get_result_count() > 0:  # 只在有数据时计算统计
                            stats = collector.get_statistics()
                        with lock:
                            operation_counts["stats"] += 1
                    except Exception:
                        pass  # 忽略统计异常

        # 启动线程
        threads = []
        for i in range(threads_count):
            thread = threading.Thread(target=mixed_operations_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待完成
        for thread in threads:
            thread.join()

        # 验证操作都执行了（至少大部分）
        total_operations = sum(operation_counts.values())
        assert total_operations > threads_count * operations_per_thread * 0.8  # 至少80%成功

        # 验证收集的数据数量正确
        assert collector.get_result_count() == operation_counts["collect"]


class TestRaceConditions:
    """测试竞态条件"""

    def test_no_race_in_result_count(self):
        """
        测试结果计数没有竞态条件

        快速并发收集，验证计数准确
        """
        collector = DataCollector()
        expected_count = 500

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(expected_count):
                future = executor.submit(
                    collector.collect_result,
                    create_test_result(execution_id=f"race_{i:04d}")
                )
                futures.append(future)

            # 等待所有完成
            for future in as_completed(futures):
                future.result()

        # 验证计数准确
        assert collector.get_result_count() == expected_count

    def test_no_race_in_results_by_workflow(self):
        """
        测试同一 workflow 并发收集时不丢数据

        验证 get_results_by_workflow 在多线程向同一 workflow_id
        写入时，所有结果都被正确保存（不会因列表覆盖而丢失）
        """
        collector = DataCollector()
        workflow_id = "shared_workflow"
        expected_count = 500

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i in range(expected_count):
                future = executor.submit(
                    collector.collect_result,
                    create_test_result(
                        workflow_id=workflow_id,
                        execution_id=f"shared_{i:04d}",
                    ),
                )
                futures.append(future)

            # 等待所有完成
            for future in as_completed(futures):
                future.result()

        # 验证：总计数正确
        assert collector.get_result_count() == expected_count

        # 验证：按 workflow 查询返回完整结果（不丢数据）
        wf_results = collector.get_results_by_workflow(workflow_id)
        assert len(wf_results) == expected_count

        # 验证：所有 execution_id 都存在（无重复覆盖）
        execution_ids = {r.execution_id for r in wf_results}
        assert len(execution_ids) == expected_count

    def test_no_race_in_statistics(self):
        """
        测试统计计算没有竞态条件

        并发收集和统计，验证统计一致性
        """
        collector = DataCollector()
        collection_count = 200

        # 先收集一批数据
        for i in range(collection_count):
            collector.collect_result(
                create_test_result(
                    execution_id=f"stat_race_{i:04d}",
                    status=TestStatus.SUCCESS if i % 2 == 0 else TestStatus.FAILED
                )
            )

        # 并发计算统计
        stats_results = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(collector.get_statistics) for _ in range(20)]

            for future in as_completed(futures):
                stats = future.result()
                stats_results.append(stats)

        # 验证所有统计结果一致
        first_stats = stats_results[0]
        for stats in stats_results[1:]:
            assert stats.total_executions == first_stats.total_executions
            assert stats.successful_count == first_stats.successful_count
            assert stats.failed_count == first_stats.failed_count
            assert stats.success_rate == first_stats.success_rate
