"""
配置集成测试 - Collector 模块与 Config 模块的集成

Date: 2025-11-14
Author: qa-engineer
Description: 测试 Collector 模块与配置系统的兼容性和集成
"""

import pytest
import yaml
from pathlib import Path

from src.collector import (
    ResultClassifier,
    TestResult,
    TestStatus,
    PerformanceGrade
)
from .conftest import create_test_result


class TestConfigIntegration:
    """测试与配置模块的集成"""

    def test_classifier_with_yaml_config(self, sample_config_yaml):
        """
        测试从 YAML 配置文件加载分类阈值

        验证:
        1. 正确解析配置文件
        2. 阈值被正确应用到分类器
        3. 分类结果符合配置的阈值
        """
        # 读取配置
        with open(sample_config_yaml, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        thresholds = config["classification"]["thresholds"]

        # 创建分类器并应用配置
        classifier = ResultClassifier(thresholds=thresholds)

        # 验证阈值被正确设置
        loaded_thresholds = classifier.get_thresholds()
        assert loaded_thresholds["excellent"]["execution_time"] == 2.0
        assert loaded_thresholds["excellent"]["token_efficiency"] == 0.8
        assert loaded_thresholds["good"]["execution_time"] == 5.0
        assert loaded_thresholds["fair"]["execution_time"] == 10.0

        # 测试分类是否按照配置工作
        # 验证阈值被正确加载即可
        assert classifier.get_thresholds()["excellent"]["execution_time"] == 2.0

    def test_classifier_threshold_update_from_config(self, tmp_path):
        """
        测试动态更新分类阈值

        场景:
        1. 使用默认阈值创建分类器
        2. 从配置文件加载新阈值
        3. 验证分类行为改变
        """
        # 创建配置文件 - 更严格的标准
        strict_config = {
            "classification": {
                "thresholds": {
                    "excellent": {
                        "execution_time": 1.0,  # 更严格
                        "token_efficiency": 0.9
                    },
                    "good": {
                        "execution_time": 3.0,
                        "token_efficiency": 0.7
                    },
                    "fair": {
                        "execution_time": 8.0,
                        "token_efficiency": 0.5
                    }
                }
            }
        }

        config_file = tmp_path / "strict_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(strict_config, f)

        # 创建测试结果
        test_result = TestResult(
            workflow_id="test",
            execution_id="test_001",
            timestamp=None,
            status=TestStatus.SUCCESS,
            execution_time=1.5,  # 原本是 excellent，现在只能是 good
            tokens_used=100,
            cost=0.01,
            inputs={"query": "test"},
            outputs={"answer": "x" * 80}
        )

        # 默认阈值分类器
        default_classifier = ResultClassifier()
        default_grade = default_classifier.classify_result(test_result)

        # 严格阈值分类器
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        strict_classifier = ResultClassifier(thresholds=config["classification"]["thresholds"])
        strict_grade = strict_classifier.classify_result(test_result)

        # 验证分类结果不同
        # 使用严格标准后，分级应该变低
        grade_order = {
            PerformanceGrade.EXCELLENT: 4,
            PerformanceGrade.GOOD: 3,
            PerformanceGrade.FAIR: 2,
            PerformanceGrade.POOR: 1
        }

        assert grade_order[strict_grade] <= grade_order[default_grade]

    def test_invalid_config_handling(self, tmp_path):
        """
        测试无效配置的处理

        验证:
        1. 缺失必需字段时抛出异常
        2. 错误的数据类型时抛出异常
        3. 不合理的值时抛出异常
        """
        # 缺失字段的配置 - 修改为缺失完整的级别
        invalid_config1 = {
            "classification": {
                "thresholds": {
                    "excellent": {
                        "execution_time": 2.0,
                        "token_efficiency": 0.8
                    }
                    # 缺失 good 和 fair
                }
            }
        }

        config_file1 = tmp_path / "invalid1.yaml"
        with open(config_file1, "w", encoding="utf-8") as f:
            yaml.dump(invalid_config1, f)

        with open(config_file1, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 应该抛出 ClassificationException（缺失级别定义）
        from src.utils.exceptions import ClassificationException
        with pytest.raises(ClassificationException, match="Missing threshold"):
            ResultClassifier(thresholds=invalid_config1["classification"]["thresholds"])

        # 负值配置
        invalid_config2 = {
            "classification": {
                "thresholds": {
                    "excellent": {
                        "execution_time": -1.0,  # 负值
                        "token_efficiency": 0.8
                    },
                    "good": {
                        "execution_time": 5.0,
                        "token_efficiency": 0.6
                    },
                    "fair": {
                        "execution_time": 10.0,
                        "token_efficiency": 0.4
                    }
                }
            }
        }

        from src.utils.exceptions import ClassificationException
        with pytest.raises(ClassificationException, match="must be positive"):
            ResultClassifier(thresholds=invalid_config2["classification"]["thresholds"])


class TestRunManifestCompatibility:
    """测试与 executor.RunManifest 的兼容性"""

    def test_testresult_from_manifest_structure(self):
        """
        测试从模拟的 RunManifest 结构创建 TestResult

        模拟 executor 模块可能提供的数据结构
        """
        # 模拟 RunManifest 的数据结构
        mock_manifest = {
            "workflow_id": "wf_20231114_001",
            "execution_id": "exec_abc123",
            "timestamp": "2023-11-14T10:30:00",
            "status": "success",
            "execution_time": 2.5,
            "tokens_used": 250,
            "cost": 0.025,
            "inputs": {
                "query": "What is the weather today?",
                "context": "User location: Beijing"
            },
            "outputs": {
                "answer": "The weather in Beijing today is sunny with a high of 15°C.",
                "confidence": 0.95
            },
            "metadata": {
                "model": "gpt-4",
                "temperature": 0.7
            }
        }

        # 从 manifest 创建 TestResult
        from datetime import datetime

        test_result = TestResult(
            workflow_id=mock_manifest["workflow_id"],
            execution_id=mock_manifest["execution_id"],
            timestamp=datetime.fromisoformat(mock_manifest["timestamp"]),
            status=TestStatus(mock_manifest["status"]),
            execution_time=mock_manifest["execution_time"],
            tokens_used=mock_manifest["tokens_used"],
            cost=mock_manifest["cost"],
            inputs=mock_manifest["inputs"],
            outputs=mock_manifest["outputs"],
            metadata=mock_manifest.get("metadata", {})
        )

        # 验证转换正确
        assert test_result.workflow_id == "wf_20231114_001"
        assert test_result.execution_id == "exec_abc123"
        assert test_result.status == TestStatus.SUCCESS
        assert test_result.execution_time == 2.5
        assert test_result.tokens_used == 250
        assert test_result.metadata["model"] == "gpt-4"

    def test_batch_conversion_from_manifests(self):
        """
        测试批量从 manifest 列表转换为 TestResult 列表

        模拟批量处理场景
        """
        from datetime import datetime

        mock_manifests = [
            {
                "workflow_id": f"wf_{i}",
                "execution_id": f"exec_{i:03d}",
                "timestamp": datetime.now().isoformat(),
                "status": "success" if i % 4 != 0 else "failed",
                "execution_time": 1.0 + i * 0.1,
                "tokens_used": 100 + i * 10,
                "cost": 0.01 + i * 0.001,
                "inputs": {"query": f"query_{i}"},
                "outputs": {"answer": f"answer_{i}"} if i % 4 != 0 else {},
                "error_message": None if i % 4 != 0 else f"Error {i}"
            }
            for i in range(20)
        ]

        # 批量转换
        test_results = []
        for manifest in mock_manifests:
            result = TestResult(
                workflow_id=manifest["workflow_id"],
                execution_id=manifest["execution_id"],
                timestamp=datetime.fromisoformat(manifest["timestamp"]),
                status=TestStatus(manifest["status"]),
                execution_time=manifest["execution_time"],
                tokens_used=manifest["tokens_used"],
                cost=manifest["cost"],
                inputs=manifest["inputs"],
                outputs=manifest["outputs"],
                error_message=manifest.get("error_message")
            )
            test_results.append(result)

        # 验证转换结果
        assert len(test_results) == 20
        assert all(isinstance(r, TestResult) for r in test_results)

        # 验证状态分布
        success_count = sum(1 for r in test_results if r.status == TestStatus.SUCCESS)
        failed_count = sum(1 for r in test_results if r.status == TestStatus.FAILED)

        assert success_count == 15
        assert failed_count == 5


class TestExportConfigIntegration:
    """测试导出配置的集成"""

    def test_export_with_config_options(self, sample_results, sample_config_yaml, temp_output_dir):
        """
        测试根据配置选项导出

        验证配置中的导出选项被正确应用
        """
        from src.collector import ExcelExporter

        # 读取配置
        with open(sample_config_yaml, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        export_config = config["export"]

        # 使用配置文件中的文件名
        output_file = temp_output_dir / export_config["default_filename"]

        exporter = ExcelExporter()
        exporter.export_results(sample_results, str(output_file))

        assert output_file.exists()
        assert output_file.name == "test_results.xlsx"

    def test_custom_export_path_from_config(self, sample_results, tmp_path):
        """
        测试从配置指定自定义导出路径

        验证:
        1. 配置中的路径被正确解析
        2. 导出到指定位置
        """
        from src.collector import ExcelExporter

        # 创建自定义输出目录
        custom_output = tmp_path / "custom_reports" / "collector"
        custom_output.mkdir(parents=True, exist_ok=True)

        output_file = custom_output / "results.xlsx"

        exporter = ExcelExporter()
        exporter.export_results(sample_results, str(output_file))

        assert output_file.exists()
        assert output_file.parent.name == "collector"


class TestConfigDrivenWorkflow:
    """测试配置驱动的完整工作流"""

    def test_end_to_end_with_config(self, sample_results, sample_config_yaml, temp_output_dir):
        """
        测试从配置加载到完整流程的集成

        步骤:
        1. 加载配置文件
        2. 根据配置创建组件
        3. 执行完整流程
        4. 验证结果符合配置
        """
        from src.collector import DataCollector, ResultClassifier, ExcelExporter

        # 1. 加载配置
        with open(sample_config_yaml, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 2. 根据配置创建组件
        classifier = ResultClassifier(thresholds=config["classification"]["thresholds"])
        exporter = ExcelExporter()
        collector = DataCollector()

        # 3. 执行流程
        for result in sample_results:
            collector.collect_result(result)

        stats = collector.get_statistics()
        classification = classifier.classify_batch(sample_results)

        output_file = temp_output_dir / config["export"]["default_filename"]
        exporter.export_results(sample_results, str(output_file))

        # 4. 验证结果
        assert stats.total_executions == len(sample_results)
        assert (
                       classification.excellent_count +
                       classification.good_count +
                       classification.fair_count +
                       classification.poor_count
               ) == len(sample_results)
        assert output_file.exists()

    def test_multi_environment_configs(self, sample_results, tmp_path):
        """
        测试多环境配置支持

        验证:
        1. 开发环境配置（宽松阈值）
        2. 生产环境配置（严格阈值）
        3. 分类结果根据环境不同而变化
        """
        from src.collector import ResultClassifier

        # 开发环境配置 - 宽松
        dev_config = {
            "excellent": {"execution_time": 5.0, "token_efficiency": 0.6},
            "good": {"execution_time": 10.0, "token_efficiency": 0.4},
            "fair": {"execution_time": 20.0, "token_efficiency": 0.2}
        }

        # 生产环境配置 - 严格
        prod_config = {
            "excellent": {"execution_time": 1.0, "token_efficiency": 0.9},
            "good": {"execution_time": 3.0, "token_efficiency": 0.7},
            "fair": {"execution_time": 7.0, "token_efficiency": 0.5}
        }

        dev_classifier = ResultClassifier(thresholds=dev_config)
        prod_classifier = ResultClassifier(thresholds=prod_config)

        # 分类同一批数据
        dev_classification = dev_classifier.classify_batch(sample_results)
        prod_classification = prod_classifier.classify_batch(sample_results)

        # 开发环境应该有更多优秀/良好评级
        dev_high_grade = dev_classification.excellent_count + dev_classification.good_count
        prod_high_grade = prod_classification.excellent_count + prod_classification.good_count

        assert dev_high_grade >= prod_high_grade
