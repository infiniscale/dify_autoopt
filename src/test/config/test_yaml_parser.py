"""
单元测试 - YamlParser

测试 YAML 解析和操作功能。
"""

import pytest
from src.config.utils.yaml_parser import YamlParser
from src.config.utils.exceptions import DSLParseError


class TestYamlParserLoad:
    """测试 load 方法"""

    def test_load_valid_yaml(self):
        """测试加载有效的 YAML"""
        parser = YamlParser()
        yaml_text = """
key: value
number: 42
nested:
  inner: data
"""
        result = parser.load(yaml_text)

        assert result == {
            "key": "value",
            "number": 42,
            "nested": {"inner": "data"}
        }

    def test_load_empty_yaml(self):
        """测试加载空 YAML 抛出异常"""
        parser = YamlParser()

        with pytest.raises(DSLParseError, match="Expected dict, got NoneType"):
            parser.load("")

    def test_load_empty_dict_yaml(self):
        """测试加载空字典 YAML"""
        parser = YamlParser()
        result = parser.load("{}")

        assert result == {}

    def test_load_invalid_yaml_syntax(self):
        """测试加载无效的 YAML 语法"""
        parser = YamlParser()
        invalid_yaml = """
key: value
  bad: [unclosed
"""
        with pytest.raises(DSLParseError, match="Invalid YAML"):
            parser.load(invalid_yaml)

    def test_load_non_dict_root(self):
        """测试加载非字典根节点的 YAML"""
        parser = YamlParser()
        list_yaml = "- item1\n- item2\n"

        with pytest.raises(DSLParseError, match="Expected dict, got list"):
            parser.load(list_yaml)

    def test_load_complex_structure(self):
        """测试加载复杂结构"""
        parser = YamlParser()
        yaml_text = """
graph:
  nodes:
    - id: node1
      type: llm
      config:
        prompt: "Hello"
    - id: node2
      type: end
      config: {}
edges:
  - source: node1
    target: node2
"""
        result = parser.load(yaml_text)

        assert len(result["graph"]["nodes"]) == 2
        assert result["graph"]["nodes"][0]["id"] == "node1"
        assert result["graph"]["nodes"][0]["config"]["prompt"] == "Hello"


class TestYamlParserDump:
    """测试 dump 方法"""

    def test_dump_simple_dict(self):
        """测试导出简单字典"""
        parser = YamlParser()
        data = {"key": "value", "number": 42}

        yaml_text = parser.dump(data)

        assert "key: value" in yaml_text
        assert "number: 42" in yaml_text

    def test_dump_nested_dict(self):
        """测试导出嵌套字典"""
        parser = YamlParser()
        data = {
            "level1": {
                "level2": {
                    "key": "value"
                }
            }
        }

        yaml_text = parser.dump(data)

        assert "level1:" in yaml_text
        assert "level2:" in yaml_text
        assert "key: value" in yaml_text

    def test_dump_with_list(self):
        """测试导出包含列表的字典"""
        parser = YamlParser()
        data = {
            "items": [
                {"name": "item1"},
                {"name": "item2"}
            ]
        }

        yaml_text = parser.dump(data)

        assert "items:" in yaml_text
        assert "name: item1" in yaml_text
        assert "name: item2" in yaml_text

    def test_dump_unicode(self):
        """测试导出包含 Unicode 字符的字典"""
        parser = YamlParser()
        data = {"message": "你好世界"}

        yaml_text = parser.dump(data)

        assert "你好世界" in yaml_text


class TestYamlParserGetNodeByPath:
    """测试 get_node_by_path 方法"""

    def test_get_node_root_path(self):
        """测试获取根路径节点"""
        parser = YamlParser()
        tree = {"key": "value"}

        result = parser.get_node_by_path(tree, "/key")

        assert result == "value"

    def test_get_node_nested_path(self):
        """测试获取嵌套路径节点"""
        parser = YamlParser()
        tree = {
            "graph": {
                "nodes": [
                    {"id": "node1", "type": "llm"},
                    {"id": "node2", "type": "end"}
                ]
            }
        }

        result = parser.get_node_by_path(tree, "/graph/nodes/0")

        assert result == {"id": "node1", "type": "llm"}

    def test_get_node_array_index(self):
        """测试通过数组索引获取节点"""
        parser = YamlParser()
        tree = {
            "items": [
                {"name": "first"},
                {"name": "second"},
                {"name": "third"}
            ]
        }

        result = parser.get_node_by_path(tree, "/items/1")

        assert result == {"name": "second"}

    def test_get_node_deep_nested(self):
        """测试获取深层嵌套节点"""
        parser = YamlParser()
        tree = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42
                    }
                }
            }
        }

        result = parser.get_node_by_path(tree, "/level1/level2/level3/value")

        assert result == 42

    def test_get_node_not_found(self):
        """测试获取不存在的节点返回 None"""
        parser = YamlParser()
        tree = {"key": "value"}

        result = parser.get_node_by_path(tree, "/nonexistent")

        assert result is None

    def test_get_node_invalid_array_index(self):
        """测试无效的数组索引返回 None"""
        parser = YamlParser()
        tree = {"items": [1, 2, 3]}

        result = parser.get_node_by_path(tree, "/items/999")

        assert result is None

    def test_get_node_invalid_path_format(self):
        """测试无效的路径格式抛出异常"""
        parser = YamlParser()
        tree = {"key": "value"}

        with pytest.raises(ValueError, match="Path must start with '/'"):
            parser.get_node_by_path(tree, "invalid/path")

    def test_get_node_empty_path_segments(self):
        """测试路径包含空段"""
        parser = YamlParser()
        tree = {"graph": {"nodes": []}}

        # 路径中的连续 '/' 会产生空段，应该被过滤掉
        result = parser.get_node_by_path(tree, "/graph//nodes")

        assert result == []


class TestYamlParserSetFieldValue:
    """测试 set_field_value 方法"""

    def test_set_simple_field(self):
        """测试设置简单字段"""
        parser = YamlParser()
        node = {"key": "old_value"}

        parser.set_field_value(node, "key", "new_value")

        assert node["key"] == "new_value"

    def test_set_nested_field(self):
        """测试设置嵌套字段"""
        parser = YamlParser()
        node = {"config": {"prompt": "old_prompt"}}

        parser.set_field_value(node, "config.prompt", "new_prompt")

        assert node["config"]["prompt"] == "new_prompt"

    def test_set_deep_nested_field(self):
        """测试设置深层嵌套字段"""
        parser = YamlParser()
        node = {"level1": {"level2": {"level3": {"value": 0}}}}

        parser.set_field_value(node, "level1.level2.level3.value", 42)

        assert node["level1"]["level2"]["level3"]["value"] == 42

    def test_set_field_creates_intermediate_dicts(self):
        """测试设置字段时创建中间字典"""
        parser = YamlParser()
        node = {}

        parser.set_field_value(node, "config.prompt", "Hello")

        assert node == {"config": {"prompt": "Hello"}}

    def test_set_field_deep_creates_path(self):
        """测试设置深层字段时创建完整路径"""
        parser = YamlParser()
        node = {}

        parser.set_field_value(node, "a.b.c.d", "value")

        assert node == {
            "a": {
                "b": {
                    "c": {
                        "d": "value"
                    }
                }
            }
        }

    def test_set_field_overwrites_existing(self):
        """测试覆盖已存在的字段"""
        parser = YamlParser()
        node = {"config": {"prompt": "old", "temperature": 0.5}}

        parser.set_field_value(node, "config.prompt", "new")

        assert node["config"]["prompt"] == "new"
        assert node["config"]["temperature"] == 0.5  # 其他字段不受影响

    def test_set_field_with_various_types(self):
        """测试设置不同类型的值"""
        parser = YamlParser()
        node = {}

        parser.set_field_value(node, "string", "text")
        parser.set_field_value(node, "number", 42)
        parser.set_field_value(node, "boolean", True)
        parser.set_field_value(node, "list", [1, 2, 3])
        parser.set_field_value(node, "dict", {"key": "value"})

        assert node == {
            "string": "text",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        }


class TestYamlParserGetFieldValue:
    """测试 get_field_value 方法"""

    def test_get_simple_field(self):
        """测试获取简单字段"""
        parser = YamlParser()
        node = {"key": "value"}

        result = parser.get_field_value(node, "key")

        assert result == "value"

    def test_get_nested_field(self):
        """测试获取嵌套字段"""
        parser = YamlParser()
        node = {"config": {"prompt": "Hello"}}

        result = parser.get_field_value(node, "config.prompt")

        assert result == "Hello"

    def test_get_deep_nested_field(self):
        """测试获取深层嵌套字段"""
        parser = YamlParser()
        node = {
            "level1": {
                "level2": {
                    "level3": {
                        "value": 42
                    }
                }
            }
        }

        result = parser.get_field_value(node, "level1.level2.level3.value")

        assert result == 42

    def test_get_field_not_found(self):
        """测试获取不存在的字段返回 None"""
        parser = YamlParser()
        node = {"key": "value"}

        result = parser.get_field_value(node, "nonexistent")

        assert result is None

    def test_get_field_nested_not_found(self):
        """测试获取不存在的嵌套字段返回 None"""
        parser = YamlParser()
        node = {"config": {"prompt": "Hello"}}

        result = parser.get_field_value(node, "config.nonexistent")

        assert result is None

    def test_get_field_partial_path_exists(self):
        """测试部分路径存在时返回 None"""
        parser = YamlParser()
        node = {"config": {"prompt": "Hello"}}

        result = parser.get_field_value(node, "config.prompt.nested")

        assert result is None

    def test_get_field_various_types(self):
        """测试获取不同类型的字段值"""
        parser = YamlParser()
        node = {
            "string": "text",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"key": "value"}
        }

        assert parser.get_field_value(node, "string") == "text"
        assert parser.get_field_value(node, "number") == 42
        assert parser.get_field_value(node, "boolean") is True
        assert parser.get_field_value(node, "list") == [1, 2, 3]
        assert parser.get_field_value(node, "dict") == {"key": "value"}


class TestYamlParserIntegration:
    """集成测试 - 测试方法组合使用"""

    def test_load_modify_dump_cycle(self):
        """测试加载-修改-导出循环"""
        parser = YamlParser()

        # 加载
        yaml_text = """
graph:
  nodes:
    - id: llm_node
      type: llm
      config:
        prompt: "Original prompt"
"""
        tree = parser.load(yaml_text)

        # 修改
        node = parser.get_node_by_path(tree, "/graph/nodes/0")
        parser.set_field_value(node, "config.prompt", "Modified prompt")

        # 导出
        modified_yaml = parser.dump(tree)

        # 验证
        assert "Modified prompt" in modified_yaml
        assert "Original prompt" not in modified_yaml

    def test_complex_dsl_manipulation(self):
        """测试复杂 DSL 操作"""
        parser = YamlParser()

        yaml_text = """
graph:
  nodes:
    - id: node1
      type: llm
      config:
        prompt: "Step 1"
        temperature: 0.7
    - id: node2
      type: code
      config:
        code: "print('hello')"
edges:
  - source: node1
    target: node2
"""
        tree = parser.load(yaml_text)

        # 修改第一个节点的 prompt
        node1 = parser.get_node_by_path(tree, "/graph/nodes/0")
        assert node1 is not None
        parser.set_field_value(node1, "config.prompt", "Modified step 1")

        # 验证修改
        assert tree["graph"]["nodes"][0]["config"]["prompt"] == "Modified step 1"
        assert tree["graph"]["nodes"][0]["config"]["temperature"] == 0.7

        # 验证其他节点未受影响
        assert tree["graph"]["nodes"][1]["config"]["code"] == "print('hello')"

    def test_get_and_set_field_consistency(self):
        """测试 get_field_value 和 set_field_value 的一致性"""
        parser = YamlParser()
        node = {"config": {"nested": {"value": 42}}}

        # 获取原始值
        original = parser.get_field_value(node, "config.nested.value")
        assert original == 42

        # 设置新值
        parser.set_field_value(node, "config.nested.value", 100)

        # 获取新值
        new_value = parser.get_field_value(node, "config.nested.value")
        assert new_value == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
