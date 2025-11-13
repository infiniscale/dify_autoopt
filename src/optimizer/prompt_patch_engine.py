"""
Optimizer Module - Prompt Patch Engine

Date: 2025-11-13
Author: Rebirthli
Description: Engine for applying prompt patches to workflow DSL files
"""

import logging
from typing import Any, Dict, List, Optional
from jinja2 import Template, TemplateError

from ..config.models import PromptPatch, PromptSelector, PromptStrategy, WorkflowCatalog, NodeMeta
from ..config.utils.exceptions import PatchTargetMissing, TemplateRenderError, DSLParseError
from ..config.utils.yaml_parser import YamlParser

logger = logging.getLogger(__name__)


class PromptPatchEngine:
    """Prompt Patch DSL engine for modifying workflow prompts"""

    def __init__(
        self,
        catalog: WorkflowCatalog,
        yaml_parser: YamlParser,
    ):
        """
        Initialize PromptPatchEngine

        Args:
            catalog: WorkflowCatalog with node metadata
            yaml_parser: YAML parser for DSL manipulation
        """
        self.catalog = catalog
        self.yaml_parser = yaml_parser
        self._node_index = self._build_node_index()

    def _build_node_index(self) -> Dict[str, Dict[str, NodeMeta]]:
        """Build workflow_id -> node_id -> NodeMeta index"""
        index = {}
        for wf in self.catalog.workflows:
            index[wf.id] = {node.node_id: node for node in wf.nodes}
        return index

    def apply_patches(
        self,
        workflow_id: str,
        dsl_text: str,
        patches: List[PromptPatch],
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Apply prompt patches to DSL

        Args:
            workflow_id: Workflow identifier
            dsl_text: Original DSL YAML text
            patches: List of patches to apply
            context: Context variables for template rendering

        Returns:
            Modified DSL YAML text

        Raises:
            PatchTargetMissing: If patch target not found and if_missing='error'
            TemplateRenderError: If template rendering fails
            DSLParseError: If DSL parsing fails
        """
        context = context or {}
        dsl_tree = self.yaml_parser.load(dsl_text)

        for patch in patches:
            node_paths = self._resolve_selector(workflow_id, patch.selector)

            # Handle missing targets
            if not node_paths:
                if_missing = patch.selector.constraints.get('if_missing', 'skip')
                if if_missing == 'error':
                    raise PatchTargetMissing(
                        f"No nodes matched selector: {patch.selector.dict()}"
                    )
                else:
                    logger.warning(f"No nodes matched selector, skipping: {patch.selector.dict()}")
                    continue

            # Apply patch to each matching node
            for node_path in node_paths:
                node = self.yaml_parser.get_node_by_path(dsl_tree, node_path)
                if node is None:
                    logger.warning(f"Node not found at path: {node_path}")
                    continue

                prompt_fields = self._get_prompt_fields(workflow_id, node_path)

                for field_path in prompt_fields:
                    original_prompt = self.yaml_parser.get_field_value(node, field_path) or ""
                    new_prompt = self._apply_strategy(
                        original_prompt,
                        patch.strategy,
                        context
                    )
                    self.yaml_parser.set_field_value(node, field_path, new_prompt)

        return self.yaml_parser.dump(dsl_tree)

    def _resolve_selector(
        self,
        workflow_id: str,
        selector: PromptSelector
    ) -> List[str]:
        """
        Resolve selector to list of node paths

        Priority: by_path > by_id > (by_label & by_type)

        Args:
            workflow_id: Workflow identifier
            selector: PromptSelector configuration

        Returns:
            List of JSON pointer paths
        """
        # Direct path takes highest priority
        if selector.by_path:
            return [selector.by_path]

        nodes = self._node_index.get(workflow_id, {})
        matched_paths = []

        for node_id, node_meta in nodes.items():
            # Match by ID
            if selector.by_id and node_meta.node_id != selector.by_id:
                continue

            # Match by type
            if selector.by_type and node_meta.type != selector.by_type:
                continue

            # Match by label (fuzzy)
            if selector.by_label:
                if not node_meta.label or selector.by_label.lower() not in node_meta.label.lower():
                    continue

            matched_paths.append(node_meta.path)

        return matched_paths

    def _get_prompt_fields(self, workflow_id: str, node_path: str) -> List[str]:
        """Get prompt field paths for a node"""
        for node in self._node_index.get(workflow_id, {}).values():
            if node.path == node_path:
                return node.prompt_fields
        return []

    def _apply_strategy(
        self,
        original: str,
        strategy: PromptStrategy,
        context: Dict[str, Any]
    ) -> str:
        """
        Apply prompt modification strategy

        Args:
            original: Original prompt text
            strategy: PromptStrategy configuration
            context: Context for template rendering

        Returns:
            Modified prompt text

        Raises:
            TemplateRenderError: If template rendering fails
        """
        try:
            if strategy.mode == 'replace':
                return strategy.content or original

            elif strategy.mode == 'prepend':
                if strategy.content:
                    return f"{strategy.content}\n\n{original}"
                return original

            elif strategy.mode == 'append':
                if strategy.content:
                    return f"{original}\n\n{strategy.content}"
                return original

            elif strategy.mode == 'template':
                if not strategy.template:
                    raise TemplateRenderError("Template mode requires template configuration")

                # Get template content
                if strategy.template.file:
                    template_text = self._load_template_file(strategy.template.file)
                elif strategy.template.inline:
                    template_text = strategy.template.inline
                else:
                    raise TemplateRenderError("Template must have 'file' or 'inline'")

                # Render template
                template = Template(template_text)
                rendered = template.render(
                    original=original,
                    **context,
                    **strategy.template.variables
                )
                return rendered

            else:
                raise TemplateRenderError(f"Unknown strategy mode: {strategy.mode}")

        except TemplateError as e:
            logger.warning(f"Template rendering failed: {e}")
            if strategy.fallback_value:
                logger.info(f"Using fallback value")
                return strategy.fallback_value
            raise TemplateRenderError(f"Template rendering failed: {e}")

        except Exception as e:
            logger.error(f"Strategy application failed: {e}")
            if strategy.fallback_value:
                return strategy.fallback_value
            raise TemplateRenderError(f"Failed to apply strategy: {e}")

    def _load_template_file(self, file_path: str) -> str:
        """Load template from file"""
        try:
            from pathlib import Path
            path = Path(file_path)
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise TemplateRenderError(f"Failed to load template file {file_path}: {e}")
