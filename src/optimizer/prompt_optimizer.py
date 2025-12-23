"""
Lightweight prompt optimizer pipeline.

Based on execution results and reference expectations, detect problematic
prompts in a workflow DSL and generate patch suggestions. The goal is to
produce reviewable patches (not auto-apply) that can later be reviewed or
fed to a more advanced optimizer (e.g., dspy) when available.
"""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import json as _json
import pickle
import random
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext, contextmanager
from dataclasses import dataclass, field, asdict, fields
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml

from src.optimizer.yaml_loader import load_workflow_yaml
from src.utils.logger import get_logger
from src.workflow.export import _safe_dirname_from_id

logger = get_logger("optimizer.prompt_optimizer")
DEFAULT_DSPY_MAX_TOKENS = 4096
DEFAULT_ACTION_BUDGET = 5
DEFAULT_MAX_CONSTRAINTS = 20
DEFAULT_SCHEMA_TIGHTEN_LIMIT = 2
DEFAULT_MAX_PROMPT_GROWTH = 1.5
DEFAULT_NO_IMPROVE_ROUNDS = 3
_PLACEHOLDER_PATTERNS = [
    re.compile(r"\{\{\s*[^{}]+\s*\}\}"),  # {{ var }} or {{a.b}}
    re.compile(r"\$\{\s*[^${}]+\s*\}"),  # ${var}
]
REF_SOFT_SIMILARITY_THRESHOLD = 0.55


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #


@dataclass
class ExecutionSample:
    """Normalized execution outcome for a single run."""

    index: int
    status: str
    output_text: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReferenceSpec:
    """Reference expectations loaded from YAML/JSON."""

    expected_outputs: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    similarity_threshold: float = 0.72
    max_failure_rate: float = 0.2
    max_latency_seconds: Optional[float] = None


@dataclass
class DetectedIssue:
    """Issue detected from execution/reference comparison."""

    kind: str
    severity: str
    message: str
    evidence_runs: List[int] = field(default_factory=list)


@dataclass
class PromptLocation:
    """Pointer to a prompt text inside the DSL tree."""

    node_id: str
    path: str  # JSON pointer-like path
    text: str


@dataclass
class NodeBrief:
    """Lightweight description of a workflow node for rewriting context."""

    node_id: str
    node_name: str
    node_type: str
    role: str = ""
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    downstream_constraints: List[str] = field(default_factory=list)
    placeholders: List[str] = field(default_factory=list)
    message_layout: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class FailSignal:
    """Structured failure hints instead of one long feedback string."""

    tags: List[str] = field(default_factory=list)
    missing_fields: List[str] = field(default_factory=list)
    format_errors: List[str] = field(default_factory=list)
    constraint_gaps: List[str] = field(default_factory=list)
    semantic_gaps: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PromptBlock:
    """A node-level prompt consisting of multiple messages."""

    node_id: str
    node_name: str
    node_type: str
    messages: List[Dict[str, Any]]  # each: {"role": str, "text": str, "path": str, "index": int}
    merged_text: str
    brief: Optional[NodeBrief] = None


@dataclass
class PromptState:
    """Structured prompt state for slot-level optimization."""

    base_prompt: str = ""
    role: str = ""
    task: str = ""
    schema: str = ""
    checkpoint_rules: Dict[str, str] = field(default_factory=dict)
    strictness_level: int = 3
    negative_examples: List[Dict[str, Any]] = field(default_factory=list)
    global_constraints: List[str] = field(default_factory=list)
    placeholders: List[str] = field(default_factory=list)
    version: int = 1
    structured: bool = False
    schema_tighten_count: int = 0
    schema_frozen: bool = False


class OptimizationStage(Enum):
    FORMAT = auto()
    RECALL = auto()
    STRICT = auto()


@dataclass
class PromptPatch:
    """Patch suggestion for a prompt field."""

    workflow_id: str
    node_id: str
    field_path: str
    old: str
    new: str
    rationale: str
    confidence: float
    evidence_runs: List[int] = field(default_factory=list)


@dataclass
class PromptAction:
    """Actionable prompt update instruction."""

    type: str
    target: Optional[str] = None
    content: Optional[str] = None
    evidence_runs: List[int] = field(default_factory=list)


@dataclass
class OptimizationReport:
    workflow_id: str
    issues: List[DetectedIssue]
    patches: List[PromptPatch]
    yaml_path: Path
    reference_path: Optional[Path]
    actions: List[PromptAction] = field(default_factory=list)
    patched_path: Optional[Path] = None
    stats: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _extract_output_text(run: Dict[str, Any]) -> str:
    """Best-effort extraction of output text from a workflow run payload."""
    if not isinstance(run, dict):
        return str(run)
    for key in ("output", "data", "result", "response"):
        if key in run and run[key] is not None:
            val = run[key]
            if isinstance(val, dict):
                # common dify structure: {"output": {"text": "..."}}
                if "text" in val:
                    return str(val["text"])
            return str(val)
    return str(run)


def _extract_workflow_input(run: Dict[str, Any]) -> str:
    """Best-effort extraction of workflow input/context for DSPy optimizer."""
    if not isinstance(run, dict):
        return str(run)

    def _stringify(value: Any) -> str:
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    candidates = []
    for key in ("input", "inputs", "workflow_input", "payload"):
        if key in run and run[key]:
            candidates.append(_stringify(run[key]))
    data = run.get("data")
    if isinstance(data, dict):
        for key in ("input", "inputs", "request"):
            if key in data and data[key]:
                candidates.append(_stringify(data[key]))
    if candidates:
        return "\n".join(candidates)
    try:
        return json.dumps(run, ensure_ascii=False)
    except Exception:
        return str(run)


def _status_from_run(run: Dict[str, Any]) -> str:
    """Infer a coarse status string from a run payload."""
    if not isinstance(run, dict):
        return "unknown"
    candidates: List[str] = []

    def _collect_status(container: Dict[str, Any]) -> None:
        for key in ("status", "state", "result"):
            if key in container and container[key] is not None:
                candidates.append(str(container[key]))

    _collect_status(run)

    result = run.get("result")
    if isinstance(result, dict):
        _collect_status(result)
        data = result.get("data")
        if isinstance(data, dict):
            _collect_status(data)

    data = run.get("data")
    if isinstance(data, dict):
        _collect_status(data)

    if not candidates and run.get("output") is not None:
        return "success"

    for cand in candidates:
        lowered = cand.lower()
        if lowered in {"success", "succeeded", "ok", "completed", "done"}:
            return "success"
        if lowered in {"failed", "failure", "error", "errored", "exception"}:
            return "failed"
    return candidates[0].lower() if candidates else "unknown"


def _load_reference_spec(path: Optional[str | Path]) -> ReferenceSpec:
    if not path:
        return ReferenceSpec()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Reference file not found: {p}")
    content = p.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(content)
    except Exception:
        data = None
    if not data:
        try:
            data = json.loads(content)
        except Exception:
            data = {}
    if not isinstance(data, dict):
        data = {}
    return ReferenceSpec(
        expected_outputs=list(data.get("expected_outputs") or []),
        constraints=list(data.get("constraints") or []),
        similarity_threshold=float(data.get("similarity_threshold", 0.72)),
        max_failure_rate=float(data.get("max_failure_rate", 0.2)),
        max_latency_seconds=data.get("max_latency_seconds"),
    )


def _extract_graph(tree: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """Extract graph section and its pointer prefix (e.g., /graph or /workflow/graph)."""
    default: Tuple[Dict[str, Any], str] = ({}, "/graph")
    if not isinstance(tree, dict):
        return default
    if isinstance(tree.get("graph"), dict):
        return (tree.get("graph") or {}, "/graph")
    workflow = tree.get("workflow")
    if isinstance(workflow, dict) and isinstance(workflow.get("graph"), dict):
        return (workflow.get("graph") or {}, "/workflow/graph")
    return default


def _iter_llm_prompts(workflow_yaml: Dict[str, Any]) -> Iterable[PromptLocation]:
    """Yield prompt locations for LLM nodes in a Dify workflow DSL."""
    graph, graph_pointer = _extract_graph(workflow_yaml or {})
    buckets = []
    if "nodes" in graph:
        buckets.append(("nodes", graph.get("nodes") or []))
    if "data" in graph:
        buckets.append(("data", graph.get("data") or []))
    if not buckets:
        try:
            logger.debug("LLM prompt扫描: graph 下未找到 nodes/data", extra={"graph_keys": list(graph.keys())})
        except Exception:
            pass
    else:
        try:
            logger.debug(
                "LLM prompt扫描: graph 概况",
                extra={
                    "graph_keys": list(graph.keys()),
                    "bucket_sizes": {name: len(lst) for name, lst in buckets},
                },
            )
        except Exception:
            pass
    for bucket_name, nodes in buckets:
        try:
            logger.debug(
                "LLM prompt扫描: bucket 概况",
                extra={
                    "bucket": bucket_name,
                    "count": len(nodes),
                    "types_sample": [n.get("type") for n in nodes[:5] if isinstance(n, dict)],
                },
            )
        except Exception:
            pass
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or f"node_{idx}")
            data = node.get("data") or {}
            prompt_template = data.get("prompt_template")
            try:
                logger.debug(
                    "LLM 节点扫描",
                    extra={
                        "node_id": node_id,
                        "bucket": bucket_name,
                        "prompt_template_type": type(prompt_template).__name__,
                        "has_system_prompt": "system_prompt" in data,
                        "node_type": node.get("type"),
                    },
                )
            except Exception:
                pass
            # Support both dict(messages=[...]) and list of messages
            messages = []
            if isinstance(prompt_template, dict):
                messages = prompt_template.get("messages") or []
            elif isinstance(prompt_template, list):
                messages = prompt_template
            if messages:
                for mi, msg in enumerate(messages):
                    if not isinstance(msg, dict):
                        continue
                    text = msg.get("text")
                    if text is None:
                        continue
                    # pointer path reflects underlying structure
                    sub_path = "messages" if isinstance(prompt_template, dict) else ""
                    path = f"{graph_pointer}/{bucket_name}/{idx}/data/prompt_template/{sub_path + '/' if sub_path else ''}{mi}/text"
                    try:
                        logger.debug(
                            "LLM prompt匹配",
                            extra={"node_id": node_id, "path": path, "text_preview": str(text)[:120]},
                        )
                    except Exception:
                        pass
                    yield PromptLocation(node_id=node_id, path=path, text=str(text))
            else:
                try:
                    logger.debug(
                        "LLM 节点未找到 prompt_template.messages 或消息列表为空",
                        extra={"node_id": node_id, "prompt_template_type": type(prompt_template).__name__},
                    )
                except Exception:
                    pass
            # Some DSLs may have system_prompt fields
            if "system_prompt" in data:
                path = f"{graph_pointer}/{bucket_name}/{idx}/data/system_prompt"
                yield PromptLocation(node_id=node_id, path=path, text=str(data["system_prompt"]))


def _merge_messages_with_markers(messages: Sequence[Dict[str, Any]]) -> str:
    """Merge multiple messages into a single text with stable markers for round-trip editing."""
    parts: List[str] = []
    for msg in messages:
        role = msg.get("role") or "user"
        idx = msg.get("index", 0)
        start = f"<ROLE::{role}::{idx}>"
        end = f"</ROLE::{role}::{idx}>"
        parts.extend([start, str(msg.get("text", "")), end])
    return "\n".join(parts)


def _split_block_by_markers(block: PromptBlock, merged_text: str) -> Optional[List[str]]:
    """Split merged text back into message texts using role/index markers; returns None if markers are missing."""
    segments: Dict[Tuple[str, int], str] = {}
    pattern = re.compile(r"<ROLE::(?P<role>[^:>]+)::(?P<idx>\d+)>")
    pos = 0
    while True:
        match = pattern.search(merged_text, pos)
        if not match:
            break
        role = match.group("role")
        idx = int(match.group("idx"))
        end_token = f"</ROLE::{role}::{idx}>"
        start_content = match.end()
        end_content = merged_text.find(end_token, start_content)
        if end_content == -1:
            return None
        segments[(role, idx)] = merged_text[start_content:end_content]
        pos = end_content + len(end_token)
    texts: List[str] = []
    for msg in block.messages:
        key = (msg.get("role") or "user", int(msg.get("index", 0)))
        if key not in segments:
            return None
        texts.append(segments[key].strip())
    return texts


def _build_node_brief(node: Dict[str, Any], merged_text: str, node_id: str) -> NodeBrief:
    """Extract a concise node brief for rewriting context."""
    placeholders = _extract_placeholders(merged_text)
    node_name = str(node.get("data", {}).get("title") or node.get("data", {}).get("label") or node.get("id") or node_id)
    node_type = str(node.get("type") or "")
    role = str(node.get("data", {}).get("role") or node.get("data", {}).get("description") or node_type)
    input_schema = {}
    output_schema = {}
    data = node.get("data") or {}
    if isinstance(data.get("inputs"), dict):
        input_schema = data.get("inputs") or {}
    if isinstance(data.get("output_schema"), dict):
        output_schema = data.get("output_schema") or {}
    layout: List[Dict[str, Any]] = []
    prompt_template = data.get("prompt_template")
    messages = []
    if isinstance(prompt_template, dict):
        messages = prompt_template.get("messages") or []
    elif isinstance(prompt_template, list):
        messages = prompt_template
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        layout.append({"role": msg.get("role"), "index": idx})
    if "system_prompt" in data:
        layout.append({"role": "system", "index": len(layout)})
    return NodeBrief(
        node_id=node_id,
        node_name=node_name,
        node_type=node_type,
        role=role,
        input_schema=input_schema,
        output_schema=output_schema,
        downstream_constraints=[],
        placeholders=placeholders,
        message_layout=layout,
    )


def _iter_prompt_blocks(workflow_yaml: Dict[str, Any]) -> Iterable[PromptBlock]:
    """Yield node-level prompt blocks (system + user messages)."""
    graph, graph_pointer = _extract_graph(workflow_yaml or {})
    buckets = []
    if "nodes" in graph:
        buckets.append(("nodes", graph.get("nodes") or []))
    if "data" in graph:
        buckets.append(("data", graph.get("data") or []))
    for bucket_name, nodes in buckets:
        for idx, node in enumerate(nodes):
            if not isinstance(node, dict):
                continue
            node_id = str(node.get("id") or f"node_{idx}")
            data = node.get("data") or {}
            prompt_template = data.get("prompt_template")
            messages: List[Dict[str, Any]] = []
            if isinstance(prompt_template, dict):
                for mi, msg in enumerate(prompt_template.get("messages") or []):
                    if not isinstance(msg, dict) or msg.get("text") is None:
                        continue
                    sub_path = "messages"
                    path = f"{graph_pointer}/{bucket_name}/{idx}/data/prompt_template/{sub_path}/{mi}/text"
                    messages.append({"role": msg.get("role") or "user", "text": str(msg.get("text")), "path": path, "index": mi})
            elif isinstance(prompt_template, list):
                for mi, msg in enumerate(prompt_template):
                    if not isinstance(msg, dict) or msg.get("text") is None:
                        continue
                    path = f"{graph_pointer}/{bucket_name}/{idx}/data/prompt_template/{mi}/text"
                    messages.append({"role": msg.get("role") or "user", "text": str(msg.get("text")), "path": path, "index": mi})
            if "system_prompt" in data:
                path = f"{graph_pointer}/{bucket_name}/{idx}/data/system_prompt"
                messages.append({"role": "system", "text": str(data["system_prompt"]), "path": path, "index": len(messages)})
            if not messages:
                continue
            merged = _merge_messages_with_markers(messages)
            brief = _build_node_brief(node, merged, node_id)
            yield PromptBlock(
                node_id=node_id,
                node_name=brief.node_name,
                node_type=brief.node_type,
                messages=messages,
                merged_text=merged,
                brief=brief,
            )


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


def _tokenize_keywords(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z0-9_]+", text.lower()) if len(w) >= 3]


def _should_skip_for_intent(prompt_text: str, constraints: Sequence[str], expected_outputs: Sequence[str]) -> bool:
    """Heuristic: if prompt shares almost no keywords with expectations, skip rewriting."""
    keyword_source = " ".join(list(constraints or []) + list(expected_outputs or []))
    keywords = set(_tokenize_keywords(keyword_source))
    if not keywords:
        return False
    prompt_tokens = set(_tokenize_keywords(prompt_text))
    if not prompt_tokens:
        return True
    overlap = keywords & prompt_tokens
    ratio = len(overlap) / max(len(keywords), 1)
    return ratio < 0.15 and len(overlap) <= 1


class PromptDeltaDetector:
    """Detect prompt-related issues based on run outcomes and references."""

    def __init__(self, reference: ReferenceSpec) -> None:
        self.reference = reference

    def detect(self, samples: Sequence[ExecutionSample]) -> List[DetectedIssue]:
        issues: List[DetectedIssue] = []
        if not samples:
            return [DetectedIssue(kind="insufficient_data", severity="low", message="No execution samples available")]

        failure_count = sum(1 for s in samples if s.status not in {"success", "succeeded"})
        failure_rate = failure_count / max(len(samples), 1)
        if failure_rate > self.reference.max_failure_rate:
            issues.append(
                DetectedIssue(
                    kind="high_failure_rate",
                    severity="high",
                    message=f"Failure rate {failure_rate:.2f} exceeds threshold {self.reference.max_failure_rate:.2f}",
                    evidence_runs=[s.index for s in samples if s.status not in {"success", "succeeded"}],
                )
            )

        # Similarity check against expected outputs (first expected acts as baseline)
        if self.reference.expected_outputs:
            baseline = self.reference.expected_outputs[0]
            scores = [
                _similarity(baseline, s.output_text)
                for s in samples
                if s.output_text
            ]
            if scores:
                avg_sim = sum(scores) / len(scores)
                if avg_sim < self.reference.similarity_threshold:
                    issues.append(
                        DetectedIssue(
                            kind="low_similarity",
                            severity="medium" if avg_sim > self.reference.similarity_threshold * 0.7 else "high",
                            message=f"Average similarity {avg_sim:.2f} below threshold {self.reference.similarity_threshold:.2f}",
                            evidence_runs=[s.index for s in samples],
                        )
                    )

        # Constraint coverage: ensure constraint phrases appear in outputs
        if self.reference.constraints:
            missing_runs: Dict[str, List[int]] = {c: [] for c in self.reference.constraints}
            for sample in samples:
                for constraint in self.reference.constraints:
                    if constraint and constraint.lower() not in sample.output_text.lower():
                        missing_runs[constraint].append(sample.index)
            for constraint, run_ids in missing_runs.items():
                if len(run_ids) == len(samples):
                    issues.append(
                        DetectedIssue(
                            kind="constraint_missing",
                            severity="high",
                            message=f"Constraint not observed in outputs: {constraint}",
                            evidence_runs=run_ids,
                        )
                    )

        if self.reference.max_latency_seconds is not None:
            slow_runs = [
                s.index
                for s in samples
                if float(s.metrics.get("duration_seconds", 0)) > float(self.reference.max_latency_seconds)
            ]
            if slow_runs:
                issues.append(
                    DetectedIssue(
                        kind="latency_exceeded",
                        severity="medium",
                        message=(
                            f"Latency exceeded {self.reference.max_latency_seconds}s in runs: "
                            f"{', '.join(map(str, slow_runs))}"
                        ),
                        evidence_runs=slow_runs,
                    )
                )

        return issues


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


class PromptOptimizer:
    """High-level orchestrator that connects detection, generation, and reporting."""

    def __init__(
        self,
        *,
        default_output_root: Optional[str | Path] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        validate_llm: bool = True,
    ) -> None:
        self.default_output_root = Path(default_output_root).expanduser() if default_output_root else None
        self.llm_config = llm_config or {}
        if validate_llm and self.llm_config:
            _ensure_llm_available(self.llm_config)

    def optimize_from_runs(
        self,
        workflow_id: str,
        run_results: Sequence[Dict[str, Any]],
        *,
        reference_path: Optional[str | Path] = None,
        reference_spec: Optional[ReferenceSpec] = None,
        reference_texts: Optional[Sequence[Optional[str]]] = None,
        workflow_yaml: Optional[Tuple[Dict[str, Any], Path]] = None,
        output_root: Optional[str | Path] = None,
        apply_patches: bool = True,
    ) -> OptimizationReport:
        """Generate prompt patches from execution results and references."""
        reference = reference_spec or _load_reference_spec(reference_path)
        samples = self._normalize_runs(run_results)
        try:
            logger.info(
                "优化输入就绪",
                extra={
                    "workflow_id": workflow_id,
                    "runs": len(samples),
                    "has_reference_texts": bool(reference_texts),
                    "llm_configured": bool(self.llm_config),
                },
            )
        except Exception:
            pass

        if workflow_yaml:
            workflow_tree, workflow_path = workflow_yaml
        else:
            workflow_tree, workflow_path = load_workflow_yaml(
                workflow_id,
                output_dir=output_root or self.default_output_root,
            )

        detector = PromptDeltaDetector(reference)
        issues = detector.detect(samples)

        llm_output_issues, judge_stats = _llm_output_judge(
            samples,
            reference_texts,
            self.llm_config,
            constraints=reference.constraints,
            expected_outputs=reference.expected_outputs,
        )
        if llm_output_issues:
            issues.extend(llm_output_issues)
        else:
            # 保守策略：判定失败时也留下一条提示，避免“无声失败”
            if judge_stats.get("llm_judge_failures"):
                issues.append(
                    DetectedIssue(
                        kind="llm_judge_unreliable",
                        severity="medium",
                        message="LLM 判定失败或不可用，参考验收未完成",
                        evidence_runs=[s.index for s in samples],
                    )
                )
        try:
            logger.info(
                "判定阶段完成",
                extra={
                    "workflow_id": workflow_id,
                    "issues": len(issues),
                    "llm_judge": judge_stats,
                },
            )
        except Exception:
            pass

        prompt_blocks = list(_iter_prompt_blocks(workflow_tree))
        judge = _LlmPromptJudge(self.llm_config) if self.llm_config else None
        # Determine which prompts need change: if judge exists, use it; else default to "issues exist"
        filtered_blocks: List[PromptBlock] = []
        skip_intent_blocks = 0
        total_prompts = len(prompt_blocks)
        for block in prompt_blocks:
            prompt_text = block.merged_text
            if _should_skip_for_intent(prompt_text, reference.constraints, reference.expected_outputs):
                skip_intent_blocks += 1
                try:
                    logger.info(
                        "跳过意图不相关的 prompt block",
                        extra={"workflow_id": workflow_id, "node_id": block.node_id},
                    )
                except Exception:
                    pass
                continue
            needs = True if not judge else judge.needs_change(prompt_text, issues, reference.constraints)
            if needs:
                filtered_blocks.append(block)
        judged_no_change = total_prompts - len(filtered_blocks)
        prompts_skip_ratio = None
        if judge and total_prompts > 0:
            prompts_skip_ratio = judged_no_change / total_prompts

        patches: List[PromptPatch] = []
        fail_signal = _build_fail_signal_from_issues(issues, reference.constraints)
        actions = _actions_from_issues(issues, reference.constraints)
        use_dspy = self.llm_config.get("use_dspy_optimizer", True)
        if use_dspy:
            dspy_optimizer = _DspyPromptOptimizer(self.llm_config)
            try:
                patches = dspy_optimizer.optimize_blocks(
                    workflow_id,
                    filtered_blocks,
                    samples,
                    reference_texts or [],
                    reference.constraints,
                    fail_signal=fail_signal,
                )
            except Exception as exc:  # noqa: BLE001
                try:
                    logger.exception(
                        "DSPy 提示词优化失败",
                        extra={"workflow_id": workflow_id, "error": str(exc)},
                    )
                except Exception:
                    pass
                raise
        else:
            try:
                logger.warning(
                    "已禁用 DSPy 优化器，提示词保持不变",
                    extra={"workflow_id": workflow_id},
                )
            except Exception:
                pass

        rewriter = None
        enable_rewrite = self.llm_config.get("enable_prompt_rewrite", True)
        if enable_rewrite and self.llm_config.get("url") and self.llm_config.get("model"):
            dspy_rewriter = _DspyRewriter(self.llm_config)
            rewriter = dspy_rewriter if dspy_rewriter.available else _LlmRewriter(self.llm_config)

        if rewriter:
            context_text = _format_fail_signal(fail_signal)
            refined: List[PromptPatch] = []
            for patch in patches:
                try:
                    new_text = rewriter.rewrite(patch.new, issues, reference.constraints, context=context_text)
                    patch.new = new_text
                    refined.append(patch)
                except Exception:
                    refined.append(patch)
            patches = refined
        else:
            try:
                logger.info(
                    "未启用 LLM 重写器，使用规则补丁结果",
                    extra={"patch_count": len(patches), "workflow_id": workflow_id},
                )
            except Exception:
                pass

        report = OptimizationReport(
            workflow_id=workflow_id,
            issues=issues,
            patches=patches,
            actions=actions,
            yaml_path=workflow_path,
            reference_path=Path(reference_path) if reference_path else None,
            stats={
                "runs": len(samples),
                "failures": len([s for s in samples if s.status not in {"success", "succeeded"}]),
                "reference_texts": len([t for t in (reference_texts or []) if t]),
                "prompts_total": total_prompts,
                "prompts_need_change": len(filtered_blocks),
                "prompts_no_change": judged_no_change,
                "prompts_skip_intent": skip_intent_blocks,
                "prompts_skip_ratio": prompts_skip_ratio,
                **judge_stats,
            },
        )

        patched_path: Optional[Path] = None
        if apply_patches and patches and workflow_tree:
            patched_path = self._apply_patches(workflow_id, workflow_tree, workflow_path, patches, output_root or self.default_output_root)
        report.patched_path = patched_path

        self._maybe_write_report(report, output_root or self.default_output_root)
        return report

    def _normalize_runs(self, run_results: Sequence[Dict[str, Any]]) -> List[ExecutionSample]:
        samples: List[ExecutionSample] = []
        for idx, run in enumerate(run_results, start=1):
            try:
                status = _status_from_run(run)
                output_text = _extract_output_text(run)
                metrics: Dict[str, Any] = {}
                if isinstance(run, dict):
                    metrics = run.get("metrics") or {}
                    if not metrics and isinstance(run.get("data"), dict):
                        metrics = run.get("data", {}).get("metrics", {}) or {}
                samples.append(ExecutionSample(index=idx, status=status, output_text=output_text, metrics=metrics, raw=run or {}))
            except Exception:
                samples.append(ExecutionSample(index=idx, status="unknown", output_text=str(run), metrics={}, raw=run or {}))
        return samples

    def _maybe_write_report(self, report: OptimizationReport, output_root: Optional[str | Path]) -> None:
        if not output_root:
            return
        try:
            root = Path(output_root).expanduser()
            safe_id = _safe_dirname_from_id(report.workflow_id)
            target_dir = root / safe_id
            target_dir.mkdir(parents=True, exist_ok=True)
            out_path = target_dir / "prompt_patches.json"
            payload = {
                "workflow_id": report.workflow_id,
                "yaml_path": str(report.yaml_path),
                "reference_path": str(report.reference_path) if report.reference_path else None,
                "issues": [
                    {
                        "kind": it.kind,
                        "severity": it.severity,
                        "message": it.message,
                        "evidence_runs": it.evidence_runs,
                    }
                    for it in report.issues
                ],
                "patches": [
                    {
                        "node_id": p.node_id,
                        "field_path": p.field_path,
                        "old": p.old,
                        "new": p.new,
                        "rationale": p.rationale,
                        "confidence": p.confidence,
                        "evidence_runs": p.evidence_runs,
                    }
                    for p in report.patches
                ],
                "actions": [
                    {
                        "type": a.type,
                        "target": a.target,
                        "content": a.content,
                        "evidence_runs": a.evidence_runs,
                    }
                    for a in report.actions
                ],
                "stats": report.stats,
                "patched_path": str(report.patched_path) if report.patched_path else None,
            }
            out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info("Wrote prompt optimization report", extra={"path": str(out_path)})
        except Exception as exc:  # noqa: BLE001
            try:
                logger.warning("Failed to write prompt optimization report", extra={"error": str(exc)})
            except Exception:
                pass

    def _apply_patches(
        self,
        workflow_id: str,
        workflow_tree: Dict[str, Any],
        workflow_path: Path,
        patches: Sequence[PromptPatch],
        output_root: Optional[str | Path],
    ) -> Optional[Path]:
        if not output_root:
            return None
        try:
            new_tree = copy.deepcopy(workflow_tree)
            sorted_patches = sorted(patches, key=lambda p: _pointer_sort_key(p.field_path))
            for patch in sorted_patches:
                _set_by_pointer(new_tree, patch.field_path, patch.new)
            root = Path(output_root).expanduser()
            safe_id = _safe_dirname_from_id(workflow_id)
            target_dir = root / safe_id
            target_dir.mkdir(parents=True, exist_ok=True)
            patched_path = target_dir / f"{Path(workflow_path).stem}_patched.yml"
            patched_path.write_text(yaml.safe_dump(new_tree, allow_unicode=True, sort_keys=False), encoding="utf-8")
            logger.info("Wrote patched workflow copy", extra={"path": str(patched_path), "workflow_id": workflow_id})
            return patched_path
        except Exception as exc:  # noqa: BLE001
            try:
                logger.warning("Failed to write patched workflow copy", extra={"workflow_id": workflow_id, "error": str(exc)})
            except Exception:
                pass
        return None


def _set_by_pointer(tree: Dict[str, Any], pointer: str, value: Any) -> None:
    if not pointer.startswith("/"):
        return
    parts = [p for p in pointer.split("/") if p]
    cur: Any = tree
    for part in parts[:-1]:
        if isinstance(cur, list):
            idx = int(part)
            while idx >= len(cur):
                cur.append({})
            if not isinstance(cur[idx], (dict, list)):
                cur[idx] = {}
            cur = cur[idx]
        else:
            if part not in cur or not isinstance(cur[part], (dict, list)):
                cur[part] = {}
            cur = cur[part]
    last = parts[-1]
    if isinstance(cur, list):
        idx = int(last)
        while idx >= len(cur):
            cur.append({})
        cur[idx] = value
    else:
        cur[last] = value


def _pointer_sort_key(pointer: str) -> List[Tuple[int, Any]]:
    parts = [p for p in pointer.split("/") if p]
    key: List[Tuple[int, Any]] = []
    for part in parts:
        if part.isdigit():
            key.append((0, int(part)))
        else:
            key.append((1, part))
    return key


def _llm_output_judge(
    samples: Sequence[ExecutionSample],
    reference_texts: Optional[Sequence[Optional[str]]],
    llm_cfg: Optional[Dict[str, Any]],
    *,
    constraints: Sequence[str] | None = None,
    expected_outputs: Sequence[str] | None = None,
) -> Tuple[List[DetectedIssue], Dict[str, Any]]:
    """
    Use an external LLM to compare each output with its paired reference text.

    Expected alignment: reference_texts[i] corresponds to samples[i] (1-based index inside ExecutionSample).
    """
    stats = {
        "llm_judge_used": False,
        "llm_judge_calls": 0,
        "llm_judge_failures": 0,
        "aligned_references": len([t for t in (reference_texts or []) if t]),
    }
    if not reference_texts or not llm_cfg or not llm_cfg.get("enable_output_judge", True):
        return [], stats
    url = llm_cfg.get("url")
    model = llm_cfg.get("model")
    api_key = llm_cfg.get("api_key")
    timeout = float(llm_cfg.get("judge_timeout_seconds", 20))
    max_calls = int(llm_cfg.get("max_judge_calls", len(samples)))
    if max_calls < 0:
        max_calls = len(samples)
    if not url or not model:
        return [], stats
    issues: List[DetectedIssue] = []
    cache: Dict[Tuple[str, str], Dict[str, str]] = {}
    stats["llm_judge_used"] = True

    def _infer_severity(missing: list[str], incorrect: list[str], fmt_ok: bool) -> str:
        if incorrect or (missing and not fmt_ok):
            return "high"
        if missing or not fmt_ok:
            return "medium"
        return "medium"

    paired_refs = list(reference_texts or [])
    if len(paired_refs) == 1 and len(samples) > 1:
        paired_refs = paired_refs * len(samples)

    for sample, ref in zip(samples, paired_refs):
        if stats["llm_judge_calls"] >= max_calls:
            break
        if not ref:
            continue
        try:
            import requests

            key = (sample.output_text or "", ref)
            cached = cache.get(key)
            if cached:
                verdict_raw = cached["verdict"]
            else:
                prompt_lines = [
                                    "You will compare a workflow output to a gold reference.",
                                    "",
                                    "Step 1 – Extract must-have content units from the reference:",
                                    "- Break the reference into small, atomic MUST-HAVE units.",
                                    "- Each unit should be one requirement, fact, constraint, key explanation,",
                                    "  or important detail that MUST all be present in a correct output.",
                                    "- Ignore superficial phrasing and formatting; focus on meaning.",
                                    "",
                                    "Step 2 – Compare workflow output against these units:",
                                    "- For each must-have unit, check whether the workflow output FULLY covers it",
                                    "  with the same meaning.",
                                    "- If a unit is only partially covered, significantly weaker, or too vague,",
                                    "  treat it as MISSING.",
                                    "- If the workflow output states something that contradicts the reference,",
                                    "  treat it as INCORRECT.",
                                    "- If the workflow output adds content that is not supported by the reference",
                                    "  (hallucinated details), list these as INCORRECT as well.",
                                    "",
                                    "Step 3 – Check structure / format:",
                                    "- Decide whether the workflow output structure/format is compatible with the",
                                    "  intent of the reference (for example: required JSON keys, required sections,",
                                    "  required fields, or other structural constraints implied by the reference).",
                                    "- If there are major structural problems that make the output unusable,",
                                    "  set format_ok to false.",
                                    "",
                                    "Decision rules (VERY IMPORTANT):",
                                    '- verdict MUST be \"fail\" if ANY of the following is true:',
                                    "  - missing_items is not empty, OR",
                                    "  - incorrect_items is not empty, OR",
                                    "  - format_ok is false.",
                                    '- verdict may be \"pass\" ONLY IF ALL of the following are true:',
                                    "  - Every must-have unit is fully and correctly covered, AND",
                                    "  - There is no incorrect or hallucinated content, AND",
                                    "  - format_ok is true.",
                                    "- Be strict. If you are uncertain whether a unit is fully satisfied,",
                                    "  treat it as missing or incorrect and FAIL.",
                                    "",
                                    "Output format (this is mandatory):",
                                    "- Return ONLY a single JSON object, no surrounding text, no markdown,",
                                    "  no explanation, no extra keys.",
                                    "- The JSON object MUST have exactly these keys:",
                                    '  - verdict: \"pass\" or \"fail\"',
                                    "  - missing_items: an array of strings. Each string describes ONE missing",
                                    "    must-have unit.",
                                    "  - incorrect_items: an array of strings. Each string describes ONE incorrect",
                                    "    or hallucinated unit.",
                                    "  - format_ok: true or false.",
                                    "",
                                    "Now perform the evaluation using the rules above.",
                                    "",
                                    "Reference (gold):",
                                    ref,
                                    "",
                                    "Workflow output:",
                                    sample.output_text or "",
                                ]
                if constraints:
                    prompt_lines.append("Constraints (must be satisfied):")
                    for c in constraints:
                        prompt_lines.append(f"- {c}")
                if expected_outputs:
                    prompt_lines.append("Expected baseline example:")
                    prompt_lines.append(str(expected_outputs[0]))
                system_msg = """You are a strict, deterministic judge that evaluates whether a workflow output matches a gold reference.
                                - Treat the reference as the ONLY source of truth.
                                - You MUST be conservative and strict: if you are unsure, choose FAIL.
                                - Ignore wording or style differences and focus on meaning.
                                - You MUST return exactly one JSON object and NOTHING else.
                                - Do NOT include explanations, comments, or markdown code fences.
                                """
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": "\n".join(prompt_lines)},
                    ],
                    "stream": False,
                }
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                try:
                    logger.info(
                        "调用 LLM 进行输出判定",
                        extra={
                            "workflow_run": sample.index,
                            "model": model,
                            "url": url,
                            "timeout": timeout,
                            "has_api_key": bool(api_key),
                        },
                    )
                except Exception:
                    pass
                resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
                stats["llm_judge_calls"] += 1
                resp.raise_for_status()
                data = resp.json()
                verdict_raw = _extract_llm_text_response(data)
                cache[key] = {"verdict": verdict_raw}
                try:
                    logger.debug(
                        "LLM 判定返回",
                        extra={
                            "workflow_run": sample.index,
                            "model": model,
                            "url": url,
                            "verdict_raw": verdict_raw[:500],
                        },
                    )
                except Exception:
                    pass

            if not verdict_raw:
                continue
            verdict_lower = verdict_raw.lower()
            parsed = None
            try:
                parsed = _json.loads(verdict_raw)
            except Exception:
                start = verdict_raw.find("{")
                end = verdict_raw.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = _json.loads(verdict_raw[start : end + 1])
                    except Exception:
                        parsed = None

            verdict = str((parsed or {}).get("verdict") or "").lower()
            missing_items = (parsed or {}).get("missing_items") or []
            incorrect_items = (parsed or {}).get("incorrect_items") or []
            fmt_ok = bool((parsed or {}).get("format_ok", True))
            reason = (parsed or {}).get("reason") or verdict_raw
            severity = str((parsed or {}).get("severity") or "").lower() or _infer_severity(missing_items, incorrect_items, fmt_ok)

            if not verdict:
                verdict = "fail" if "fail" in verdict_lower else "pass" if "pass" in verdict_lower else "fail"

            if verdict == "pass" and not missing_items and not incorrect_items and fmt_ok:
                continue

            # 保守策略：只要缺失/错误/格式问题，即视为 fail
            if verdict == "pass" and (missing_items or incorrect_items or not fmt_ok):
                verdict = "fail"
            if parsed is None:
                severity = "high"
                fmt_ok = False
            issues.append(
                DetectedIssue(
                    kind="llm_output_mismatch",
                    severity=severity if severity in {"low", "medium", "high"} else "medium",
                    message=(
                        f"LLM judge flagged mismatch: {reason} | "
                        f"missing_items={missing_items or []} | incorrect_items={incorrect_items or []} | format_ok={fmt_ok}"
                    ),
                    evidence_runs=[sample.index],
                )
            )
        except Exception as exc:  # noqa: BLE001
            try:
                body = ""
                if "resp" in locals():
                    try:
                        body = str(resp.text)[:500]  # type: ignore
                    except Exception:
                        body = ""
                logger.warning(
                    "LLM 输出判定失败",
                    extra={
                        "workflow_run": sample.index,
                        "error": str(exc),
                        "model": model,
                        "url": url,
                        "response_body": body,
                    },
                )
            except Exception:
                pass
            stats["llm_judge_failures"] += 1
            continue
    return issues, stats


def _ensure_llm_available(cfg: Dict[str, Any]) -> None:
    """Validate LLM configuration and endpoint reachability."""
    url = cfg.get("url")
    model = cfg.get("model")
    if not url or not model:
        raise ValueError("LLM 配置缺少 url 或 model，无法启用基于 LLM 的判定/重写")
    if not str(url).startswith(("http://", "https://")):
        raise ValueError(f"LLM url 非法: {url}")
    try:
        import requests  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        raise ImportError("LLM 判定需要 requests 库") from exc

    timeout = float(cfg.get("health_timeout_seconds", 5))
    api_key = cfg.get("api_key")

    def _probe(endpoint: str) -> Optional[str]:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=timeout)
        if resp.status_code < 300:
            return None
        return f"status={resp.status_code}, body={resp.text[:200]}"

    endpoints = [str(url).rstrip("/")]
    if "/v1/" not in endpoints[0]:
        endpoints.append(f"{endpoints[0]}/v1/chat/completions")

    errors: List[str] = []
    for ep in endpoints:
        try:
            err = _probe(ep)
            if err is None:
                return
            errors.append(f"{ep}: {err}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{ep}: {exc}")
    raise RuntimeError(f"LLM endpoint 校验失败: {'; '.join(errors)}")


def _adapt_dspy_api_base(url: Optional[str]) -> Optional[str]:
    """
    DSPy expects an api_base without the '/chat/completions' suffix that OpenAI-compatible
    endpoints often expose. Other components still need the full URL, so we normalize only
    for the DSPy LM constructor.
    """
    if not url:
        return url
    trimmed = str(url).strip()
    if not trimmed:
        return None
    trimmed = trimmed.rstrip("/")
    lowered = trimmed.lower()
    for suffix in ("/chat/completions", "/completions"):
        if lowered.endswith(suffix):
            return trimmed[: -len(suffix)]
    return trimmed


def _extract_placeholders(text: str) -> List[str]:
    """Extract placeholder tokens (e.g., {{var}}, ${var}) in order of appearance."""
    if not text:
        return []
    seen = set()
    result: List[str] = []
    for pattern in _PLACEHOLDER_PATTERNS:
        for match in pattern.findall(text):
            if match not in seen:
                seen.add(match)
                result.append(match)
    return result


def _mask_placeholders(text: str, placeholders: Sequence[str]) -> Tuple[str, Dict[str, str]]:
    """Replace placeholders with unlikely tokens to avoid LLM tampering."""
    masked = text
    token_map: Dict[str, str] = {}
    for idx, ph in enumerate(placeholders):
        token = f"<<<PH_{idx}>>>"
        masked = masked.replace(ph, token)
        token_map[token] = ph
    return masked, token_map


def _restore_placeholders(text: str, token_map: Dict[str, str]) -> str:
    restored = text
    for token, ph in token_map.items():
        restored = restored.replace(token, ph)
    return restored


def _tokens_intact(original_masked: str, rewritten: str, token_map: Dict[str, str]) -> bool:
    for token in token_map:
        if original_masked.count(token) != rewritten.count(token):
            return False
    return True


def _placeholders_preserved(original_placeholders: Sequence[str], new_text: str) -> bool:
    return set(original_placeholders) == set(_extract_placeholders(new_text))


def _build_fail_signal_from_issues(issues: Sequence[DetectedIssue], constraints: Sequence[str]) -> FailSignal:
    """Map detected issues into structured fail signals for rewrites."""
    tags: List[str] = []
    missing_fields: List[str] = []
    format_errors: List[str] = []
    constraint_gaps: List[str] = []
    semantic_gaps: List[str] = []
    for issue in issues:
        kind = (issue.kind or "").lower()
        msg = (issue.message or "").lower()
        if kind not in tags:
            tags.append(kind)
        if "format" in msg or "json" in msg:
            format_errors.append(issue.message)
        if "missing" in msg or "not observed" in msg:
            missing_fields.append(issue.message)
        if "constraint" in msg:
            constraint_gaps.append(issue.message)
        if "similarity" in msg or "mismatch" in msg:
            semantic_gaps.append(issue.message)
    for c in constraints or []:
        if c and all(c.lower() not in item.lower() for item in constraint_gaps):
            constraint_gaps.append(f"Ensure constraint is satisfied: {c}")
    return FailSignal(
        tags=tags,
        missing_fields=missing_fields,
        format_errors=format_errors,
        constraint_gaps=constraint_gaps,
        semantic_gaps=semantic_gaps,
        examples=[],
    )


def _format_fail_signal(signal: FailSignal) -> str:
    lines = []
    if signal.tags:
        lines.append(f"tags: {', '.join(signal.tags)}")
    if signal.missing_fields:
        lines.append("missing_fields:")
        lines.extend(f"- {m}" for m in signal.missing_fields)
    if signal.format_errors:
        lines.append("format_errors:")
        lines.extend(f"- {m}" for m in signal.format_errors)
    if signal.constraint_gaps:
        lines.append("constraint_gaps:")
        lines.extend(f"- {m}" for m in signal.constraint_gaps)
    if signal.semantic_gaps:
        lines.append("semantic_gaps:")
        lines.extend(f"- {m}" for m in signal.semantic_gaps)
    return "\n".join(lines)


def _format_node_brief(brief: Optional[NodeBrief]) -> str:
    if not brief:
        return ""
    parts = [
        f"node_id: {brief.node_id}",
        f"name: {brief.node_name}",
        f"type: {brief.node_type}",
    ]
    if brief.role:
        parts.append(f"role: {brief.role}")
    if brief.input_schema:
        parts.append(f"input_schema keys: {list(brief.input_schema.keys())}")
    if brief.output_schema:
        parts.append(f"output_schema keys: {list(brief.output_schema.keys())}")
    if brief.downstream_constraints:
        parts.append(f"downstream_constraints: {brief.downstream_constraints}")
    if brief.message_layout:
        parts.append(f"message_layout: {brief.message_layout}")
    if brief.placeholders:
        parts.append(f"placeholders (do not change): {brief.placeholders}")
    return "\n".join(parts)


def _build_block_context(block: PromptBlock, fail_signal: FailSignal, constraints: Sequence[str]) -> str:
    parts = [
        "Node brief:",
        _format_node_brief(block.brief),
        "",
        "Failure signals:",
        _format_fail_signal(fail_signal),
    ]
    if constraints:
        parts.append("")
        parts.append("Global constraints:")
        parts.extend(f"- {c}" for c in constraints)
    if block.brief and block.brief.placeholders:
        parts.append("")
        parts.append("Placeholders (must remain exactly):")
        parts.extend(f"- {ph}" for ph in block.brief.placeholders)
    return "\n".join(part for part in parts if part is not None)


def _validate_block_rewrite(block: PromptBlock, new_text: str) -> bool:
    """Validate rewritten block keeps markers and placeholders."""
    if block.brief and not _placeholders_preserved(block.brief.placeholders, new_text):
        return False
    markers_ok = _split_block_by_markers(block, new_text) is not None
    return markers_ok


def _normalize_prompt_state_mode(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if raw in {"true", "1", "yes", "auto"}:
        return "auto"
    if raw in {"force", "strict"}:
        return "force"
    return "off"


def _parse_prompt_sections(text: str) -> Dict[str, str]:
    headers = {
        "ROLE": "role",
        "TASK": "task",
        "GLOBAL CONSTRAINTS": "global_constraints",
        "CHECKPOINT RULES": "checkpoint_rules",
        "STRICTNESS LEVEL": "strictness_level",
        "OUTPUT JSON SCHEMA": "schema",
        "JSON SCHEMA": "schema",
        "NEGATIVE EXAMPLES": "negative_examples",
    }
    sections: Dict[str, List[str]] = {}
    current: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.endswith(":"):
            key = stripped[:-1].strip().upper()
            if key in headers:
                current = headers[key]
                sections[current] = []
                continue
        if current:
            sections[current].append(line)
    return {key: "\n".join(lines).strip() for key, lines in sections.items() if lines}


def _parse_bullet_list(text: str) -> List[str]:
    items: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            stripped = stripped[1:].strip()
        if stripped:
            items.append(stripped)
    return items


def _parse_checkpoint_rules(text: str) -> Dict[str, str]:
    rules: Dict[str, str] = {}
    for line in _parse_bullet_list(text):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            rules[key] = value
    return rules


def _parse_negative_examples(text: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            stripped = stripped[1:].strip()
        lower = stripped.lower()
        if lower.startswith("wrong:"):
            if current:
                examples.append(current)
            current = {"bad_output": stripped.split(":", 1)[1].strip()}
        elif lower.startswith("reason:"):
            if current:
                current["reason"] = stripped.split(":", 1)[1].strip()
        else:
            if not current:
                current = {"bad_output": stripped}
            else:
                current["bad_output"] = f"{current.get('bad_output', '')}\n{stripped}".strip()
    if current:
        examples.append(current)
    return examples


def _build_prompt_state(
        prompt_text: str,
        constraints: Sequence[str],
        placeholders_text: str,
        mode: str,
) -> PromptState:
    sections = _parse_prompt_sections(prompt_text)
    structured = bool(sections)
    role = sections.get("role", "").strip()
    task = sections.get("task", "").strip()
    schema = sections.get("schema", "").strip()
    strictness = 3
    strictness_text = sections.get("strictness_level", "").strip()
    if strictness_text:
        match = re.search(r"\d+", strictness_text)
        if match:
            strictness = max(1, min(5, int(match.group(0))))
    global_constraints = _parse_bullet_list(sections.get("global_constraints", ""))
    checkpoint_rules = _parse_checkpoint_rules(sections.get("checkpoint_rules", ""))
    negative_examples = _parse_negative_examples(sections.get("negative_examples", ""))
    placeholders = _extract_placeholders(prompt_text)
    if not placeholders and placeholders_text:
        placeholders = [p.strip() for p in placeholders_text.split(";") if p.strip()]
    base_prompt = ""
    if not structured:
        base_prompt = prompt_text
    state = PromptState(
        base_prompt=base_prompt,
        role=role,
        task=task,
        schema=schema,
        checkpoint_rules=checkpoint_rules,
        strictness_level=strictness,
        negative_examples=negative_examples,
        global_constraints=global_constraints,
        placeholders=placeholders,
        structured=structured,
    )
    if mode == "force" and not structured and not state.base_prompt:
        state.task = prompt_text
        state.structured = True
    for constraint in constraints:
        if constraint and constraint not in state.global_constraints:
            state.global_constraints.append(constraint)
    return state


def _render_prompt_state(state: PromptState, context_text: str = "") -> str:
    parts: List[str] = []
    if state.base_prompt:
        parts.append(state.base_prompt)
    if state.role:
        parts.append(f"ROLE:\n{state.role}")
    if state.task:
        parts.append(f"TASK:\n{state.task}")
    if state.global_constraints:
        parts.append("GLOBAL CONSTRAINTS:")
        parts.extend(f"- {c}" for c in state.global_constraints)
    if state.checkpoint_rules:
        parts.append("CHECKPOINT RULES:")
        for key, value in state.checkpoint_rules.items():
            parts.append(f"- {key}: {value}")
    parts.append(f"STRICTNESS LEVEL:\n{state.strictness_level}")
    if state.schema:
        parts.append("OUTPUT JSON SCHEMA:")
        parts.append(state.schema)
    if state.negative_examples:
        parts.append("NEGATIVE EXAMPLES:")
        for ex in state.negative_examples:
            bad = ex.get("bad_output") or ex.get("bad") or ""
            if bad:
                parts.append(f"- Wrong: {bad}")
            reason = ex.get("reason") or ""
            if reason:
                parts.append(f"  Reason: {reason}")
    if context_text:
        parts.append("NODE CONTEXT:")
        parts.append(context_text)
    if state.placeholders:
        parts.append("PLACEHOLDERS (MUST REMAIN EXACTLY):")
        parts.extend(f"- {ph}" for ph in state.placeholders)
    return "\n\n".join(part for part in parts if part)


def _serialize_prompt_state(state: PromptState) -> str:
    payload = {
        "base_prompt": state.base_prompt,
        "role": state.role,
        "task": state.task,
        "schema": state.schema,
        "checkpoint_rules": state.checkpoint_rules,
        "strictness_level": state.strictness_level,
        "negative_examples": state.negative_examples,
        "global_constraints": state.global_constraints,
        "placeholders": state.placeholders,
        "version": state.version,
        "schema_tighten_count": state.schema_tighten_count,
        "schema_frozen": state.schema_frozen,
    }
    try:
        return _json.dumps(payload, ensure_ascii=False)
    except Exception:
        return _coerce_text(payload)


def _safe_json_from_text(text: str) -> Optional[Any]:
    if not text:
        return None
    try:
        return _json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return _json.loads(text[start: end + 1])
            except Exception:
                return None
    return None


def _parse_enum_constraint(constraint: str) -> Optional[List[str]]:
    lower = constraint.lower()
    if "enum" not in lower:
        return None
    parts = constraint.split(":", 1)
    if len(parts) != 2:
        return None
    payload = parts[1].strip()
    if not payload:
        return None
    try:
        return _json.loads(payload.replace("'", "\""))
    except Exception:
        return None


def _apply_schema_constraint(state: PromptState, field: str, constraint: str) -> None:
    if not field or not constraint:
        return
    schema_obj = _safe_json_from_text(state.schema)
    if not isinstance(schema_obj, dict):
        state.global_constraints.append(f"Schema constraint for {field}: {constraint}")
        return
    props = schema_obj.setdefault("properties", {})
    field_schema = props.get(field)
    if not isinstance(field_schema, dict):
        field_schema = {}
    enum_values = _parse_enum_constraint(constraint)
    if enum_values:
        field_schema["enum"] = enum_values
    else:
        desc = str(field_schema.get("description") or "").strip()
        if constraint not in desc:
            field_schema["description"] = f"{desc} {constraint}".strip()
    props[field] = field_schema
    try:
        state.schema = _json.dumps(schema_obj, ensure_ascii=False, indent=2)
    except Exception:
        state.schema = _coerce_text(schema_obj)


def _apply_prompt_actions(
        state: PromptState,
        actions: Sequence[Any],
        *,
        action_budget: Optional[int] = DEFAULT_ACTION_BUDGET,
        max_constraints: int = DEFAULT_MAX_CONSTRAINTS,
        max_strictness: int = 5,
        schema_tighten_limit: int = DEFAULT_SCHEMA_TIGHTEN_LIMIT,
        normalize: bool = True,
) -> PromptState:
    if normalize:
        normalized = _normalize_actions(actions, state, action_budget=action_budget)
    else:
        normalized = list(actions)
    for action in normalized:
        payload = action
        if isinstance(action, PromptAction):
            payload = {
                "type": action.type,
                "target": action.target,
                "content": action.content,
            }
        if not isinstance(payload, dict):
            continue
        action_type = str(payload.get("type") or payload.get("action") or "").strip().lower()
        if action_type in {"add_checkpoint_rule", "modify_checkpoint_rule", "add_rule"}:
            checkpoint = str(payload.get("checkpoint") or payload.get("target") or "").strip()
            content = str(payload.get("content") or payload.get("rule") or "").strip()
            if checkpoint and content:
                state.checkpoint_rules[checkpoint] = content
        elif action_type in {"add_global_constraint", "add_constraint"}:
            content = str(payload.get("content") or payload.get("constraint") or "").strip()
            if content and content not in state.global_constraints:
                state.global_constraints.append(content)
        elif action_type == "tighten_schema":
            if state.schema_frozen:
                continue
            field = str(payload.get("field") or payload.get("target") or "").strip()
            constraint = str(payload.get("constraint") or payload.get("content") or "").strip()
            if not constraint:
                continue
            if field and not _safe_json_from_text(constraint):
                temp_state = PromptState(schema=state.schema)
                _apply_schema_constraint(temp_state, field, constraint)
                new_schema_text = temp_state.schema
            else:
                new_schema_text = constraint
            if not _safe_json_from_text(new_schema_text):
                if constraint and constraint not in state.global_constraints:
                    state.global_constraints.append(constraint)
                continue
            current_score = _schema_strength_score(state.schema)
            new_score = _schema_strength_score(new_schema_text)
            if new_score < current_score:
                continue
            state.schema = new_schema_text
            state.schema_tighten_count += 1
            if state.schema_tighten_count >= schema_tighten_limit:
                state.schema_frozen = True
        elif action_type == "add_negative_example":
            example = payload.get("example") or {}
            if isinstance(example, dict):
                state.negative_examples.append(example)
            else:
                bad_output = str(payload.get("bad_output") or payload.get("output") or example).strip()
                reason = str(payload.get("reason") or "").strip()
                if bad_output:
                    entry: Dict[str, Any] = {"bad_output": bad_output}
                    if reason:
                        entry["reason"] = reason
                    state.negative_examples.append(entry)
        elif action_type == "increase_strictness":
            amount = payload.get("amount")
            try:
                delta = int(amount) if amount is not None else 1
            except Exception:
                delta = 1
            state.strictness_level = max(1, min(max_strictness, state.strictness_level + delta))
    if max_constraints is not None and len(state.global_constraints) > max_constraints:
        state.global_constraints = state.global_constraints[-max_constraints:]
    state.version += 1
    state.structured = True
    return state


def _actions_from_fail_signal(
        signal: FailSignal,
        issues: Sequence[DetectedIssue],
) -> List[PromptAction]:
    evidence: List[int] = sorted({idx for issue in issues for idx in issue.evidence_runs})
    actions: List[PromptAction] = []
    seen: Set[Tuple[str, Optional[str], Optional[str]]] = set()
    for gap in signal.constraint_gaps:
        content = gap
        marker = "constraint not observed in outputs:"
        if marker in gap.lower():
            content = gap.split(":", 1)[-1].strip() or gap
        key = ("add_global_constraint", None, content)
        if key in seen:
            continue
        seen.add(key)
        actions.append(
            PromptAction(
                type="add_global_constraint",
                content=content,
                evidence_runs=evidence,
            )
        )
    for missing in signal.missing_fields:
        key = ("add_checkpoint_rule", "missing", missing)
        if key in seen:
            continue
        seen.add(key)
        actions.append(
            PromptAction(
                type="add_checkpoint_rule",
                target="missing",
                content=missing,
                evidence_runs=evidence,
            )
        )
    for fmt in signal.format_errors:
        key = ("tighten_schema", None, fmt)
        if key in seen:
            continue
        seen.add(key)
        actions.append(
            PromptAction(
                type="tighten_schema",
                content=fmt,
                evidence_runs=evidence,
            )
        )
    for gap in signal.semantic_gaps:
        key = ("modify_checkpoint_rule", "semantic", gap)
        if key in seen:
            continue
        seen.add(key)
        actions.append(
            PromptAction(
                type="modify_checkpoint_rule",
                target="semantic",
                content=gap,
                evidence_runs=evidence,
            )
        )
    return actions


def _parse_list_literal(text: str) -> List[str]:
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if isinstance(parsed, list):
        return [str(item) for item in parsed if str(item)]
    return []


def _extract_items_from_message(message: str, key: str) -> List[str]:
    marker = f"{key}="
    if marker not in message:
        return []
    tail = message.split(marker, 1)[1]
    start = tail.find("[")
    end = tail.find("]")
    if start == -1 or end == -1 or end <= start:
        return []
    return _parse_list_literal(tail[start: end + 1])


def _extract_constraint_from_message(message: str) -> str:
    if ":" in message:
        return message.split(":", 1)[1].strip()
    return message


def _actions_from_llm_mismatch(issue: DetectedIssue) -> List[PromptAction]:
    actions: List[PromptAction] = []
    missing_items = _extract_items_from_message(issue.message, "missing_items")
    incorrect_items = _extract_items_from_message(issue.message, "incorrect_items")
    fmt_ok = "format_ok=false" in issue.message.lower()
    for item in missing_items:
        actions.append(
            PromptAction(
                type="add_checkpoint_rule",
                target=item,
                content=f"Must include: {item}",
                evidence_runs=issue.evidence_runs,
            )
        )
    for item in incorrect_items:
        actions.append(
            PromptAction(
                type="add_global_constraint",
                content=f"Avoid: {item}",
                evidence_runs=issue.evidence_runs,
            )
        )
    if fmt_ok:
        actions.append(
            PromptAction(
                type="tighten_schema",
                content="Ensure output strictly matches JSON schema.",
                evidence_runs=issue.evidence_runs,
            )
        )
    if not actions:
        actions.append(
            PromptAction(
                type="modify_checkpoint_rule",
                target="mismatch",
                content=issue.message,
                evidence_runs=issue.evidence_runs,
            )
        )
    return actions


def _action_from_constraint_missing(issue: DetectedIssue, _: Sequence[str]) -> List[PromptAction]:
    constraint = _extract_constraint_from_message(issue.message)
    if not constraint:
        return []
    return [
        PromptAction(
            type="add_global_constraint",
            content=constraint,
            evidence_runs=issue.evidence_runs,
        )
    ]


def _action_from_low_similarity(issue: DetectedIssue, _: Sequence[str]) -> List[PromptAction]:
    return [
        PromptAction(
            type="modify_checkpoint_rule",
            target="semantic",
            content=issue.message,
            evidence_runs=issue.evidence_runs,
        )
    ]


def _action_from_llm_mismatch(issue: DetectedIssue, _: Sequence[str]) -> List[PromptAction]:
    return _actions_from_llm_mismatch(issue)


ISSUE_TO_ACTION: Dict[str, Callable[[DetectedIssue, Sequence[str]], List[PromptAction]]] = {
    "constraint_missing": _action_from_constraint_missing,
    "low_similarity": _action_from_low_similarity,
    "llm_output_mismatch": _action_from_llm_mismatch,
}


def _dedupe_actions(actions: Sequence[PromptAction]) -> List[PromptAction]:
    seen: Set[Tuple[str, Optional[str], Optional[str]]] = set()
    result: List[PromptAction] = []
    for action in actions:
        key = (action.type, action.target, action.content)
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def _actions_from_issues(
        issues: Sequence[DetectedIssue],
        constraints: Sequence[str],
) -> List[PromptAction]:
    actions: List[PromptAction] = []
    for issue in issues:
        handler = ISSUE_TO_ACTION.get(issue.kind)
        if handler:
            actions.extend(handler(issue, constraints))
    if not actions and issues:
        fail_signal = _build_fail_signal_from_issues(issues, constraints)
        actions = _actions_from_fail_signal(fail_signal, issues)
    return _normalize_actions(actions, action_budget=None)


def normalize_action(action: PromptAction) -> PromptAction:
    action.type = (action.type or "").strip().lower()
    if action.target is not None:
        action.target = str(action.target).strip()
    if action.content is not None:
        action.content = str(action.content).strip()
    return action


def dedupe_merge_actions(actions: Sequence[PromptAction]) -> List[PromptAction]:
    merged: Dict[Tuple[str, Optional[str]], PromptAction] = {}
    constraint_seen: Set[str] = set()
    for action in actions:
        action = normalize_action(action)
        if action.type == "add_global_constraint":
            if not action.content or action.content in constraint_seen:
                continue
            constraint_seen.add(action.content)
            merged[(action.type, action.content)] = action
            continue
        key = (action.type, action.target)
        if key not in merged:
            merged[key] = action
            continue
        existing = merged[key]
        existing.evidence_runs = sorted(set(existing.evidence_runs + action.evidence_runs))
        if len(action.content or "") > len(existing.content or ""):
            existing.content = action.content
    return list(merged.values())


def limit_actions(actions: Sequence[PromptAction], k: int = DEFAULT_ACTION_BUDGET) -> List[PromptAction]:
    priority = {
        "tighten_schema": 0,
        "add_checkpoint_rule": 1,
        "modify_checkpoint_rule": 1,
        "add_global_constraint": 2,
        "increase_strictness": 3,
    }
    sorted_actions = sorted(actions, key=lambda a: priority.get(a.type, 99))
    return sorted_actions[: max(0, k)]


def clamp_state(state: PromptState) -> PromptState:
    state.strictness_level = max(1, min(state.strictness_level, 5))
    if len(state.global_constraints) > DEFAULT_MAX_CONSTRAINTS:
        state.global_constraints = state.global_constraints[-DEFAULT_MAX_CONSTRAINTS:]
    state.schema = state.schema or ""
    return state


def _schema_strength_score(schema_text: str) -> int:
    schema_obj = _safe_json_from_text(schema_text)
    if not isinstance(schema_obj, dict):
        text = schema_text or ""
        return (
                text.count('"required"') * 2
                + text.count('"enum"') * 2
                + text.count("additionalProperties") * 3
        )
    score = 0
    if schema_obj.get("additionalProperties") is False:
        score += 5
    required = schema_obj.get("required")
    if isinstance(required, list):
        score += len(required) * 3
    properties = schema_obj.get("properties")
    if isinstance(properties, dict):
        score += len(properties)
        for value in properties.values():
            if not isinstance(value, dict):
                continue
            enum_vals = value.get("enum")
            if isinstance(enum_vals, list):
                score += len(enum_vals)
    return score


def _coerce_prompt_action(action: Any) -> Optional[PromptAction]:
    if isinstance(action, PromptAction):
        return action
    if not isinstance(action, dict):
        return None
    action_type = str(action.get("type") or action.get("action") or "").strip()
    if not action_type:
        return None
    target = action.get("target")
    content = action.get("content")
    evidence_runs = action.get("evidence_runs") or []
    if isinstance(evidence_runs, list):
        evidence_runs = [int(x) for x in evidence_runs if str(x).isdigit()]
    else:
        evidence_runs = []
    return PromptAction(
        type=action_type,
        target=str(target).strip() if target is not None else None,
        content=str(content).strip() if content is not None else None,
        evidence_runs=evidence_runs,
    )


def _normalize_actions(
        actions: Sequence[Any],
        state: Optional[PromptState] = None,
        *,
        action_budget: Optional[int] = DEFAULT_ACTION_BUDGET,
) -> List[PromptAction]:
    merged: Dict[Tuple[str, Optional[str]], PromptAction] = {}
    schema_scores: Dict[Tuple[str, Optional[str]], int] = {}
    for raw in actions:
        action = _coerce_prompt_action(raw)
        if not action:
            continue
        action_type = action.type.strip().lower()
        action.type = action_type
        if action_type == "add_global_constraint":
            if not action.content:
                continue
            if not action.target:
                action.target = action.content
        if action_type in {"add_checkpoint_rule", "modify_checkpoint_rule"} and not action.target:
            action.target = "generic"
        key = (action_type, action.target)
        if action_type == "tighten_schema" and not action.content:
            continue
        if key not in merged:
            merged[key] = action
            if action_type == "tighten_schema":
                schema_scores[key] = _schema_strength_score(action.content or "")
            continue
        existing = merged[key]
        if action_type in {"add_checkpoint_rule", "modify_checkpoint_rule"}:
            if len(action.evidence_runs) >= len(existing.evidence_runs):
                merged[key] = action
        elif action_type == "tighten_schema":
            new_score = _schema_strength_score(action.content or "")
            if new_score >= schema_scores.get(key, 0):
                merged[key] = action
                schema_scores[key] = new_score
        elif action_type == "add_global_constraint":
            continue
        elif action_type == "increase_strictness":
            if len(action.evidence_runs) >= len(existing.evidence_runs):
                merged[key] = action
    normalized = dedupe_merge_actions(list(merged.values()))
    if action_budget is not None and action_budget > 0 and len(normalized) > action_budget:
        normalized.sort(
            key=lambda item: (-len(item.evidence_runs), item.type, item.target or "", item.content or ""),
        )
        normalized = normalized[:action_budget]
    return normalized


def apply_actions(
        state: PromptState,
        actions: Sequence[PromptAction],
        *,
        action_budget: Optional[int] = DEFAULT_ACTION_BUDGET,
        max_constraints: int = DEFAULT_MAX_CONSTRAINTS,
        schema_tighten_limit: int = DEFAULT_SCHEMA_TIGHTEN_LIMIT,
        max_strictness: int = 5,
        normalize: bool = True,
) -> PromptState:
    """Public action executor for prompt state updates."""
    normalized = _normalize_actions(actions, state, action_budget=action_budget) if normalize else list(actions)
    return _apply_prompt_actions(
        state,
        normalized,
        action_budget=None,
        max_constraints=max_constraints,
        schema_tighten_limit=schema_tighten_limit,
        max_strictness=max_strictness,
        normalize=False,
    )


class ConcurrencyManager:
    def __init__(self, max_concurrency: int) -> None:
        self.limit = max(1, int(max_concurrency))
        self.sem = threading.Semaphore(self.limit)

    @contextmanager
    def slot(self):
        self.sem.acquire()
        try:
            yield
        finally:
            self.sem.release()

    def set_limit(self, n: int) -> None:
        self.limit = max(1, int(n))
        self.sem = threading.Semaphore(self.limit)


class SqliteCache:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value BLOB)")

    def get(self, key: str) -> Any:
        cursor = self.conn.execute("SELECT value FROM cache WHERE key=?", (key,))
        row = cursor.fetchone()
        if not row:
            return None
        try:
            return pickle.loads(row[0])
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        blob = pickle.dumps(value)
        self.conn.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, blob),
        )
        self.conn.commit()


def safe_run_once(run_fn, prompt_text: str, retries: int = 3):
    for attempt in range(retries):
        try:
            return run_fn(prompt_text)
        except TimeoutError:
            time.sleep((2 ** attempt) + random.random())
        except Exception as exc:
            if "JSON" in str(exc):
                raise
            time.sleep(1)
    raise RuntimeError("run_once failed after retries")


def generate_variants(state: PromptState) -> List[PromptState]:
    variants: List[PromptState] = []
    base = copy.deepcopy(state)
    variants.append(base)
    up = copy.deepcopy(state)
    up.strictness_level = min(5, state.strictness_level + 1)
    variants.append(up)
    down = copy.deepcopy(state)
    down.strictness_level = max(1, state.strictness_level - 1)
    variants.append(down)
    return variants


def _output_to_sample(output: Any) -> ExecutionSample:
    if isinstance(output, ExecutionSample):
        return output
    if isinstance(output, dict):
        status = _status_from_run(output)
        output_text = _extract_output_text(output)
        metrics: Dict[str, Any] = {}
        if isinstance(output.get("metrics"), dict):
            metrics = output.get("metrics") or {}
        return ExecutionSample(index=1, status=status, output_text=output_text, metrics=metrics, raw=output)
    return ExecutionSample(index=1, status="success", output_text=str(output), metrics={}, raw={"output": output})


def _analyze_single_output(
        output: Any,
        reference: ReferenceSpec,
        llm_config: Optional[Dict[str, Any]] = None,
        reference_texts: Optional[Sequence[Optional[str]]] = None,
) -> Tuple[List[DetectedIssue], List[PromptAction]]:
    sample = _output_to_sample(output)
    detector = PromptDeltaDetector(reference)
    issues = detector.detect([sample])
    judge_refs = reference_texts
    if not judge_refs and reference.expected_outputs:
        judge_refs = [reference.expected_outputs[0]]
    llm_issues, _ = _llm_output_judge(
        [sample],
        judge_refs,
        llm_config,
        constraints=reference.constraints,
        expected_outputs=reference.expected_outputs,
    )
    if llm_issues:
        issues.extend(llm_issues)
    actions = _actions_from_issues(issues, reference.constraints)
    return issues, actions


def prompt_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def score_report(
        report: Optional[Any] = None,
        *,
        issues: Optional[Sequence[DetectedIssue]] = None,
        actions: Optional[Sequence[PromptAction]] = None,
) -> float:
    if report is not None:
        issues = getattr(report, "issues", None)
        actions = getattr(report, "actions", None)
    issue_count = len(issues or [])
    action_count = len(actions or [])
    return 100.0 - 10.0 * issue_count - 2.0 * action_count


def _issue_is_format_related(issue: DetectedIssue) -> bool:
    kind = (issue.kind or "").lower()
    message = (issue.message or "").lower()
    if "format" in message or "json" in message or "schema" in message:
        return True
    if "format_ok" in message:
        return True
    if kind in {"llm_output_mismatch"} and "format_ok" in message:
        return True
    return False


def _filter_actions_for_stage(stage: OptimizationStage, actions: Sequence[PromptAction]) -> List[PromptAction]:
    if stage == OptimizationStage.FORMAT:
        allowed = {"tighten_schema", "add_global_constraint"}
        return [action for action in actions if action.type in allowed]
    if stage == OptimizationStage.RECALL:
        return [action for action in actions if action.type != "increase_strictness"]
    return list(actions)


def _serialize_issues(issues: Sequence[DetectedIssue]) -> List[Dict[str, Any]]:
    return [
        {
            "kind": issue.kind,
            "severity": issue.severity,
            "message": issue.message,
            "evidence_runs": issue.evidence_runs,
        }
        for issue in issues
    ]


def _serialize_actions(actions: Sequence[PromptAction]) -> List[Dict[str, Any]]:
    return [
        {
            "type": action.type,
            "target": action.target,
            "content": action.content,
            "evidence_runs": action.evidence_runs,
        }
        for action in actions
    ]


def _prompt_state_from_dict(payload: Dict[str, Any]) -> PromptState:
    allowed = {field_def.name for field_def in fields(PromptState)}
    filtered = {key: value for key, value in payload.items() if key in allowed}
    return PromptState(**filtered)


def save_checkpoint(
        run_dir: str | Path,
        step: int,
        state: PromptState,
        report: Dict[str, Any],
        prompt_text: str,
) -> None:
    root = Path(run_dir)
    step_dir = root / f"{step:03d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / "state.json").write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (step_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (step_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")
    latest_dir = root / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    (latest_dir / "state.json").write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (latest_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (latest_dir / "prompt.txt").write_text(prompt_text, encoding="utf-8")


def _load_latest_checkpoint(run_dir: str | Path) -> Optional[PromptState]:
    latest_state = Path(run_dir) / "latest" / "state.json"
    if not latest_state.exists():
        return None
    try:
        payload = json.loads(latest_state.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return _prompt_state_from_dict(payload)
    except Exception:
        return None
    return None


def optimize_prompt(
        initial_state: PromptState,
        run_once_fn,
        reference: ReferenceSpec,
        *,
        llm_config: Optional[Dict[str, Any]] = None,
        reference_texts: Optional[Sequence[Optional[str]]] = None,
        max_steps: int = 5,
        action_budget: Optional[int] = DEFAULT_ACTION_BUDGET,
        max_constraints: int = DEFAULT_MAX_CONSTRAINTS,
        schema_tighten_limit: int = DEFAULT_SCHEMA_TIGHTEN_LIMIT,
        max_strictness: int = 5,
        max_prompt_growth: float = DEFAULT_MAX_PROMPT_GROWTH,
        max_no_improve_rounds: int = DEFAULT_NO_IMPROVE_ROUNDS,
        checkpoint_dir: Optional[str | Path] = None,
        resume_from_checkpoint: bool = False,
        concurrency_manager: Optional[ConcurrencyManager] = None,
        max_concurrency: Optional[int] = None,
        variant_count: int = 1,
        variants_fn: Optional[Callable[[PromptState], List[PromptState]]] = None,
        beam_width: int = 1,
        cache: Optional[Dict[Tuple[str, str], Any]] = None,
        input_hash: str = "",
        run_dir: Optional[str | Path] = None,
        reference_hash: str = "",
        cache_path: Optional[str | Path] = None,
        retries: int = 3,
) -> PromptState:
    state = initial_state
    if run_dir and not checkpoint_dir:
        checkpoint_dir = run_dir
    if checkpoint_dir and resume_from_checkpoint:
        resumed = _load_latest_checkpoint(checkpoint_dir)
        if resumed:
            state = resumed
    cm = concurrency_manager
    if not cm and max_concurrency:
        cm = ConcurrencyManager(max_concurrency)
    cache_store = cache
    cache_backend = None
    if cache_path:
        cache_backend = SqliteCache(cache_path)
    elif checkpoint_dir:
        cache_backend = SqliteCache(Path(checkpoint_dir) / "cache.sqlite")
    cache_lock = threading.Lock()
    best_state = copy.deepcopy(state)
    best_score = float("-inf")
    no_improve_rounds = 0
    seen_hashes: Set[str] = set()
    initial_prompt = _render_prompt_state(state)
    initial_length = max(1, len(initial_prompt))
    action_budget_value = action_budget if action_budget is not None else DEFAULT_ACTION_BUDGET
    stage = OptimizationStage.FORMAT
    stage_plateau = 0
    beam_width = max(1, int(beam_width))
    beam: List[Tuple[float, PromptState]] = [(0.0, state)]
    run_root = Path(checkpoint_dir) if checkpoint_dir else None
    timeline_path = None
    if run_root:
        run_root.mkdir(parents=True, exist_ok=True)
        timeline_path = run_root / "timeline.csv"
        if not timeline_path.exists():
            timeline_path.write_text(
                "step,stage,score,issue_count,action_count,concurrency\n",
                encoding="utf-8",
            )

    def _run_once(prompt_text: str):
        key = (prompt_hash(prompt_text), input_hash or reference_hash)
        if cache_backend is not None:
            with cache_lock:
                cached = cache_backend.get(f"{key[0]}:{key[1]}")
            if cached is not None:
                return cached
        if cache_store is not None:
            with cache_lock:
                if key in cache_store:
                    return cache_store[key]

        def _call(text: str):
            with (cm.slot() if cm else nullcontext()):
                return run_once_fn(text)

        output = safe_run_once(_call, prompt_text, retries=retries)
        if cache_backend is not None:
            with cache_lock:
                cache_backend.set(f"{key[0]}:{key[1]}", output)
        if cache_store is not None:
            with cache_lock:
                cache_store[key] = output
        return output

    def _evaluate_state(candidate: PromptState):
        prompt_text = _render_prompt_state(candidate)
        output = _run_once(prompt_text)
        issues, actions = _analyze_single_output(
            output,
            reference,
            llm_config=llm_config,
            reference_texts=reference_texts,
        )
        score = score_report(issues=issues, actions=actions)
        return candidate, prompt_text, issues, actions, score

    stop_state: Optional[PromptState] = None
    last_step = -1
    for step in range(max_steps):
        last_step = step
        json_repair_needed = False
        candidates: List[PromptState] = []
        for _, base_state in beam:
            if variants_fn:
                variants = variants_fn(base_state)
            else:
                variants = generate_variants(base_state)
            if variant_count > 0:
                variants = variants[: max(1, variant_count)]
            if not variants:
                variants = [base_state]
            candidates.extend(variants)
        if not candidates:
            candidates = [state]
        results: List[Tuple[PromptState, str, List[DetectedIssue], List[PromptAction]]] = []
        if len(candidates) <= 1:
            cand, prompt_text, issues, actions, _ = _evaluate_state(candidates[0])
            results.append((cand, prompt_text, issues, actions))
        else:
            max_workers = len(candidates)
            if max_concurrency:
                max_workers = min(max_workers, int(max_concurrency))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_evaluate_state, cand) for cand in candidates]
                for future in as_completed(futures):
                    try:
                        cand, prompt_text, issues, actions, _ = future.result()
                        results.append((cand, prompt_text, issues, actions))
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Variant evaluation failed", extra={"error": str(exc)})
                        if "JSON" in str(exc):
                            json_repair_needed = True
        if not results:
            if json_repair_needed:
                repair_action = PromptAction(
                    type="add_global_constraint",
                    content="你必须输出严格合法的 JSON，不得输出任何额外文本",
                )
                state = apply_actions(
                    state,
                    [repair_action],
                    action_budget=None,
                    max_constraints=max_constraints,
                    schema_tighten_limit=schema_tighten_limit,
                    max_strictness=max_strictness,
                    normalize=False,
                )
                state = clamp_state(state)
                beam = [(0.0, state)]
                continue
            break
        results_with_scores: List[Tuple[float, PromptState, str, List[DetectedIssue], List[PromptAction]]] = []
        for cand, prompt_text, issues, actions in results:
            normalized_actions = [normalize_action(action) for action in actions]
            stage_actions = _filter_actions_for_stage(stage, normalized_actions)
            merged_actions = dedupe_merge_actions(stage_actions)
            limited_actions = limit_actions(merged_actions, k=action_budget_value)
            score = score_report(issues=issues, actions=limited_actions)
            results_with_scores.append((score, cand, prompt_text, issues, limited_actions))
        results_with_scores.sort(key=lambda item: item[0], reverse=True)
        score, candidate, prompt, issues, limited_actions = results_with_scores[0]
        has_format_issue = any(_issue_is_format_related(issue) for issue in issues)
        if stage == OptimizationStage.FORMAT:
            if not has_format_issue:
                stage = OptimizationStage.RECALL
                stage_plateau = 0
                logger.info("enter stage: RECALL")
        elif stage == OptimizationStage.RECALL:
            if not has_format_issue:
                stage_plateau += 1
                if stage_plateau >= 2:
                    stage = OptimizationStage.STRICT
                    stage_plateau = 0
                    logger.info("enter stage: STRICT")
            else:
                stage_plateau = 0
        prompt_key = prompt_hash(prompt)
        if prompt_key in seen_hashes:
            logger.warning("Prompt optimization detected loop, rolling back to best state")
            stop_state = best_state
            break
        seen_hashes.add(prompt_key)
        if len(prompt) > int(initial_length * max_prompt_growth):
            logger.warning("Prompt exceeded growth budget, rolling back to best state")
            stop_state = best_state
            break
        if score > best_score:
            best_score = score
            best_state = copy.deepcopy(candidate)
            no_improve_rounds = 0
        else:
            no_improve_rounds += 1
        if not issues:
            break
        if not limited_actions:
            break
        if no_improve_rounds >= max_no_improve_rounds:
            logger.warning("Prompt optimization plateaued, rolling back to best state")
            stop_state = best_state
            break
        error_rate = min(1.0, len(issues) / max(1, len(reference.expected_outputs) or 1))
        report_payload = {
            "step": step,
            "stage": stage.name,
            "score": score,
            "error_rate": error_rate,
            "issues": _serialize_issues(issues),
            "actions": _serialize_actions(limited_actions),
        }
        state = apply_actions(
            candidate,
            limited_actions,
            action_budget=None,
            max_constraints=max_constraints,
            schema_tighten_limit=schema_tighten_limit,
            max_strictness=max_strictness,
            normalize=False,
        )
        state = clamp_state(state)
        beam = [(score, state)]
        if beam_width > 1:
            beam_candidates: List[Tuple[float, PromptState]] = []
            for cand_score, cand_state, _, cand_issues, cand_actions in results_with_scores:
                next_state = cand_state
                if cand_actions:
                    next_state = apply_actions(
                        copy.deepcopy(cand_state),
                        cand_actions,
                        action_budget=None,
                        max_constraints=max_constraints,
                        schema_tighten_limit=schema_tighten_limit,
                        max_strictness=max_strictness,
                        normalize=False,
                    )
                    next_state = clamp_state(next_state)
                beam_candidates.append((cand_score, next_state))
            beam_candidates.sort(key=lambda item: item[0], reverse=True)
            beam = beam_candidates[:beam_width]
        if checkpoint_dir:
            save_checkpoint(checkpoint_dir, step, state, report_payload, prompt)
        if timeline_path:
            concurrency_value = getattr(cm, "limit", 1) if cm else 1
            with timeline_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{step},{stage.name},{score},{len(issues)},{len(limited_actions)},{concurrency_value}\n"
                )
        if cm and hasattr(cm, "set_limit"):
            if error_rate > 0.3 and cm.limit > 1:
                cm.set_limit(cm.limit - 1)
                logger.warning("high error rate, reduce concurrency to %d", cm.limit)
            elif error_rate < 0.05 and max_concurrency:
                cm.set_limit(min(cm.limit + 1, int(max_concurrency)))
    if stop_state:
        state = stop_state
    if run_root:
        summary_path = run_root / "summary.md"
        concurrency_value = getattr(cm, "limit", 1) if cm else 1
        total_steps = last_step + 1 if last_step >= 0 else 0
        summary_path.write_text(
            "\n".join(
                [
                    "# Optimization Summary",
                    "",
                    f"- Final Stage: {stage.name}",
                    f"- Best Score: {best_score}",
                    f"- Total Steps: {total_steps}",
                    f"- Final Concurrency: {concurrency_value}",
                ]
            ),
            encoding="utf-8",
        )
    return state


def _parse_action_judge_output(raw: str) -> Tuple[str, List[Dict[str, Any]]]:
    verdict = ""
    actions: List[Dict[str, Any]] = []
    parsed = _safe_json_from_text(raw)
    if isinstance(parsed, dict):
        verdict = str(parsed.get("verdict") or "").lower()
        action_items = parsed.get("actions") or []
        if isinstance(action_items, list):
            actions = [item for item in action_items if isinstance(item, dict)]
    if not verdict:
        lower = raw.lower()
        if "pass" in lower:
            verdict = "pass"
        elif "fail" in lower:
            verdict = "fail"
    return verdict or "fail", actions


def _safe_rewrite_prompt(current: str, rewrite_fn) -> str:
    """
    Run rewrite_fn with placeholder masking. If placeholders are dropped/altered, fall back to original.
    rewrite_fn receives the masked prompt and must return a string (masked or restored).
    """
    placeholders = _extract_placeholders(current)
    if not placeholders:
        try:
            return rewrite_fn(current) or current
        except Exception:
            return current

    masked, token_map = _mask_placeholders(current, placeholders)
    try:
        rewritten = rewrite_fn(masked) or masked
    except Exception as exc:  # noqa: BLE001
        try:
            logger.warning("Prompt rewrite failed, keep original", extra={"error": str(exc)})
        except Exception:
            pass
        return current

    if not _tokens_intact(masked, rewritten, token_map):
        try:
            logger.warning(
                "Prompt rewrite dropped/duplicated placeholders, keep original",
                extra={"original": current[:200], "rewritten": rewritten[:200]},
            )
        except Exception:
            pass
        return current

    restored = _restore_placeholders(rewritten, token_map)
    if not _placeholders_preserved(placeholders, restored):
        try:
            logger.warning(
                "Prompt rewrite altered placeholder set, keep original",
                extra={"original": current[:200], "restored": restored[:200]},
            )
        except Exception:
            pass
        return current
    return restored


def _is_reference_soft_satisfied(reference: str, candidate: str, threshold: float = REF_SOFT_SIMILARITY_THRESHOLD) -> Tuple[bool, float]:
    """
    A relaxed check: if candidate roughly matches the reference text (semantics/length-wise),
    we consider it "good enough" even when the judge returns uncertain/empty verdicts.
    """
    if not reference or not candidate:
        return (False, 0.0)
    score = _similarity(reference, candidate)
    return (score >= threshold, score)


def _coerce_text(value: Any) -> str:
    """Convert arbitrary values (dict/list/objects) to a printable text string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
    except Exception:
        pass
    try:
        return str(value)
    except Exception:
        return ""


def _extract_lm_response_from_exception(exc: Exception) -> Optional[str]:
    """
    Try to recover the raw LM response from DSPy AdapterParseError text so that we can fall back
    to heuristic processing instead of failing the entire optimization run.
    """
    # dspy.utils.exceptions.AdapterParseError exposes .lm_response in recent versions
    response = getattr(exc, "lm_response", None)
    if response:
        return _coerce_text(response)
    text = str(exc)
    marker = "LM Response:"
    if marker not in text:
        return None
    remainder = text.split(marker, 1)[1]
    for stop in ("\nExpected to find", "\nActual output", "\nDuring handling"):
        idx = remainder.find(stop)
        if idx != -1:
            remainder = remainder[:idx]
            break
    extracted = remainder.strip()
    return extracted or None


def _extract_llm_text_response(data: Dict[str, Any]) -> str:
    """
    Best-effort extraction of text content from an LLM response.
    Supports OpenAI-style choices[0].message.content in addition to text/content/output.
    """
    if not data:
        return ""
    for key in ("text", "content", "output"):
        if key in data and data[key]:
            return str(data[key])
    # OpenAI-style chat/completion
    try:
        choices = data.get("choices") or []
        if choices and isinstance(choices, list):
            first = choices[0] or {}
            message = first.get("message") or {}
            if message.get("content"):
                return str(message["content"])
            if first.get("text"):
                return str(first["text"])
    except Exception:
        pass
    return ""


class _LlmRewriter:
    """Very small wrapper to rewrite prompt text with an external LLM."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.url = cfg.get("url")
        self.model = cfg.get("model")
        self.api_key = cfg.get("api_key")

    def rewrite(
        self,
        current: str,
        issues: Sequence[DetectedIssue],
        constraints: Sequence[str],
        context: Optional[str] = None,
    ) -> str:
        import requests

        placeholders = _extract_placeholders(current)

        def _do_rewrite(masked_prompt: str) -> str:
            prompt_lines = [
                "You are a prompt editor. Improve the prompt to address the following issues:",
            ]
            for it in issues:
                prompt_lines.append(f"- {it.kind}: {it.message}")
            if constraints:
                prompt_lines.append("Ensure these constraints are included:")
                for c in constraints:
                    prompt_lines.append(f"- {c}")
            if placeholders:
                prompt_lines.append("Do NOT change or drop these placeholders/variables:")
                for ph in placeholders:
                    prompt_lines.append(f"- {ph}")
            if context:
                prompt_lines.append("Context:")
                prompt_lines.append(context)
            prompt_lines.append("Original prompt:")
            prompt_lines.append(masked_prompt)
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a prompt editor. Improve the prompt."},
                    {"role": "user", "content": "\n".join(prompt_lines)},
                ],
                "stream": False,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            try:
                logger.info(
                    "调用 LLM 重写提示词",
                    extra={"model": self.model, "url": self.url, "has_api_key": bool(self.api_key)},
                )
            except Exception:
                pass
            resp = requests.post(self.url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return _extract_llm_text_response(data) or masked_prompt

        return _safe_rewrite_prompt(current, _do_rewrite)


class _LlmPromptJudge:
    """Use LLM to decide whether a prompt needs modification."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.url = cfg.get("url")
        self.model = cfg.get("model")
        self.api_key = cfg.get("api_key")

    def needs_change(self, prompt_text: str, issues: Sequence[DetectedIssue], constraints: Sequence[str]) -> bool:
        if not self.url or not self.model:
            # default: if we have detected issues, assume change is needed
            return bool(issues)
        import requests

        guidance = [
            "You are a prompt reviewer. Answer with 'yes' or 'no' only.",
            "Determine if this prompt likely needs improvement based on the following issues and constraints.",
            "Issues:",
        ]
        if issues:
            for it in issues:
                guidance.append(f"- {it.kind}: {it.message}")
        else:
            guidance.append("- none observed")
        if constraints:
            guidance.append("Constraints:")
            for c in constraints:
                guidance.append(f"- {c}")
        guidance.append("Prompt:")
        guidance.append(prompt_text)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a strict prompt reviewer. Reply with yes or no only."},
                {"role": "user", "content": "\n".join(guidance)},
            ],
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            resp = requests.post(self.url, json=payload, headers=headers, timeout=20)
            resp.raise_for_status()
            data = resp.json()
            answer = str(_extract_llm_text_response(data)).lower()
            if "yes" in answer and "no" not in answer:
                return True
            if "no" in answer and "yes" not in answer:
                return False
        except Exception:
            return bool(issues)
        return bool(issues)


class _DspyRewriter:
    """Optional dspy-based rewriter; falls back if dspy is unavailable."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        try:
            import dspy  # type: ignore
            from dspy.adapters import ChatAdapter  # 根据你安装的版本稍微调整

            self._dspy = dspy
        except Exception:
            self._dspy = None
        self.cfg = cfg
        self.available = self._dspy is not None

    def rewrite(
        self,
        current: str,
        issues: Sequence[DetectedIssue],
        constraints: Sequence[str],
        context: Optional[str] = None,
    ) -> str:
        if not self.available:
            return current

        placeholders = _extract_placeholders(current)

        def _do_rewrite(masked_prompt: str) -> str:
            # Minimal dspy-based rewrite: use LM to propose a revised prompt
            lm = self._dspy.LM(  # type: ignore[attr-defined]
                model=self.cfg.get("model"),
                api_base=self.cfg.get("url"),
                api_key=self.cfg.get("api_key"),
            )
            class RewriteSpec(self._dspy.Signature):  # type: ignore[attr-defined]
                """Rewrite a prompt to address issues and constraints."""

                issues: str
                constraints: str
                context: str
                placeholders: str
                original_prompt: str
                improved_prompt: str

            chain = self._dspy.ChainOfThought(RewriteSpec, lm=lm)  # type: ignore[attr-defined]
            issues_text = "; ".join(f"{it.kind}: {it.message}" for it in issues) or "none"
            constraints_text = "; ".join(constraints) or "none"
            placeholders_text = "; ".join(placeholders) or "none"
            res = chain(
                issues=issues_text,
                constraints=constraints_text,
                context=context or "none",
                placeholders=placeholders_text,
                original_prompt=masked_prompt,
            )
            return getattr(res, "improved_prompt", masked_prompt) or masked_prompt

        return _safe_rewrite_prompt(current, _do_rewrite)


class _DspyPromptOptimizer:
    """DSPy-based prompt optimizer that loops Predictor -> Judge -> Prompt update."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.available = False
        self.max_iterations = int(cfg.get("dspy_optimizer_iterations", 2))
        self.max_samples = int(cfg.get("dspy_optimizer_max_samples", 4))
        self.max_tokens = int(cfg.get("dspy_optimizer_max_tokens", DEFAULT_DSPY_MAX_TOKENS))
        self._optimizer_cm = None
        self._lm = None
        url = cfg.get("url")
        model = cfg.get("model")
        if not url or not model:
            try:
                logger.info(
                    "DSPy 优化器缺少 LLM url 或 model，无法启用",
                    extra={"workflow_llm_url": url, "workflow_llm_model": model},
                )
            except Exception:
                pass
            return
        try:
            import dspy  # type: ignore

            self._dspy = dspy
        except Exception:
            self._dspy = None
            try:
                logger.warning(
                    "DSPy 库未安装或导入失败，无法启用优化器",
                    extra={"workflow_llm_model": model},
                )
            except Exception:
                pass
            return
        self.available = self._dspy is not None
        if not self.available:
            return
        try:
            self._lm = self._build_lm()
            # 配置为全局 LM，满足 DSPy Predict/_forward_preprocess 的要求
            # self._dspy.configure(lm=self._lm, adapter=ChatAdapter())  # type: ignore[attr-defined]
            # self._dspy.settings.configure(lm=self._lm, adapter=ChatAdapter(),)
            self._dspy.configure(lm=self._lm)  # type: ignore[attr-defined]
            logger.info(
                "DSPy LM 已配置",
                extra={
                    "workflow_llm_model": model,
                    "api_base": url,
                    "max_tokens": self.max_tokens,
                },
            )
        except Exception as exc:  # noqa: BLE001
            try:
                logger.warning(
                    "DSPy LM 创建或配置失败，将无法进行提示词优化",
                    extra={"error": str(exc), "workflow_llm_model": model},
                )
            except Exception:
                pass
            self.available = False
            return
        try:
            logger.info(
                "DSPy 优化器已启用",
                extra={
                    "workflow_llm_url": url,
                    "workflow_llm_model": model,
                    "iterations": self.max_iterations,
                    "max_samples": self.max_samples,
                },
            )
        except Exception:
            pass

    def optimize_prompts(
        self,
        workflow_id: str,
        prompts: Sequence[PromptLocation],
        samples: Sequence[ExecutionSample],
        reference_texts: Sequence[Optional[str]],
        constraints: Sequence[str],
    ) -> List[PromptPatch]:
        if not self.available or not prompts:
            return []
        optimizer_cm = _get_optimizer_concurrency_manager()

        dataset = self._build_dataset(samples, reference_texts)
        if not dataset:
            return []
        patches: List[PromptPatch] = []
        max_workers = self._effective_workers(len(prompts))
        try:
            logger.info(
                "DSPy optimizer concurrency",
                extra={
                    "mode": "prompts",
                    "total": len(prompts),
                    "max_workers": max_workers,
                    "optimizer_limit": getattr(optimizer_cm, "limit", None) if optimizer_cm else None,
                    "fallback_used": optimizer_cm is None,
                },
            )
        except Exception:
            pass

        def _optimize_prompt(prompt: PromptLocation) -> Optional[PromptPatch]:
            # Avoid sharing ChainOfThought across threads
            predictor = self._dspy.ChainOfThought(self._predict_signature())  # type: ignore[attr-defined]
            judge = self._dspy.ChainOfThought(self._judge_signature())  # type: ignore[attr-defined]
            rewriter = self._dspy.ChainOfThought(self._rewrite_signature())  # type: ignore[attr-defined]
            action_judge = None
            if self.cfg.get("dspy_action_judge", False):
                action_judge = self._dspy.ChainOfThought(self._action_judge_signature())  # type: ignore[attr-defined]

            improved = self._optimize_single(
                prompt.text,
                dataset,
                constraints,
                predictor,
                judge,
                rewriter,
                optimizer_cm=optimizer_cm,
                action_judge=action_judge,
            )

            if not improved or improved.strip() == prompt.text.strip():
                return None
            try:
                logger.info(
                    "DSPy 优化完成 prompt",
                    extra={
                        "workflow_id": workflow_id,
                        "node_id": prompt.node_id,
                        "field_path": prompt.path,
                    },
                )
            except Exception:
                pass
            return PromptPatch(
                workflow_id=workflow_id,
                node_id=prompt.node_id,
                field_path=prompt.path,
                old=prompt.text,
                new=improved,
                rationale="DSPy optimizer updated prompt based on reference feedback",
                confidence=0.75,
                evidence_runs=[record["run_index"] for record in dataset],
            )

        if max_workers <= 1:
            for prompt in prompts:
                patch = _optimize_prompt(prompt)

                if patch:
                    patches.append(patch)
            return patches

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_optimize_prompt, prompt): prompt for prompt in prompts}

            for future in as_completed(future_map):
                patch = future.result()
                if patch:
                    patches.append(patch)

        prompt_order = {p.path: idx for idx, p in enumerate(prompts)}
        patches.sort(key=lambda p: prompt_order.get(p.field_path, 0))

    def optimize_blocks(
            self,
            workflow_id: str,
            blocks: Sequence[PromptBlock],
            samples: Sequence[ExecutionSample],
            reference_texts: Sequence[Optional[str]],
            constraints: Sequence[str],
            fail_signal: Optional[FailSignal] = None,
    ) -> List[PromptPatch]:
        if not self.available or not blocks:
            return []
        optimizer_cm = _get_optimizer_concurrency_manager()
        dataset = self._build_dataset(samples, reference_texts)
        if not dataset:
            return []
        max_workers = self._effective_workers(len(blocks))
        patches: List[PromptPatch] = []
        effective_signal = fail_signal or FailSignal()
        try:
            logger.info(
                "DSPy optimizer concurrency",
                extra={
                    "mode": "blocks",
                    "total": len(blocks),
                    "max_workers": max_workers,
                    "optimizer_limit": getattr(optimizer_cm, "limit", None) if optimizer_cm else None,
                    "fallback_used": optimizer_cm is None,
                },
            )
        except Exception:
            pass
        block_order = {block.node_id: idx for idx, block in enumerate(blocks)}
        message_order: Dict[Tuple[str, str], int] = {}
        for block in blocks:
            for idx, msg in enumerate(block.messages):
                key = (block.node_id, str(msg.get("path") or ""))
                if key not in message_order:
                    message_order[key] = idx

        def _optimize_block(block: PromptBlock) -> List[PromptPatch]:
            predictor = self._dspy.ChainOfThought(self._predict_signature())  # type: ignore[attr-defined]
            judge = self._dspy.ChainOfThought(self._judge_signature())  # type: ignore[attr-defined]
            rewriter = self._dspy.ChainOfThought(self._rewrite_signature())  # type: ignore[attr-defined]
            action_judge = None
            if self.cfg.get("dspy_action_judge", False):
                action_judge = self._dspy.ChainOfThought(self._action_judge_signature())  # type: ignore[attr-defined]

            context_text = _build_block_context(block, effective_signal, constraints)
            placeholders_text = (
                "; ".join(block.brief.placeholders) if block.brief and block.brief.placeholders else ""
            )
            improved = self._optimize_single(
                block.merged_text,
                dataset,
                constraints,
                predictor,
                judge,
                rewriter,
                optimizer_cm=optimizer_cm,
                context_text=context_text,
                placeholders_text=placeholders_text,
                action_judge=action_judge,
            )
            if not improved or improved.strip() == block.merged_text.strip():
                return []
            if not _validate_block_rewrite(block, improved):
                try:
                    logger.warning(
                        "DSPy block rewrite校验失败，保持原提示词",
                        extra={"workflow_id": workflow_id, "node_id": block.node_id},
                    )
                except Exception:
                    pass
                return []
            split_texts = _split_block_by_markers(block, improved)
            if not split_texts:
                return []
            block_patches: List[PromptPatch] = []
            for msg, new_text in zip(block.messages, split_texts):
                if new_text is None:
                    continue
                old_text = str(msg.get("text") or "")
                if new_text.strip() == old_text.strip():
                    continue
                block_patches.append(
                    PromptPatch(
                        workflow_id=workflow_id,
                        node_id=block.node_id,
                        field_path=msg.get("path", ""),
                        old=old_text,
                        new=new_text,
                        rationale="DSPy optimizer updated prompt block with node context",
                        confidence=0.75,
                        evidence_runs=[record["run_index"] for record in dataset],
                    )
                )
            return block_patches

        if max_workers <= 1:
            for block in blocks:
                patches.extend(_optimize_block(block))
            patches.sort(
                key=lambda p: (
                    block_order.get(p.node_id, 0),
                    message_order.get((p.node_id, p.field_path), 0),
                )
            )
            return patches

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_optimize_block, block): block for block in blocks}

            for future in as_completed(future_map):
                block_patches = future.result()
                if block_patches:
                    patches.extend(block_patches)
        patches.sort(
            key=lambda p: (
                block_order.get(p.node_id, 0),
                message_order.get((p.node_id, p.field_path), 0),
            )
        )
        return patches

    def _build_dataset(
        self,
        samples: Sequence[ExecutionSample],
        reference_texts: Sequence[Optional[str]],
    ) -> List[Dict[str, Any]]:
        dataset: List[Dict[str, Any]] = []
        for idx, sample in enumerate(samples):
            ref = ""
            if reference_texts and idx < len(reference_texts):
                ref = reference_texts[idx] or ""
            if not ref:
                ref = sample.output_text or ""
            if not ref:
                continue
            dataset.append(
                {
                    "workflow_input": _extract_workflow_input(sample.raw),
                    "reference": ref,
                    "run_index": sample.index,
                }
            )
            if len(dataset) >= self.max_samples:
                break
        try:
            logger.debug(
                "DSPy 构建训练样本",
                extra={"sample_count": len(dataset), "max_samples": self.max_samples},
            )
        except Exception:
            pass
        return dataset

    def _effective_workers(self, total: int) -> int:
        """
        Determine safe worker count based on runtime concurrency manager.

        Falls back to total items when no manager is configured.
        """
        if total <= 1:
            return 1
        cm = _get_optimizer_concurrency_manager()
        if cm and getattr(cm, "limit", None):
            try:
                return max(1, min(int(cm.limit), int(total)))
            except Exception:
                return max(1, min(4, total))
        # Fallback cap to avoid QPS spikes when no manager is configured
        return max(1, min(4, total))

    def _optimize_single(
        self,
        current_prompt: str,
        dataset: Sequence[Dict[str, Any]],
        constraints: Sequence[str],
        predictor,
        judge,
        rewriter,
            *,
        optimizer_cm=None,
            context_text: str = "",
            placeholders_text: str = "",
            action_judge=None,
    ) -> str:
        cm = optimizer_cm

        with (cm.slot() if cm else nullcontext()):
            prompt_text = current_prompt
            use_action_judge = bool(self.cfg.get("dspy_action_judge", False)) and action_judge is not None
            state_mode = _normalize_prompt_state_mode(self.cfg.get("dspy_prompt_state_mode", "off"))
            state: Optional[PromptState] = None
            if state_mode != "off":
                state = _build_prompt_state(current_prompt, constraints, placeholders_text, state_mode)
                if state_mode == "auto" and state and not state.structured:
                    state = None
            if state:
                prompt_text = _render_prompt_state(state, context_text=context_text)
            for _ in range(max(1, self.max_iterations)):
                failures: List[str] = []
                actions_to_apply: List[Dict[str, Any]] = []
                for record in dataset:
                    candidate = ""
                    try:
                        prediction = predictor(
                            workflow_input=record["workflow_input"],
                            prompt_template=prompt_text,
                        )
                        candidate = self._extract_prediction_output(prediction)
                    except Exception as exc:  # noqa: BLE001
                        fallback = _extract_lm_response_from_exception(exc)
                        if fallback:
                            candidate = fallback
                            try:
                                logger.debug(
                                    "DSPy predictor解析失败，使用原始 LM 响应回退",
                                    extra={
                                        "workflow_input": record["workflow_input"][:200],
                                        "fallback_preview": candidate[:200],
                                    },
                                )
                            except Exception:
                                pass
                        else:
                            try:
                                logger.warning(
                                    "DSPy predictor 执行失败",
                                    extra={
                                        "workflow_input": record["workflow_input"][:200],
                                        "error": str(exc),
                                    },
                                )
                            except Exception:
                                pass
                            failures.append(f"DSPy predictor failed: {exc}")
                            continue
                    if not candidate.strip():
                        failures.append("DSPy predictor returned empty response")
                        continue
                    if use_action_judge and state:
                        try:
                            action_resp = action_judge(
                                reference=record["reference"],
                                candidate=candidate,
                                prompt_state=_serialize_prompt_state(state),
                            )
                            verdict_raw = _coerce_text(getattr(action_resp, "verdict", "") or "")
                            actions_raw = _coerce_text(
                                getattr(action_resp, "actions_json", "") or getattr(action_resp, "actions", "") or ""
                            )
                            verdict_text = verdict_raw.lower().strip()
                            actions = []
                            if actions_raw:
                                parsed_actions = _safe_json_from_text(actions_raw)
                                if isinstance(parsed_actions, list):
                                    actions = [item for item in parsed_actions if isinstance(item, dict)]
                                elif isinstance(parsed_actions, dict):
                                    actions = [parsed_actions]
                            if not verdict_text or not actions:
                                payload_raw = _coerce_text(action_resp)
                                verdict_text, actions = _parse_action_judge_output(payload_raw)
                            if verdict_text == "pass":
                                continue
                            if actions:
                                actions_to_apply.extend(actions)
                            else:
                                failures.append("Action judge returned fail without actions")
                            continue
                        except Exception as exc:  # noqa: BLE001
                            fallback_feedback = _extract_lm_response_from_exception(exc)
                            detail = fallback_feedback or str(exc)
                            try:
                                logger.warning(
                                    "DSPy action judge 执行失败，回退到文本反馈",
                                    extra={
                                        "workflow_input": record["workflow_input"][:200],
                                        "error": str(exc),
                                        "fallback": detail[:200],
                                    },
                                )
                            except Exception:
                                pass
                            failures.append(f"DSPy action judge failed: {detail}")
                            continue
                    try:
                        verdict = judge(
                            reference=record["reference"],
                            candidate=candidate,
                        )
                        verdict_raw = getattr(verdict, "verdict", None)
                        verdict_text = (verdict_raw or "").lower()
                        feedback = getattr(verdict, "feedback", "") or ""
                        soft_pass, soft_score = _is_reference_soft_satisfied(record["reference"], candidate)
                        if ("fail" in verdict_text or not verdict_text) and soft_pass:
                            verdict_text = "pass"
                            if not feedback:
                                feedback = f"Soft-passed by similarity≈{soft_score:.2f} to reference"
                    except Exception as exc:  # noqa: BLE001
                        fallback_feedback = _extract_lm_response_from_exception(exc)
                        detail = fallback_feedback or str(exc)
                        try:
                            logger.warning(
                                "DSPy judge 执行失败，记为不通过",
                                extra={
                                    "workflow_input": record["workflow_input"][:200],
                                    "error": str(exc),
                                    "fallback": detail[:200],
                                },
                            )
                        except Exception:
                            pass
                        failures.append(f"DSPy judge failed: {detail}")
                        continue
                    if "fail" in verdict_text or not verdict_text:
                        failures.append(feedback or f"Mismatch against reference:\n{record['reference'][:400]}")
                if use_action_judge and state and actions_to_apply:
                    state = _apply_prompt_actions(state, actions_to_apply)
                    prompt_text = _render_prompt_state(state, context_text=context_text)
                    continue
                if use_action_judge and state and failures:
                    use_action_judge = False
                    state = None
                if not failures:
                    break
                feedback_text = "\n".join(f"- {item}" for item in failures)

                def _do_rewrite(masked_prompt: str) -> str:
                    try:
                        rewrite = rewriter(
                            current_prompt=masked_prompt,
                            failures=feedback_text,
                            constraints="\n".join(constraints) if constraints else "None",
                            context=context_text,
                            placeholders=placeholders_text,
                        )
                    except TypeError:
                        rewrite = rewriter(
                            current_prompt=masked_prompt,
                            failures=feedback_text,
                            constraints="\n".join(constraints) if constraints else "None",
                        )
                    return getattr(rewrite, "optimized_prompt", "") or masked_prompt


                new_prompt = _safe_rewrite_prompt(prompt_text, _do_rewrite)
                if new_prompt == prompt_text:
                    break
                prompt_text = new_prompt
            return prompt_text

    def _build_lm(self):
        raw_model = str(self.cfg.get("dspy_model") or self.cfg.get("model"))
        qualified_model = raw_model
        provider = self.cfg.get("dspy_provider") or self.cfg.get("llm_provider")
        if not provider:
            base = str(self.cfg.get("url") or "")
            if "/v1" in base:
                provider = "openai"
        if provider and "/" not in qualified_model:
            qualified_model = f"{provider}/{qualified_model}"
        api_base = _adapt_dspy_api_base(self.cfg.get("url"))

        def _create_lm():
            return self._dspy.LM(  # type: ignore[attr-defined]
                model=qualified_model,
                api_base=api_base,
                api_key=self.cfg.get("api_key"),
                max_tokens=self.max_tokens,
                model_alias_map={qualified_model: raw_model} if qualified_model != raw_model else None,
            )

        try:
            logger.debug(
                "DSPy 构建 LM",
                extra={
                    "qualified_model": qualified_model,
                    "provider": provider,
                    "api_base": api_base,
                    "raw_url": self.cfg.get("url"),
                },
            )
            return _create_lm()
        except Exception as exc:
            try:
                logger.error(
                    "DSPy 构建 LM 失败",
                    extra={
                        "qualified_model": qualified_model,
                        "provider": provider,
                        "api_base": api_base,
                        "raw_url": self.cfg.get("url"),
                        "error": str(exc),
                    },
                )
            except Exception:
                pass
            raise

    def _predict_signature(self):
        dspy = self._dspy

        class WorkflowPredictor(dspy.Signature):  # type: ignore[attr-defined]
            """Simulate workflow output for a prompt template."""

            workflow_input = dspy.InputField(desc="Serialized workflow input for this node")  # type: ignore[attr-defined]
            prompt_template = dspy.InputField(desc="Prompt text to be optimized")  # type: ignore[attr-defined]
            response_json = dspy.OutputField(desc="LLM response")  # type: ignore[attr-defined]
            # response_json = dspy.OutputField(desc="LLM response in json format")  # type: ignore[attr-defined]

        return WorkflowPredictor

    def _judge_signature(self):
        dspy = self._dspy

        class JudgeSignature(dspy.Signature):  # type: ignore[attr-defined]
            """Compare candidate output with reference JSON."""

            reference = dspy.InputField(desc="Reference JSON text")  # type: ignore[attr-defined]
            candidate = dspy.InputField(desc="Candidate JSON output")  # type: ignore[attr-defined]
            verdict = dspy.OutputField(desc="Judge verdict pass/fail")  # type: ignore[attr-defined]
            feedback = dspy.OutputField(desc="Feedback for failures")  # type: ignore[attr-defined]

        return JudgeSignature

    def _action_judge_signature(self):
        dspy = self._dspy

        class ActionJudgeSignature(dspy.Signature):  # type: ignore[attr-defined]
            """Return actionable prompt update instructions."""

            reference = dspy.InputField(desc="Reference JSON text")  # type: ignore[attr-defined]
            candidate = dspy.InputField(desc="Candidate JSON output")  # type: ignore[attr-defined]
            prompt_state = dspy.InputField(desc="Current prompt state as JSON")  # type: ignore[attr-defined]
            verdict = dspy.OutputField(desc="Judge verdict pass/fail")  # type: ignore[attr-defined]
            actions_json = dspy.OutputField(  # type: ignore[attr-defined]
                desc="JSON array of actions to update prompt state",
            )

        return ActionJudgeSignature

    def _rewrite_signature(self):
        dspy = self._dspy

        class RewriteSignature(dspy.Signature):  # type: ignore[attr-defined]
            """Rewrite prompt using judge feedback and constraints."""

            current_prompt = dspy.InputField(desc="Original prompt text")  # type: ignore[attr-defined]
            failures = dspy.InputField(desc="Judge feedback details")  # type: ignore[attr-defined]
            constraints = dspy.InputField(desc="Constraints checklist")  # type: ignore[attr-defined]
            context = dspy.InputField(desc="Node brief and failure signals")  # type: ignore[attr-defined]
            placeholders = dspy.InputField(desc="Placeholders that must stay intact")  # type: ignore[attr-defined]
            optimized_prompt = dspy.OutputField(desc="Improved prompt text")  # type: ignore[attr-defined]

        return RewriteSignature

    def _extract_prediction_output(self, prediction: Any) -> str:
        fields = ("response_json", "response", "output", "text")
        for name in fields:
            value = getattr(prediction, name, None)
            text = _coerce_text(value)
            if text:
                return text
        return _coerce_text(prediction)


def _get_optimizer_concurrency_manager():
    """Fetch optimizer-level concurrency manager from runtime if available."""
    try:
        from src.config.bootstrap import get_runtime

        rt = get_runtime()
        cm = getattr(rt, "optimizer_concurrency", None)
        if cm:
            return cm
        conc_map = getattr(rt, "concurrency", None) or {}
        return conc_map.get("optimizer")
    except Exception:
        return None
