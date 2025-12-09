"""
Lightweight prompt optimizer pipeline.

Based on execution results and reference expectations, detect problematic
prompts in a workflow DSL and generate patch suggestions. The goal is to
produce reviewable patches (not auto-apply) that can later be reviewed or
fed to a more advanced optimizer (e.g., dspy) when available.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import copy
import json as _json

import yaml


from src.optimizer.yaml_loader import load_workflow_yaml
from src.utils.logger import get_logger
from src.workflow.export import _safe_dirname_from_id

logger = get_logger("optimizer.prompt_optimizer")
DEFAULT_DSPY_MAX_TOKENS = 4096


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
class OptimizationReport:
    workflow_id: str
    issues: List[DetectedIssue]
    patches: List[PromptPatch]
    yaml_path: Path
    reference_path: Optional[Path]
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


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


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

        prompts = list(_iter_llm_prompts(workflow_tree))
        judge = _LlmPromptJudge(self.llm_config) if self.llm_config else None
        # Determine which prompts need change: if judge exists, use it; else default to "issues exist"
        filtered_prompts: List[PromptLocation] = []
        for prompt in prompts:
            needs = True if not judge else judge.needs_change(prompt.text, issues, reference.constraints)
            if needs:
                filtered_prompts.append(prompt)

        patches: List[PromptPatch] = []
        use_dspy = self.llm_config.get("use_dspy_optimizer", True)
        if use_dspy:
            dspy_optimizer = _DspyPromptOptimizer(self.llm_config)
            try:
                patches = dspy_optimizer.optimize_prompts(
                    workflow_id,
                    filtered_prompts,
                    samples,
                    reference_texts or [],
                    reference.constraints,
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
            refined: List[PromptPatch] = []
            for patch in patches:
                try:
                    new_text = rewriter.rewrite(patch.new, issues, reference.constraints)
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
            yaml_path=workflow_path,
            reference_path=Path(reference_path) if reference_path else None,
            stats={
                "runs": len(samples),
                "failures": len([s for s in samples if s.status not in {"success", "succeeded"}]),
                "reference_texts": len([t for t in (reference_texts or []) if t]),
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
            for patch in patches:
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

    def rewrite(self, current: str, issues: Sequence[DetectedIssue], constraints: Sequence[str]) -> str:
        import requests

        prompt_lines = [
            "You are a prompt editor. Improve the prompt to address the following issues:",
        ]
        for it in issues:
            prompt_lines.append(f"- {it.kind}: {it.message}")
        if constraints:
            prompt_lines.append("Ensure these constraints are included:")
            for c in constraints:
                prompt_lines.append(f"- {c}")
        prompt_lines.append("Original prompt:")
        prompt_lines.append(current)
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
        # Best-effort extraction
        return _extract_llm_text_response(data) or current


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

    def rewrite(self, current: str, issues: Sequence[DetectedIssue], constraints: Sequence[str]) -> str:
        if not self.available:
            return current
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
            original_prompt: str
            improved_prompt: str

        chain = self._dspy.ChainOfThought(RewriteSpec, lm=lm)  # type: ignore[attr-defined]
        issues_text = "; ".join(f"{it.kind}: {it.message}" for it in issues) or "none"
        constraints_text = "; ".join(constraints) or "none"
        res = chain(issues=issues_text, constraints=constraints_text, original_prompt=current)
        return getattr(res, "improved_prompt", current) or current


class _DspyPromptOptimizer:
    """DSPy-based prompt optimizer that loops Predictor -> Judge -> Prompt update."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.available = False
        self.max_iterations = int(cfg.get("dspy_optimizer_iterations", 2))
        self.max_samples = int(cfg.get("dspy_optimizer_max_samples", 4))
        self.max_tokens = int(cfg.get("dspy_optimizer_max_tokens", DEFAULT_DSPY_MAX_TOKENS))
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
        dataset = self._build_dataset(samples, reference_texts)
        if not dataset:
            return []
        # lm = self._lm or self._build_lm()
        # predictor = self._dspy.ChainOfThought(self._predict_signature(), lm=lm)  # type: ignore[attr-defined]
        # judge = self._dspy.ChainOfThought(self._judge_signature(), lm=lm)  # type: ignore[attr-defined]
        # rewriter = self._dspy.ChainOfThought(self._rewrite_signature(), lm=lm)  # type: ignore[attr-defined]
        predictor = self._dspy.ChainOfThought(self._predict_signature())  # type: ignore[attr-defined]
        judge = self._dspy.ChainOfThought(self._judge_signature())  # type: ignore[attr-defined]
        rewriter = self._dspy.ChainOfThought(self._rewrite_signature())  # type: ignore[attr-defined]
        patches: List[PromptPatch] = []
        for prompt in prompts:
            improved = self._optimize_single(prompt.text, dataset, constraints, predictor, judge, rewriter)
            if not improved or improved.strip() == prompt.text.strip():
                continue
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
            patches.append(
                PromptPatch(
                    workflow_id=workflow_id,
                    node_id=prompt.node_id,
                    field_path=prompt.path,
                    old=prompt.text,
                    new=improved,
                    rationale="DSPy optimizer updated prompt based on reference feedback",
                    confidence=0.75,
                    evidence_runs=[record["run_index"] for record in dataset],
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

    def _optimize_single(
        self,
        current_prompt: str,
        dataset: Sequence[Dict[str, Any]],
        constraints: Sequence[str],
        predictor,
        judge,
        rewriter,
    ) -> str:
        prompt_text = current_prompt
        for _ in range(max(1, self.max_iterations)):
            failures: List[str] = []
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
                try:
                    verdict = judge(
                        reference=record["reference"],
                        candidate=candidate,
                    )
                    verdict_raw = getattr(verdict, "verdict", None)
                    verdict_text = (verdict_raw or "").lower()
                    feedback = getattr(verdict, "feedback", "") or ""
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
            if not failures:
                break
            feedback_text = "\n".join(f"- {item}" for item in failures)
            try:
                rewrite = rewriter(
                    current_prompt=prompt_text,
                    failures=feedback_text,
                    constraints="\n".join(constraints) if constraints else "None",
                )
                prompt_text = getattr(rewrite, "optimized_prompt", "") or prompt_text
            except Exception as exc:  # noqa: BLE001
                fallback_prompt = _extract_lm_response_from_exception(exc)
                if fallback_prompt:
                    prompt_text = fallback_prompt
                    try:
                        logger.debug(
                            "DSPy rewriter 解析失败，使用回退提示词",
                            extra={"fallback_preview": fallback_prompt[:200]},
                        )
                    except Exception:
                        pass
                else:
                    try:
                        logger.warning(
                            "DSPy rewriter 执行失败，保持原提示词",
                            extra={"error": str(exc)},
                        )
                    except Exception:
                        pass
                    break
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

    def _rewrite_signature(self):
        dspy = self._dspy

        class RewriteSignature(dspy.Signature):  # type: ignore[attr-defined]
            """Rewrite prompt using judge feedback and constraints."""

            current_prompt = dspy.InputField(desc="Original prompt text")  # type: ignore[attr-defined]
            failures = dspy.InputField(desc="Judge feedback details")  # type: ignore[attr-defined]
            constraints = dspy.InputField(desc="Constraints checklist")  # type: ignore[attr-defined]
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
