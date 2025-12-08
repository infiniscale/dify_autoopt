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


def _extract_graph(tree: Dict[str, Any]) -> Dict[str, Any]:
    """Extract graph section from either root.graph or root.workflow.graph."""
    if not isinstance(tree, dict):
        return {}
    if isinstance(tree.get("graph"), dict):
        return tree.get("graph") or {}
    workflow = tree.get("workflow")
    if isinstance(workflow, dict) and isinstance(workflow.get("graph"), dict):
        return workflow.get("graph") or {}
    return {}


def _iter_llm_prompts(workflow_yaml: Dict[str, Any]) -> Iterable[PromptLocation]:
    """Yield prompt locations for LLM nodes in a Dify workflow DSL."""
    graph = _extract_graph(workflow_yaml or {})
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
                    path = f"/graph/{bucket_name}/{idx}/data/prompt_template/{sub_path + '/' if sub_path else ''}{mi}/text"
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
                path = f"/graph/{bucket_name}/{idx}/data/system_prompt"
                yield PromptLocation(node_id=node_id, path=path, text=str(data["system_prompt"]))


def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #


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
                if len(run_ids) == len(samples):  # all missing -> strong signal
                    issues.append(
                        DetectedIssue(
                            kind="constraint_missing",
                            severity="high",
                            message=f"Constraint not observed in outputs: {constraint}",
                            evidence_runs=run_ids,
                        )
                    )

        # Latency guard
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


class RuleBasedPatchGenerator:
    """Generate simple prompt patches to address detected issues."""

    def __init__(self, constraints: Sequence[str]) -> None:
        self.constraints = list(constraints)

    def generate(self, workflow_id: str, issues: Sequence[DetectedIssue], prompts: Sequence[PromptLocation]) -> List[PromptPatch]:
        patches: List[PromptPatch] = []
        if not issues or not prompts:
            try:
                logger.info(
                    "未生成补丁，可能原因：无 issue 或未找到 LLM prompt",
                    extra={
                        "issues": len(issues),
                        "prompts": len(prompts),
                        "workflow_id": workflow_id,
                        "prompt_samples": [p.text[:120] for p in prompts[:3]],
                    },
                )
            except Exception:
                pass
            return patches

        for prompt in prompts:
            appended_segments: List[str] = []
            rationales: List[str] = []
            confidence = 0.55

            for issue in issues:
                if issue.kind == "constraint_missing":
                    # add explicit constraint reminder
                    for constraint in self.constraints:
                        appended_segments.append(f"Ensure the response includes: {constraint}.")
                    rationales.append("Enforce missing constraints in output.")
                    confidence = max(confidence, 0.7)
                elif issue.kind == "low_similarity":
                    appended_segments.append("Align responses closely with the reference answer and avoid deviations.")
                    rationales.append("Outputs diverge from reference expectation.")
                    confidence = max(confidence, 0.65)
                elif issue.kind == "high_failure_rate":
                    appended_segments.append("Ensure robustness and handle edge cases explicitly.")
                    rationales.append("Observed high failure rate across runs.")
                    confidence = max(confidence, 0.6)
                elif issue.kind == "latency_exceeded":
                    appended_segments.append("Prioritize concise responses to reduce latency.")
                    rationales.append("Response latency above target.")
                    confidence = max(confidence, 0.6)
                elif issue.kind == "llm_output_mismatch":
                    appended_segments.append("Ensure the response aligns with the reference answer and satisfies listed constraints.")
                    # Pull missing/incorrect hints if present in message
                    hints = ""
                    if "missing_items=" in issue.message or "incorrect_items=" in issue.message:
                        hints = issue.message
                    rationales.append("LLM judge flagged mismatch against reference/constraints." + (f" Details: {hints}" if hints else ""))
                    confidence = max(confidence, 0.7)
                    # Add checklist guidance
                    appended_segments.append("Use a checklist: cover all must-have items from reference, fix incorrect interpretations, and respect required format if specified.")
                else:
                    rationales.append(f"Address issue: {issue.kind}")
                    confidence = max(confidence, 0.55)

            if not appended_segments:
                continue

            # dedupe while preserving order
            seen = set()
            unique_segments = []
            for seg in appended_segments:
                if seg in seen:
                    continue
                seen.add(seg)
                unique_segments.append(seg)

            new_text_parts = [prompt.text.strip(), ""]
            new_text_parts.extend(f"- {seg}" for seg in unique_segments)
            new_prompt = "\n".join(new_text_parts).strip()
            patches.append(
                PromptPatch(
                    workflow_id=workflow_id,
                    node_id=prompt.node_id,
                    field_path=prompt.path,
                    old=prompt.text,
                    new=new_prompt,
                    rationale="; ".join(rationales) or "Improve prompt robustness",
                    confidence=round(confidence, 2),
                    evidence_runs=sorted({rid for issue in issues for rid in issue.evidence_runs}),
                )
            )
        return patches


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

        generator = RuleBasedPatchGenerator(reference.constraints)
        patches = generator.generate(workflow_id, issues, filtered_prompts)

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
