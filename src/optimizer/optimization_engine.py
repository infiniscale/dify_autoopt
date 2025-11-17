"""
Optimizer Module - Optimization Engine

Date: 2025-11-17
Author: backend-developer
Description: Rule-based prompt optimization engine with multiple strategies.
"""

import re
from datetime import datetime
from typing import Any, List, Optional

from loguru import logger

from .exceptions import InvalidStrategyError, OptimizationError, OptimizationFailedError
from .interfaces.llm_client import LLMClient
from .models import (
    OptimizationConfig,
    OptimizationResult,
    OptimizationStrategy,
    Prompt,
    PromptAnalysis,
)
from .prompt_analyzer import PromptAnalyzer


class OptimizationEngine:
    """Rule-based prompt optimization engine.

    Generates optimized prompt variants using different strategies.
    Uses heuristic transformations (no LLM calls in MVP).

    Strategies:
        - clarity_focus: Improve readability and structure
        - efficiency_focus: Reduce length and remove redundancy
        - structure_focus: Add formatting and organization

    Attributes:
        _analyzer: PromptAnalyzer for scoring.
        _llm_client: Optional LLM client (for future use).
        _logger: Loguru logger instance.

    Example:
        >>> engine = OptimizationEngine(analyzer)
        >>> result = engine.optimize(prompt, "clarity_focus")
        >>> print(f"Improvement: {result.improvement_score}")
    """

    def __init__(
        self,
        analyzer: PromptAnalyzer,
        llm_client: Optional[LLMClient] = None,
        custom_logger: Optional[Any] = None,
    ) -> None:
        """Initialize OptimizationEngine.

        Args:
            analyzer: PromptAnalyzer for quality scoring.
            llm_client: Optional LLM client for advanced optimization.
            custom_logger: Optional custom logger instance.
        """
        self._analyzer = analyzer
        self._llm_client = llm_client
        self._logger = custom_logger or logger.bind(module="optimizer.engine")

    def optimize(
        self,
        prompt: Prompt,
        strategy: str,
        config: Optional[OptimizationConfig] = None,
    ) -> OptimizationResult:
        """Generate optimized prompt using specified strategy.

        Args:
            prompt: Prompt to optimize.
            strategy: Strategy name (clarity_focus, efficiency_focus, structure_focus).
            config: Optional optimization configuration.

        Returns:
            OptimizationResult with original and optimized prompts.

        Raises:
            InvalidStrategyError: If strategy name is invalid.
            OptimizationFailedError: If optimization process fails.

        Example:
            >>> result = engine.optimize(prompt, "clarity_focus")
            >>> print(result.optimized_prompt)
        """
        self._logger.info(f"Optimizing prompt '{prompt.id}' with strategy '{strategy}'")

        # Validate strategy
        valid_strategies = ["clarity_focus", "efficiency_focus", "structure_focus"]
        if strategy not in valid_strategies:
            raise InvalidStrategyError(strategy, valid_strategies)

        try:
            # Analyze original prompt
            original_analysis = self._analyzer.analyze_prompt(prompt)

            # Apply optimization strategy
            if strategy == "clarity_focus":
                optimized_text = self._apply_clarity_focus(prompt)
            elif strategy == "efficiency_focus":
                optimized_text = self._apply_efficiency_focus(prompt)
            elif strategy == "structure_focus":
                optimized_text = self._apply_structure_focus(prompt)
            else:
                optimized_text = prompt.text

            # Create optimized prompt object for re-analysis
            optimized_prompt = Prompt(
                id=prompt.id,
                workflow_id=prompt.workflow_id,
                node_id=prompt.node_id,
                node_type=prompt.node_type,
                text=optimized_text,
                role=prompt.role,
                variables=self._extract_variables(optimized_text),
                context=prompt.context,
                extracted_at=prompt.extracted_at,
            )

            # Re-analyze optimized version
            optimized_analysis = self._analyzer.analyze_prompt(optimized_prompt)

            # Calculate improvement
            improvement_score = (
                optimized_analysis.overall_score - original_analysis.overall_score
            )

            # Calculate confidence
            confidence = self._calculate_confidence(
                original_analysis, optimized_analysis
            )

            # Detect changes
            changes = self._detect_changes(prompt.text, optimized_text)

            result = OptimizationResult(
                prompt_id=prompt.id,
                original_prompt=prompt.text,
                optimized_prompt=optimized_text,
                strategy=OptimizationStrategy(strategy),
                improvement_score=improvement_score,
                confidence=confidence,
                changes=changes,
                metadata={
                    "original_score": original_analysis.overall_score,
                    "optimized_score": optimized_analysis.overall_score,
                    "original_clarity": original_analysis.clarity_score,
                    "optimized_clarity": optimized_analysis.clarity_score,
                    "original_efficiency": original_analysis.efficiency_score,
                    "optimized_efficiency": optimized_analysis.efficiency_score,
                },
                optimized_at=datetime.now(),
            )

            self._logger.info(
                f"Optimization complete: improvement={improvement_score:.1f}, "
                f"confidence={confidence:.2f}"
            )

            return result

        except InvalidStrategyError:
            raise
        except Exception as e:
            self._logger.error(f"Optimization failed: {str(e)}")
            raise OptimizationFailedError(
                prompt_id=prompt.id,
                strategy=strategy,
                reason=str(e),
            )

    def _apply_clarity_focus(self, prompt: Prompt) -> str:
        """Apply clarity-focused optimization.

        Transformations:
            - Add clear section headers
            - Break long sentences
            - Replace vague language
            - Add explicit instructions

        Args:
            prompt: Prompt to optimize.

        Returns:
            Optimized prompt text.
        """
        text = prompt.text

        # 1. Add structure if missing (headers)
        if len(text) > 200 and not re.search(r"^#+\s+", text, re.MULTILINE):
            text = self._add_clarity_structure(text)

        # 2. Break long sentences
        text = self._break_long_sentences(text)

        # 3. Replace vague language with specific terms
        text = self._replace_vague_terms(text)

        # 4. Add explicit instruction prefix if missing
        text = self._ensure_clear_instruction(text)

        return text.strip()

    def _apply_efficiency_focus(self, prompt: Prompt) -> str:
        """Apply efficiency-focused optimization.

        Transformations:
            - Remove filler words
            - Eliminate redundancy
            - Compress verbose phrases
            - Remove unnecessary repetition

        Args:
            prompt: Prompt to optimize.

        Returns:
            Optimized prompt text.
        """
        text = prompt.text

        # 1. Remove filler words
        text = self._remove_filler_words(text)

        # 2. Compress verbose phrases
        text = self._compress_verbose_phrases(text)

        # 3. Remove redundant phrases
        text = self._remove_redundancy(text)

        # 4. Clean up whitespace
        text = self._clean_whitespace(text)

        return text.strip()

    def _apply_structure_focus(self, prompt: Prompt) -> str:
        """Apply structure-focused optimization.

        Transformations:
            - Add markdown formatting
            - Create numbered steps
            - Add clear sections
            - Improve visual organization

        Args:
            prompt: Prompt to optimize.

        Returns:
            Optimized prompt text.
        """
        text = prompt.text

        # 1. Add template structure
        text = self._add_template_structure(text)

        # 2. Format as numbered list if contains sequential instructions
        text = self._format_sequential_instructions(text)

        # 3. Add section separators
        text = self._add_section_separators(text)

        return text.strip()

    def apply_clarity_focus(self, text: str) -> str:
        """Public method for clarity optimization (used by StubLLMClient).

        Args:
            text: Raw prompt text.

        Returns:
            Optimized text.
        """
        # Create temporary prompt
        temp_prompt = Prompt(
            id="temp",
            workflow_id="temp",
            node_id="temp",
            node_type="llm",
            text=text,
            role="system",
            variables=[],
            context={},
        )
        return self._apply_clarity_focus(temp_prompt)

    def apply_efficiency_focus(self, text: str) -> str:
        """Public method for efficiency optimization (used by StubLLMClient).

        Args:
            text: Raw prompt text.

        Returns:
            Optimized text.
        """
        temp_prompt = Prompt(
            id="temp",
            workflow_id="temp",
            node_id="temp",
            node_type="llm",
            text=text,
            role="system",
            variables=[],
            context={},
        )
        return self._apply_efficiency_focus(temp_prompt)

    def apply_structure_optimization(self, text: str) -> str:
        """Public method for structure optimization (used by StubLLMClient).

        Args:
            text: Raw prompt text.

        Returns:
            Optimized text.
        """
        temp_prompt = Prompt(
            id="temp",
            workflow_id="temp",
            node_id="temp",
            node_type="llm",
            text=text,
            role="system",
            variables=[],
            context={},
        )
        return self._apply_structure_focus(temp_prompt)

    def _add_clarity_structure(self, text: str) -> str:
        """Add clarity-focused structure to text.

        Args:
            text: Original text.

        Returns:
            Text with added structure.
        """
        # Split into logical parts
        parts = text.split("\n\n")

        if len(parts) >= 2:
            # Add headers to sections
            structured = "## Instructions\n\n"
            structured += parts[0] + "\n\n"

            if len(parts) > 1:
                structured += "## Details\n\n"
                structured += "\n\n".join(parts[1:])

            return structured

        # Single block: wrap with header
        return f"## Task Instructions\n\n{text}"

    def _break_long_sentences(self, text: str) -> str:
        """Break long sentences into shorter ones.

        Args:
            text: Original text.

        Returns:
            Text with shorter sentences.
        """
        # Find sentences longer than 40 words
        sentences = re.split(r"(?<=[.!?])\s+", text)
        result = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) > 40:
                # Try to split at conjunctions
                split_points = [
                    " and ",
                    " but ",
                    " however ",
                    " therefore ",
                    ", which ",
                ]
                modified = sentence
                for point in split_points:
                    if point in modified:
                        parts = modified.split(point, 1)
                        if len(parts) == 2 and len(parts[0].split()) > 10:
                            modified = parts[0] + ".\n" + parts[1].capitalize()
                            break
                result.append(modified)
            else:
                result.append(sentence)

        return " ".join(result)

    def _replace_vague_terms(self, text: str) -> str:
        """Replace vague terms with specific language.

        Args:
            text: Original text.

        Returns:
            Text with specific language.
        """
        replacements = {
            r"\bsome\b": "specific",
            r"\bmaybe\b": "if applicable",
            r"\bstuff\b": "content",
            r"\bthings\b": "items",
            r"\ba bit\b": "slightly",
            r"\bkind of\b": "type of",
            r"\bsort of\b": "type of",
            r"\bprobably\b": "likely",
            r"\bretc\.?\b": "and similar items",
        }

        result = text
        for pattern, replacement in replacements.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def _ensure_clear_instruction(self, text: str) -> str:
        """Ensure prompt starts with clear instruction.

        Args:
            text: Original text.

        Returns:
            Text with clear instruction.
        """
        # Check if starts with action verb
        first_word = text.strip().split()[0].lower() if text.strip() else ""

        action_verbs = [
            "summarize",
            "analyze",
            "extract",
            "generate",
            "write",
            "create",
            "list",
            "identify",
            "classify",
            "compare",
            "explain",
            "describe",
            "provide",
            "format",
        ]

        if first_word not in action_verbs:
            # Add instruction prefix
            return f"Please {text}"

        return text

    def _remove_filler_words(self, text: str) -> str:
        """Remove filler words from text.

        Args:
            text: Original text.

        Returns:
            Text without filler words.
        """
        filler_patterns = [
            r"\bvery\s+",
            r"\breally\s+",
            r"\bjust\s+",
            r"\bactually\s+",
            r"\bbasically\s+",
            r"\bliterally\s+",
            r"\btotally\s+",
            r"\bquite\s+",
            r"\bsimply\s+",
        ]

        result = text
        for pattern in filler_patterns:
            result = re.sub(pattern, "", result, flags=re.IGNORECASE)

        return result

    def _compress_verbose_phrases(self, text: str) -> str:
        """Compress verbose phrases to concise versions.

        Args:
            text: Original text.

        Returns:
            Text with compressed phrases.
        """
        compressions = {
            r"\bin order to\b": "to",
            r"\bdue to the fact that\b": "because",
            r"\bat this point in time\b": "now",
            r"\bin the event that\b": "if",
            r"\bfor the purpose of\b": "to",
            r"\bwith regard to\b": "about",
            r"\bin accordance with\b": "following",
            r"\bprior to\b": "before",
            r"\bsubsequent to\b": "after",
            r"\bin spite of\b": "despite",
            r"\bnotwithstanding\b": "despite",
            r"\bby means of\b": "by",
        }

        result = text
        for pattern, replacement in compressions.items():
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def _remove_redundancy(self, text: str) -> str:
        """Remove redundant repeated phrases.

        Args:
            text: Original text.

        Returns:
            Text without redundancy.
        """
        # Find and remove duplicate adjacent phrases
        words = text.split()
        if len(words) < 6:
            return text

        # Check for repeated trigrams
        i = 0
        result_words = []
        while i < len(words):
            if i + 5 < len(words):
                trigram1 = " ".join(words[i : i + 3])
                trigram2 = " ".join(words[i + 3 : i + 6])
                if trigram1.lower() == trigram2.lower():
                    # Skip duplicate
                    result_words.extend(words[i : i + 3])
                    i += 6
                    continue
            result_words.append(words[i])
            i += 1

        return " ".join(result_words)

    def _clean_whitespace(self, text: str) -> str:
        """Clean up excessive whitespace.

        Args:
            text: Original text.

        Returns:
            Text with normalized whitespace.
        """
        # Remove multiple spaces
        text = re.sub(r" +", " ", text)

        # Remove multiple newlines
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove trailing spaces
        lines = [line.rstrip() for line in text.split("\n")]

        return "\n".join(lines)

    def _add_template_structure(self, text: str) -> str:
        """Add template-based structure.

        Args:
            text: Original text.

        Returns:
            Text with template structure.
        """
        # If no clear structure, wrap in template
        if not re.search(r"^#+\s+", text, re.MULTILINE):
            template = "# Task\n\n"
            template += text + "\n\n"
            template += "# Expected Output\n\n"
            template += "Please provide a clear and structured response."
            return template

        return text

    def _format_sequential_instructions(self, text: str) -> str:
        """Format sequential instructions as numbered list.

        Args:
            text: Original text.

        Returns:
            Text with numbered instructions.
        """
        # Check for sequential indicators
        sequential_markers = [
            "first",
            "second",
            "third",
            "then",
            "next",
            "finally",
            "after that",
        ]

        has_sequence = any(marker in text.lower() for marker in sequential_markers)

        if has_sequence and not re.search(r"^\d+\.\s+", text, re.MULTILINE):
            # Split by markers and number
            parts = re.split(
                r"\b(first|second|third|then|next|finally|after that)\b",
                text,
                flags=re.IGNORECASE,
            )

            if len(parts) > 2:
                numbered = []
                step = 1
                current = ""

                for i, part in enumerate(parts):
                    if part.lower() in sequential_markers:
                        if current.strip():
                            numbered.append(f"{step}. {current.strip()}")
                            step += 1
                        current = ""
                    else:
                        current += part

                if current.strip():
                    numbered.append(f"{step}. {current.strip()}")

                return "\n".join(numbered)

        return text

    def _add_section_separators(self, text: str) -> str:
        """Add visual section separators.

        Args:
            text: Original text.

        Returns:
            Text with section separators.
        """
        # Add separators between major sections
        if "---" not in text and len(text) > 300:
            # Add separator after headers
            text = re.sub(r"(^#+\s+.+\n)", r"\1---\n", text, flags=re.MULTILINE)

        return text

    def _calculate_confidence(
        self,
        original: PromptAnalysis,
        optimized: PromptAnalysis,
    ) -> float:
        """Calculate confidence in the optimization result.

        Confidence based on:
            - Score improvement magnitude
            - Both clarity and efficiency improved
            - No significant regression

        Args:
            original: Original prompt analysis.
            optimized: Optimized prompt analysis.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        # Base confidence
        confidence = 0.5

        # Improvement factors
        overall_improvement = optimized.overall_score - original.overall_score
        clarity_improvement = optimized.clarity_score - original.clarity_score
        efficiency_improvement = optimized.efficiency_score - original.efficiency_score

        # Add confidence for overall improvement
        if overall_improvement > 10:
            confidence += 0.3
        elif overall_improvement > 5:
            confidence += 0.2
        elif overall_improvement > 0:
            confidence += 0.1

        # Add confidence if both metrics improved
        if clarity_improvement > 0 and efficiency_improvement > 0:
            confidence += 0.15

        # Reduce confidence for regression in any metric
        if clarity_improvement < -5 or efficiency_improvement < -5:
            confidence -= 0.2

        # Cap confidence
        return max(0.0, min(1.0, confidence))

    def _detect_changes(self, original: str, optimized: str) -> List[str]:
        """Detect and describe changes between prompts.

        Args:
            original: Original prompt text.
            optimized: Optimized prompt text.

        Returns:
            List of change descriptions.
        """
        changes = []

        # Length change
        len_diff = len(optimized) - len(original)
        if len_diff > 50:
            changes.append(f"Added {len_diff} characters")
        elif len_diff < -50:
            changes.append(f"Reduced by {-len_diff} characters")

        # Structure changes
        if re.search(r"^#+\s+", optimized, re.MULTILINE) and not re.search(
            r"^#+\s+", original, re.MULTILINE
        ):
            changes.append("Added section headers")

        if re.search(r"^\d+\.\s+", optimized, re.MULTILINE) and not re.search(
            r"^\d+\.\s+", original, re.MULTILINE
        ):
            changes.append("Added numbered list")

        if re.search(r"^[\-\*]\s+", optimized, re.MULTILINE) and not re.search(
            r"^[\-\*]\s+", original, re.MULTILINE
        ):
            changes.append("Added bullet points")

        # Line break changes
        orig_lines = original.count("\n")
        opt_lines = optimized.count("\n")
        if opt_lines > orig_lines + 2:
            changes.append("Improved formatting with line breaks")

        # Word count change
        orig_words = len(original.split())
        opt_words = len(optimized.split())
        if opt_words < orig_words * 0.8:
            changes.append("Reduced word count for efficiency")
        elif opt_words > orig_words * 1.2:
            changes.append("Added more detailed instructions")

        if not changes:
            changes.append("Minor text adjustments")

        return changes

    def _extract_variables(self, text: str) -> List[str]:
        """Extract variables from text.

        Args:
            text: Prompt text.

        Returns:
            List of variable names.
        """
        pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
        matches = re.findall(pattern, text)
        seen = set()
        unique_vars = []
        for var in matches:
            if var not in seen:
                seen.add(var)
                unique_vars.append(var)
        return unique_vars
