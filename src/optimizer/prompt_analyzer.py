"""
Optimizer Module - Prompt Analyzer

Date: 2025-11-17
Author: backend-developer
Description: Rule-based prompt quality analyzer with scoring and issue detection.
"""

import re
from datetime import datetime
from typing import Any, List, Optional

from loguru import logger

from .exceptions import AnalysisError, ScoringError
from .interfaces.llm_client import LLMClient
from .models import (
    IssueSeverity,
    IssueType,
    Prompt,
    PromptAnalysis,
    PromptIssue,
    PromptSuggestion,
    SuggestionType,
)


class PromptAnalyzer:
    """Rule-based prompt quality analyzer.

    Analyzes prompts using heuristic rules to calculate quality scores
    and detect common issues.

    Scoring Formula (from SRS):
        - Clarity score = 0.4 * structure + 0.3 * specificity + 0.3 * coherence
        - Efficiency score = 0.5 * token_efficiency + 0.5 * information_density
        - Overall score = 0.6 * clarity + 0.4 * efficiency

    Issue Detection Rules:
        1. too_long: > 2000 characters
        2. too_short: < 20 characters
        3. vague_language: Contains vague terms (some, maybe, etc.)
        4. missing_structure: No clear sections or bullet points
        5. redundancy: Repeated phrases
        6. poor_formatting: No line breaks in long prompts
        7. ambiguous_instructions: Missing concrete action verbs

    Attributes:
        _llm_client: Optional LLM client for advanced analysis (future).
        _logger: Loguru logger instance.

    Example:
        >>> analyzer = PromptAnalyzer()
        >>> analysis = analyzer.analyze_prompt(prompt)
        >>> print(f"Score: {analysis.overall_score}")
    """

    # Vague language patterns
    VAGUE_PATTERNS = [
        r"\bsome\b",
        r"\bmaybe\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bsomehow\b",
        r"\bsomewhat\b",
        r"\bkind of\b",
        r"\bsort of\b",
        r"\ba bit\b",
        r"\bstuff\b",
        r"\bthings\b",
        r"\betc\b",
        r"\b(and )?so on\b",
        r"\bwhatever\b",
    ]

    # Pre-compiled regex for performance
    _VAGUE_REGEX = re.compile('|'.join(VAGUE_PATTERNS), re.IGNORECASE)

    # Action verb patterns
    ACTION_VERBS = [
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
        "translate",
        "convert",
        "calculate",
        "evaluate",
        "recommend",
        "suggest",
    ]

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        custom_logger: Optional[Any] = None,
    ) -> None:
        """Initialize PromptAnalyzer.

        Args:
            llm_client: Optional LLM client for advanced analysis.
            custom_logger: Optional custom logger instance.
        """
        self._llm_client = llm_client
        self._logger = custom_logger or logger.bind(module="optimizer.analyzer")

    def analyze_prompt(self, prompt: Prompt) -> PromptAnalysis:
        """Analyze prompt quality and return assessment.

        Args:
            prompt: Prompt object to analyze.

        Returns:
            PromptAnalysis with scores, issues, and suggestions.

        Raises:
            AnalysisError: If analysis fails.

        Example:
            >>> analysis = analyzer.analyze_prompt(prompt)
            >>> print(f"Clarity: {analysis.clarity_score}")
            >>> print(f"Issues: {len(analysis.issues)}")
        """
        self._logger.info(f"Analyzing prompt: {prompt.id}")

        try:
            # Calculate scores
            clarity_score = self._calculate_clarity_score(prompt)
            efficiency_score = self._calculate_efficiency_score(prompt)

            # Overall score (weighted average)
            overall_score = 0.6 * clarity_score + 0.4 * efficiency_score

            # Detect issues
            issues = self._detect_issues(prompt)

            # Generate suggestions
            suggestions = self._generate_suggestions(prompt, issues)

            # Compute metadata
            metadata = self._compute_metadata(prompt)

            analysis = PromptAnalysis(
                prompt_id=prompt.id,
                overall_score=overall_score,
                clarity_score=clarity_score,
                efficiency_score=efficiency_score,
                issues=issues,
                suggestions=suggestions,
                metadata=metadata,
                analyzed_at=datetime.now(),
            )

            self._logger.info(
                f"Analysis complete for '{prompt.id}': "
                f"overall={overall_score:.1f}, clarity={clarity_score:.1f}, "
                f"efficiency={efficiency_score:.1f}, issues={len(issues)}"
            )

            return analysis

        except Exception as e:
            self._logger.error(f"Analysis failed for '{prompt.id}': {str(e)}")
            raise AnalysisError(
                message=f"Failed to analyze prompt '{prompt.id}'",
                error_code="OPT-ANA-010",
                context={"prompt_id": prompt.id, "error": str(e)},
            )

    def _calculate_clarity_score(self, prompt: Prompt) -> float:
        """Calculate clarity score (0-100).

        Formula: 0.4 * structure + 0.3 * specificity + 0.3 * coherence

        Args:
            prompt: Prompt to score.

        Returns:
            Clarity score between 0 and 100.
        """
        try:
            structure_score = self._score_structure(prompt.text)
            specificity_score = self._score_specificity(prompt.text)
            coherence_score = self._score_coherence(prompt.text)

            clarity = (
                0.4 * structure_score + 0.3 * specificity_score + 0.3 * coherence_score
            )

            return min(100.0, max(0.0, clarity))

        except Exception as e:
            raise ScoringError(
                metric_name="clarity",
                reason=str(e),
            )

    def _calculate_efficiency_score(self, prompt: Prompt) -> float:
        """Calculate efficiency score (0-100).

        Formula: 0.5 * token_efficiency + 0.5 * information_density

        Args:
            prompt: Prompt to score.

        Returns:
            Efficiency score between 0 and 100.
        """
        try:
            token_efficiency = self._score_token_efficiency(prompt.text)
            info_density = self._score_information_density(prompt.text)

            efficiency = 0.5 * token_efficiency + 0.5 * info_density

            return min(100.0, max(0.0, efficiency))

        except Exception as e:
            raise ScoringError(
                metric_name="efficiency",
                reason=str(e),
            )

    def _score_structure(self, text: str) -> float:
        """Score prompt structure (0-100).

        Higher score for:
            - Clear sections (headers, bullets)
            - Proper formatting (line breaks)
            - Organized layout

        Args:
            text: Prompt text.

        Returns:
            Structure score.
        """
        score = 50.0  # Base score

        # Check for markdown headers
        if re.search(r"^#+\s+", text, re.MULTILINE):
            score += 15

        # Check for bullet points
        if re.search(r"^[\-\*\d+\.]\s+", text, re.MULTILINE):
            score += 15

        # Check for line breaks (proper formatting)
        line_count = text.count("\n")
        if line_count > 2:
            score += 10
        elif line_count > 0:
            score += 5

        # Check for numbered steps
        if re.search(r"^\d+\.\s+", text, re.MULTILINE):
            score += 10

        # Penalize very long single lines (no formatting)
        lines = text.split("\n")
        avg_line_length = sum(len(line) for line in lines) / max(1, len(lines))
        if avg_line_length > 200:
            score -= 20

        return min(100.0, max(0.0, score))

    def _score_specificity(self, text: str) -> float:
        """Score prompt specificity (0-100).

        Higher score for:
            - Specific instructions
            - Action verbs
            - Concrete examples
            - No vague language

        Args:
            text: Prompt text.

        Returns:
            Specificity score.
        """
        score = 60.0  # Base score

        # Check for action verbs
        text_lower = text.lower()
        action_count = sum(1 for verb in self.ACTION_VERBS if verb in text_lower)
        score += min(20, action_count * 5)

        # Penalize vague language
        vague_count = sum(
            len(re.findall(pattern, text_lower, re.IGNORECASE))
            for pattern in self.VAGUE_PATTERNS
        )
        score -= min(30, vague_count * 5)

        # Check for specific numbers or quantities
        if re.search(r"\b\d+\b", text):
            score += 10

        # Check for examples
        if "example" in text_lower or "e.g." in text_lower or "such as" in text_lower:
            score += 10

        return min(100.0, max(0.0, score))

    def _score_coherence(self, text: str) -> float:
        """Score prompt coherence (0-100).

        Higher score for:
            - Logical flow
            - Consistent terminology
            - Clear sentence structure

        Args:
            text: Prompt text.

        Returns:
            Coherence score.
        """
        score = 70.0  # Base score

        # Analyze sentence structure
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return 50.0

        # Check sentence length consistency
        lengths = [len(s.split()) for s in sentences]
        if lengths:
            avg_length = sum(lengths) / len(lengths)

            # Good average sentence length (10-25 words)
            if 10 <= avg_length <= 25:
                score += 15
            elif avg_length < 5 or avg_length > 40:
                score -= 15

            # Check for very long sentences
            long_sentences = sum(1 for length in lengths if length > 35)
            score -= min(20, long_sentences * 10)

        # Check for transition words
        transition_words = [
            "first",
            "second",
            "then",
            "next",
            "finally",
            "therefore",
            "however",
            "additionally",
            "furthermore",
        ]
        transition_count = sum(1 for word in transition_words if word in text.lower())
        score += min(15, transition_count * 5)

        return min(100.0, max(0.0, score))

    def _score_token_efficiency(self, text: str) -> float:
        """Score token efficiency (0-100).

        Higher score for:
            - Optimal length (not too long, not too short)
            - No redundancy
            - Concise language

        Args:
            text: Prompt text.

        Returns:
            Token efficiency score.
        """
        # Estimate token count (rough: 1 token ~= 4 characters)
        token_count = len(text) / 4

        # Optimal range: 50-500 tokens
        if 50 <= token_count <= 500:
            score = 90.0
        elif 20 <= token_count < 50:
            score = 70.0
        elif 500 < token_count <= 1000:
            score = 70.0
        elif token_count < 20:
            score = 40.0
        else:  # > 1000 tokens
            score = 50.0

        # Penalize repetition
        words = text.lower().split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 20
            elif unique_ratio < 0.7:
                score -= 10

        return min(100.0, max(0.0, score))

    def _score_information_density(self, text: str) -> float:
        """Score information density (0-100).

        Higher score for:
            - Rich content relative to length
            - No filler words
            - High semantic value

        Args:
            text: Prompt text.

        Returns:
            Information density score.
        """
        score = 70.0  # Base score

        words = text.lower().split()
        if not words:
            return 50.0

        # Check for filler words
        filler_words = [
            "very",
            "really",
            "just",
            "actually",
            "basically",
            "literally",
            "totally",
            "absolutely",
            "definitely",
            "certainly",
        ]
        filler_count = sum(1 for word in words if word in filler_words)
        filler_ratio = filler_count / len(words)
        score -= min(30, filler_ratio * 100)

        # Check for meaningful content words
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "under",
            "again",
            "further",
            "once",
            "and",
            "or",
            "but",
            "if",
            "because",
            "while",
            "where",
            "when",
            "how",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "it",
            "its",
        }

        content_words = [w for w in words if w not in stopwords]
        content_ratio = len(content_words) / len(words) if words else 0
        score += min(20, content_ratio * 30)

        return min(100.0, max(0.0, score))

    def _detect_issues(self, prompt: Prompt) -> List[PromptIssue]:
        """Detect issues in prompt.

        Args:
            prompt: Prompt to check.

        Returns:
            List of detected issues.
        """
        issues: List[PromptIssue] = []
        text = prompt.text

        # 1. Check for too long prompt
        if len(text) > 2000:
            issues.append(
                PromptIssue(
                    severity=IssueSeverity.WARNING,
                    type=IssueType.TOO_LONG,
                    description=f"Prompt is too long ({len(text)} characters)",
                    location="Entire prompt",
                    suggestion="Consider breaking down into smaller parts or removing redundant content",
                )
            )

        # 2. Check for too short prompt
        if len(text) < 20:
            issues.append(
                PromptIssue(
                    severity=IssueSeverity.WARNING,
                    type=IssueType.TOO_SHORT,
                    description=f"Prompt is too short ({len(text)} characters)",
                    location="Entire prompt",
                    suggestion="Add more context and specific instructions",
                )
            )

        # 3. Check for vague language
        vague_matches = []
        for pattern in self.VAGUE_PATTERNS:
            matches = re.findall(pattern, text.lower())
            vague_matches.extend(matches)

        if vague_matches:
            issues.append(
                PromptIssue(
                    severity=IssueSeverity.WARNING,
                    type=IssueType.VAGUE_LANGUAGE,
                    description=f"Contains vague terms: {', '.join(set(vague_matches))}",
                    location="Throughout prompt",
                    suggestion="Replace vague language with specific terms",
                )
            )

        # 4. Check for missing structure
        has_structure = (
            re.search(r"^#+\s+", text, re.MULTILINE)  # Headers
            or re.search(r"^[\-\*]\s+", text, re.MULTILINE)  # Bullets
            or re.search(r"^\d+\.\s+", text, re.MULTILINE)  # Numbers
        )

        if len(text) > 200 and not has_structure:
            issues.append(
                PromptIssue(
                    severity=IssueSeverity.INFO,
                    type=IssueType.MISSING_STRUCTURE,
                    description="Long prompt lacks clear structure",
                    location="Entire prompt",
                    suggestion="Add headers, bullet points, or numbered steps",
                )
            )

        # 5. Check for redundancy (repeated phrases)
        words = text.lower().split()
        if len(words) > 10:
            # Check for repeated 3-grams
            trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
            from collections import Counter

            trigram_counts = Counter(trigrams)
            repeated = [tg for tg, count in trigram_counts.items() if count > 1]

            if len(repeated) > 2:
                issues.append(
                    PromptIssue(
                        severity=IssueSeverity.INFO,
                        type=IssueType.REDUNDANCY,
                        description=f"Contains {len(repeated)} repeated phrases",
                        location="Throughout prompt",
                        suggestion="Remove redundant phrases to improve conciseness",
                    )
                )

        # 6. Check for poor formatting
        if len(text) > 500 and text.count("\n") < 2:
            issues.append(
                PromptIssue(
                    severity=IssueSeverity.INFO,
                    type=IssueType.POOR_FORMATTING,
                    description="Long prompt has no line breaks",
                    location="Entire prompt",
                    suggestion="Add line breaks to improve readability",
                )
            )

        # 7. Check for ambiguous instructions
        text_lower = text.lower()
        has_action_verb = any(verb in text_lower for verb in self.ACTION_VERBS)

        if not has_action_verb:
            issues.append(
                PromptIssue(
                    severity=IssueSeverity.WARNING,
                    type=IssueType.AMBIGUOUS_INSTRUCTIONS,
                    description="No clear action verbs found",
                    location="Entire prompt",
                    suggestion="Add specific action verbs like 'summarize', 'analyze', 'extract'",
                )
            )

        return issues

    def _generate_suggestions(
        self, prompt: Prompt, issues: List[PromptIssue]
    ) -> List[PromptSuggestion]:
        """Generate improvement suggestions based on analysis.

        Args:
            prompt: Analyzed prompt.
            issues: Detected issues.

        Returns:
            List of improvement suggestions.
        """
        suggestions: List[PromptSuggestion] = []
        text = prompt.text

        # Suggestion based on issues
        for issue in issues:
            if issue.type == IssueType.MISSING_STRUCTURE:
                suggestions.append(
                    PromptSuggestion(
                        type=SuggestionType.ADD_STRUCTURE,
                        description="Add section headers and bullet points for better organization",
                        priority=8,
                    )
                )
            elif issue.type == IssueType.VAGUE_LANGUAGE:
                suggestions.append(
                    PromptSuggestion(
                        type=SuggestionType.CLARIFY_INSTRUCTIONS,
                        description="Replace vague terms with specific, measurable criteria",
                        priority=9,
                    )
                )
            elif issue.type == IssueType.TOO_LONG:
                suggestions.append(
                    PromptSuggestion(
                        type=SuggestionType.REDUCE_LENGTH,
                        description="Remove redundant content and focus on key instructions",
                        priority=7,
                    )
                )
            elif issue.type == IssueType.POOR_FORMATTING:
                suggestions.append(
                    PromptSuggestion(
                        type=SuggestionType.IMPROVE_FORMATTING,
                        description="Add line breaks and proper formatting for readability",
                        priority=6,
                    )
                )

        # Additional suggestions based on content analysis
        if "example" not in text.lower() and len(text) > 100:
            suggestions.append(
                PromptSuggestion(
                    type=SuggestionType.ADD_EXAMPLES,
                    description="Include specific examples to clarify expected output format",
                    priority=7,
                )
            )

        # Sort by priority (highest first)
        suggestions.sort(key=lambda s: s.priority, reverse=True)

        return suggestions

    def _compute_metadata(self, prompt: Prompt) -> dict:
        """Compute additional analysis metadata.

        Args:
            prompt: Analyzed prompt.

        Returns:
            Dictionary of metadata metrics.
        """
        text = prompt.text
        words = text.split()
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "estimated_tokens": len(text) / 4,
            "avg_word_length": sum(len(w) for w in words) / max(1, len(words)),
            "avg_sentence_length": len(words) / max(1, len(sentences)),
            "variable_count": len(prompt.variables),
        }
