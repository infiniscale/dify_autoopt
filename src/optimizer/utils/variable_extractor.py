"""Centralized variable extraction supporting all Dify syntax."""
import re
from typing import List


class VariableExtractor:
    """Extract Jinja2/Dify variables from prompt text.

    Supports all Dify variable syntax variants:
    - Standard: {{variable}}
    - Nested: {{obj.field}}
    - Context: {{#context#}}
    - User: {{@user}}
    - System: {{$sys}}
    """

    # Pre-compile patterns for performance
    _PATTERNS = [
        re.compile(r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\}\}'),  # {{var}} or {{var.field}}
        re.compile(r'\{\{\s*#([a-zA-Z_][a-zA-Z0-9_\.]*)#\s*\}\}'),  # {{#context#}}
        re.compile(r'\{\{\s*@([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\}\}'),   # {{@user}}
        re.compile(r'\{\{\s*\$([a-zA-Z_][a-zA-Z0-9_\.]*)\s*\}\}'),  # {{$sys}}
    ]

    @classmethod
    def extract(cls, text: str) -> List[str]:
        """Extract all variables from text.

        Supports:
        - Standard: {{variable}}
        - Nested: {{obj.field}}
        - Context: {{#context#}}
        - User: {{@user.name}}
        - System: {{$sys.query}}

        Args:
            text: Prompt text to extract variables from

        Returns:
            List of unique variable names (preserving order)
        """
        seen = set()
        variables = []

        for pattern in cls._PATTERNS:
            for match in pattern.finditer(text):
                var = match.group(1)
                if var not in seen:
                    seen.add(var)
                    variables.append(var)

        return variables

    @classmethod
    def validate_variables(cls, text: str, required: List[str]) -> List[str]:
        """Check for missing required variables.

        Args:
            text: Optimized prompt text
            required: List of variables that must be present

        Returns:
            List of missing variable names
        """
        found = set(cls.extract(text))
        required_set = set(required)
        return list(required_set - found)
