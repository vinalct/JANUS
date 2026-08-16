"""The collected-issue vocabulary every builder writes into.

Builders append to a shared ``list[ValidationIssue]`` and **never raise**, so one load
reports every problem in a file instead of the first. ``SourceConfig.from_mapping`` owns
the single raise site.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    path: str
    message: str

    def render(self) -> str:
        """Return the issue in the same path-first format used in validation errors."""
        return f"{self.path}: {self.message}"


class SourceConfigValidationError(ValueError):
    def __init__(self, config_path: Path, issues: list[ValidationIssue]) -> None:
        """Build a readable validation error for a single source config file."""
        self.config_path = config_path
        self.issues = tuple(issues)
        message_lines = [f"Invalid source config: {config_path}"]
        message_lines.extend(f"- {issue.render()}" for issue in self.issues)
        super().__init__("\n".join(message_lines))
