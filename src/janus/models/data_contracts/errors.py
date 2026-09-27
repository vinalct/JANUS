"""Errors raised while loading an ODCS-shaped JANUS data contract."""

from __future__ import annotations

from pathlib import Path

from janus.models.config.issues import ValidationIssue


class ContractValidationError(ValueError):
    """All structural problems found while loading one contract file."""

    def __init__(self, contract_path: Path, issues: list[ValidationIssue]) -> None:
        self.contract_path = contract_path
        self.issues = tuple(sorted(issues, key=lambda issue: issue.path))
        super().__init__(
            "\n".join(
                f"{self.contract_path}: {issue.path}: {issue.message}"
                for issue in self.issues
            )
        )
