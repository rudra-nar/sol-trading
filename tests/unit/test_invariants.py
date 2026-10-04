"""Invariant tests -- enforce critical system-wide properties.

These tests guard the most important invariants that MUST NEVER be broken.
"""

from __future__ import annotations

import ast
from pathlib import Path


class TestNoLookaheadImports:
    """Invariant: research/ modules must NEVER be imported by agent/feature/pipeline code."""

    def test_no_research_imports_in_agents(self) -> None:
        """Scan all non-research .py files for 'from sol_ew.research' imports."""
        src_root = Path("src/sol_ew")
        violations: list[str] = []

        for py_file in src_root.rglob("*.py"):
            # Skip research/ itself
            if "research" in py_file.parts:
                continue

            source = py_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sol_ew.research"):
                    violations.append(
                        f"{py_file}:{node.lineno} imports {node.module}"
                    )

        assert violations == [], (
            "LOOKAHEAD LEAKAGE: research modules imported by agent code!\n"
            + "\n".join(violations)
        )

    def test_no_research_imports_in_tests_except_test_research(self) -> None:
        """Research imports in test files should only be in test_research.py."""
        test_root = Path("tests")
        violations: list[str] = []

        for py_file in test_root.rglob("*.py"):
            if py_file.name == "test_research.py":
                continue

            source = py_file.read_text(encoding="utf-8")
            try:
                tree = ast.parse(source)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sol_ew.research"):
                    violations.append(
                        f"{py_file}:{node.lineno} imports {node.module}"
                    )

        assert violations == [], (
            "Research modules should only be imported in test_research.py!\n"
            + "\n".join(violations)
        )
