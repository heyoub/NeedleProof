from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path


def _registry_errors(invariants: list[dict[str, object]], test_names: set[str]) -> list[str]:
    identifiers = [str(item.get("id", "")) for item in invariants]
    dispositions = {"open", "fixed", "invalid", "superseded", "accepted_limitation"}
    severities = {"P0", "P1", "P2", "P3"}
    errors: list[str] = []

    if len(identifiers) != len(set(identifiers)):
        errors.append("invariant IDs must be unique")
    for item in invariants:
        identifier = str(item.get("id", ""))
        severity = str(item.get("severity", ""))
        disposition = str(item.get("disposition", ""))
        tests = item.get("tests")
        if re.fullmatch(r"INV-[A-Z]+-\d{3}", identifier) is None:
            errors.append(f"invalid invariant ID: {identifier}")
        if severity not in severities:
            errors.append(f"invalid severity for {identifier}: {severity}")
        if disposition not in dispositions:
            errors.append(f"invalid disposition for {identifier}: {disposition}")
        if not item.get("owner") or not isinstance(tests, list) or not tests:
            errors.append(f"owner and tests are required for {identifier}")
        elif not set(map(str, tests)) <= test_names:
            errors.append(f"unknown test reference for {identifier}")
        if severity in {"P0", "P1"} and disposition == "open":
            errors.append(f"merge-blocking invariant remains open: {identifier}")
    return errors


def _test_names() -> set[str]:
    return {
        node.name
        for path in Path("tests").glob("test_*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def test_review_invariant_registry_is_complete_and_merge_adjudicated():
    registry = tomllib.loads(Path("docs/invariants.toml").read_text(encoding="utf-8"))
    assert _registry_errors(registry["invariant"], _test_names()) == []


def test_review_invariant_registry_rejects_schema_typos_that_bypass_policy():
    base: dict[str, object] = {
        "id": "INV-PROCESS-001",
        "family": "review-process",
        "severity": "P1",
        "invariant": "A registry typo cannot bypass a merge policy.",
        "finding": "Mutation fixture.",
        "disposition": "fixed",
        "owner": "quality",
        "tests": ["test_review_invariant_registry_rejects_schema_typos_that_bypass_policy"],
    }
    mutations = (
        {**base, "severity": "P01"},
        {**base, "severity": "p1"},
        {**base, "id": "INV-process-001"},
        {**base, "id": "INV-PROCESS-1"},
    )

    assert all(_registry_errors([mutation], _test_names()) for mutation in mutations)
