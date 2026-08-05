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
        source = str(item.get("source", ""))
        tests = item.get("tests")
        if re.fullmatch(r"INV-[A-Z]+-\d{3}", identifier) is None:
            errors.append(f"invalid invariant ID: {identifier}")
        if severity not in severities:
            errors.append(f"invalid severity for {identifier}: {severity}")
        if disposition not in dispositions:
            errors.append(f"invalid disposition for {identifier}: {disposition}")
        if re.fullmatch(r"PR\d+:(?:review-thread|review-memo):[A-Za-z0-9_.:-]+", source) is None:
            errors.append(f"invalid or missing review source for {identifier}: {source}")
        if not all(item.get(field) for field in ("family", "invariant", "finding", "owner")):
            errors.append(f"descriptive invariant fields are required for {identifier}")
        if not isinstance(tests, list) or not tests:
            errors.append(f"tests are required for {identifier}")
        elif not set(map(str, tests)) <= test_names:
            errors.append(f"unknown test reference for {identifier}")
        if severity in {"P0", "P1"} and disposition not in {
            "fixed",
            "invalid",
            "superseded",
        }:
            errors.append(f"merge-blocking invariant is not closed safely: {identifier}")
        if disposition == "accepted_limitation" and item.get("authority_effect") != (
            "false_negative_only"
        ):
            errors.append(
                f"accepted limitation must prove false-negative-only authority: {identifier}"
            )
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
        "source": "PR4:review-memo:registry-mutation",
        "tests": ["test_review_invariant_registry_rejects_schema_typos_that_bypass_policy"],
    }
    mutations = (
        {**base, "severity": "P01"},
        {**base, "severity": "p1"},
        {**base, "id": "INV-process-001"},
        {**base, "id": "INV-PROCESS-1"},
        {key: value for key, value in base.items() if key != "family"},
        {key: value for key, value in base.items() if key != "invariant"},
        {key: value for key, value in base.items() if key != "finding"},
    )

    assert _registry_errors([base], _test_names()) == []
    assert all(_registry_errors([mutation], _test_names()) for mutation in mutations)


def test_merge_blockers_cannot_be_blessed_as_accepted_limitations():
    allowed_limitation: dict[str, object] = {
        "id": "INV-PROCESS-002",
        "family": "review-process",
        "severity": "P3",
        "invariant": "Authority-increasing limitations cannot bypass readiness.",
        "finding": "Mutation fixture.",
        "disposition": "accepted_limitation",
        "authority_effect": "false_negative_only",
        "owner": "quality",
        "source": "PR4:review-memo:accepted-limitation-mutation",
        "tests": ["test_merge_blockers_cannot_be_blessed_as_accepted_limitations"],
    }

    assert _registry_errors([allowed_limitation], _test_names()) == []
    merge_blocker = {**allowed_limitation, "severity": "P1"}
    assert _registry_errors([merge_blocker], _test_names())


def test_accepted_limitations_require_false_negative_only_proof():
    allowed_limitation: dict[str, object] = {
        "id": "INV-PROCESS-003",
        "family": "review-process",
        "severity": "P3",
        "invariant": "Accepted syntax limitations cannot increase authority.",
        "finding": "Mutation fixture.",
        "disposition": "accepted_limitation",
        "authority_effect": "false_negative_only",
        "owner": "quality",
        "source": "PR4:review-memo:authority-effect-mutation",
        "tests": ["test_accepted_limitations_require_false_negative_only_proof"],
    }

    assert _registry_errors([allowed_limitation], _test_names()) == []
    limitation_without_proof = {
        key: value for key, value in allowed_limitation.items() if key != "authority_effect"
    }
    assert _registry_errors([limitation_without_proof], _test_names())
