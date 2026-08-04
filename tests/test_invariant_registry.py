from __future__ import annotations

import ast
import tomllib
from pathlib import Path


def test_review_invariant_registry_is_complete_and_merge_adjudicated():
    registry = tomllib.loads(Path("docs/invariants.toml").read_text(encoding="utf-8"))
    invariants = registry["invariant"]
    identifiers = [item["id"] for item in invariants]
    dispositions = {"open", "fixed", "invalid", "superseded", "accepted_limitation"}
    test_names = {
        node.name
        for path in Path("tests").glob("test_*.py")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }

    assert len(identifiers) == len(set(identifiers))
    assert all(item["disposition"] in dispositions for item in invariants)
    assert all(item["owner"] and item["tests"] for item in invariants)
    assert all(set(item["tests"]) <= test_names for item in invariants)
    assert not any(
        item["severity"] in {"P0", "P1"} and item["disposition"] == "open" for item in invariants
    )
