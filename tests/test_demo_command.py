from __future__ import annotations

import json
from pathlib import Path


def test_demo_loads_documented_dotenv_for_corpus_and_api_processes():
    scripts = json.loads(Path("package.json").read_text(encoding="utf-8"))["scripts"]

    assert "uv run --env-file .env" in scripts["corpus:ensure"]
    assert "uv run --env-file .env" in scripts["demo"]
