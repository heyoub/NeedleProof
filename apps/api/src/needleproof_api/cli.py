from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import Settings
from .corpus import CorpusBuilder, load_current_manifest


def _corpus_build(settings: Settings) -> int:
    summary = CorpusBuilder(settings).build()
    print(summary.model_dump_json(indent=2))
    return 0


def _corpus_ensure(settings: Settings) -> int:
    try:
        manifest = load_current_manifest(settings)
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return _corpus_build(settings)
    print(
        f"Corpus ready: {manifest['corpus_version']} "
        f"({manifest['chunk_count']} chunks, {manifest['manifest_sha256'][:12]}…)"
    )
    return 0


async def _model_smoke(settings: Settings) -> int:
    from .main import verify_model_access

    await verify_model_access(settings)
    print(f"Model access confirmed: {settings.model}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="needleproof")
    subparsers = parser.add_subparsers(dest="command", required=True)
    corpus = subparsers.add_parser("corpus")
    corpus.add_argument("action", choices=["build", "ensure"])
    subparsers.add_parser("model-smoke")
    args = parser.parse_args()
    settings = Settings()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    if args.command == "corpus":
        return _corpus_build(settings) if args.action == "build" else _corpus_ensure(settings)
    if args.command == "model-smoke":
        return asyncio.run(_model_smoke(settings))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
