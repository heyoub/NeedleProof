from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .config import Settings
from .corpus import CorpusBuilder
from .golden import run_golden_retrieval
from .receipt import validate_receipt
from .retrieval import CorpusStore


def _corpus_build(settings: Settings) -> int:
    summary = CorpusBuilder(settings).build()
    print(summary.model_dump_json(indent=2))
    return 0


def _corpus_ensure(settings: Settings) -> int:
    try:
        manifest = CorpusStore(settings).manifest
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        return _corpus_build(settings)
    print(
        f"Corpus ready: {manifest['corpus_version']} "
        f"({manifest['chunk_count']} chunks, {manifest['manifest_sha256'][:12]}…)"
    )
    return 0


async def _model_smoke(settings: Settings) -> int:
    from .readiness import ModelAvailability

    await ModelAvailability(settings, ttl_seconds=0).require()
    print(f"Model access confirmed: {settings.model}")
    return 0


async def _golden(settings: Settings) -> int:
    report = await run_golden_retrieval(CorpusStore(settings))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["passed"] else 1


def _rehearsal(settings: Settings) -> int:
    if not settings.rehearsal_path.exists():
        print(f"Rehearsal receipt is missing: {settings.rehearsal_path}")
        return 1
    receipt = json.loads(settings.rehearsal_path.read_text(encoding="utf-8"))
    errors = validate_receipt(receipt)
    statuses = {claim["status"] for claim in receipt.get("claims", [])}
    if "conflict" not in statuses:
        errors.append("Rehearsal receipt does not contain the planted conflict.")
    if "date_variant" not in statuses:
        errors.append("Rehearsal receipt does not contain the AUM date variant.")
    if receipt.get("status") != "completed":
        errors.append("Rehearsal receipt is not a completed live run.")
    report = {
        "path": str(settings.rehearsal_path),
        "run_id": receipt.get("run_id"),
        "receipt_sha256": receipt.get("receipt_sha256"),
        "claim_count": len(receipt.get("claims", [])),
        "event_count": len(receipt.get("events", [])),
        "openai_call_count": len(receipt.get("openai_calls", [])),
        "valid": not errors,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


def main() -> int:
    parser = argparse.ArgumentParser(prog="needleproof")
    subparsers = parser.add_subparsers(dest="command", required=True)
    corpus = subparsers.add_parser("corpus")
    corpus.add_argument("action", choices=["build", "ensure"])
    subparsers.add_parser("model-smoke")
    subparsers.add_parser("golden")
    subparsers.add_parser("rehearsal")
    args = parser.parse_args()
    settings = Settings()
    Path(settings.data_dir).mkdir(parents=True, exist_ok=True)
    if args.command == "corpus":
        return _corpus_build(settings) if args.action == "build" else _corpus_ensure(settings)
    if args.command == "model-smoke":
        return asyncio.run(_model_smoke(settings))
    if args.command == "golden":
        return asyncio.run(_golden(settings))
    if args.command == "rehearsal":
        return _rehearsal(settings)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
