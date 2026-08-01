from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

import pytest
from needleproof_api.config import Settings
from needleproof_api.corpus import chunk_records_digest
from needleproof_api.retrieval import CorpusStore
from needleproof_api.util import canonical_json, sha256_file, sha256_text


def copied_corpus(tmp_path: Path) -> tuple[Settings, Path]:
    shutil.copytree(Path("data/corpora"), tmp_path / "corpora")
    settings = Settings(data_dir=tmp_path)
    pointer = json.loads((settings.corpora_dir / "current.json").read_text(encoding="utf-8"))
    return settings, settings.corpora_dir / pointer["corpus_version"]


def test_startup_rejects_modified_turbovec_artifact(tmp_path):
    settings, version_dir = copied_corpus(tmp_path)
    index_path = version_dir / "index.tvim"
    payload = bytearray(index_path.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match="TurboVec artifact digest"):
        CorpusStore(settings)


def test_startup_rejects_modified_sqlite_artifact(tmp_path):
    settings, version_dir = copied_corpus(tmp_path)
    with sqlite3.connect(version_dir / "corpus.sqlite3") as connection:
        connection.execute(
            "UPDATE chunks SET normalized_text = normalized_text || ' tampered' "
            "WHERE internal_id = (SELECT MIN(internal_id) FROM chunks)"
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    with pytest.raises(ValueError, match="SQLite .*digest"):
        CorpusStore(settings)


def test_startup_reconciles_index_id_set_not_only_length(tmp_path):
    settings, version_dir = copied_corpus(tmp_path)
    db_path = version_dir / "corpus.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE chunks SET chunk_external_id = 'chk_fffffffffffffffe' "
            "WHERE internal_id = (SELECT MIN(internal_id) FROM chunks)"
        )
        connection.commit()
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        ordered_digest = chunk_records_digest(connection)

    manifest_path = version_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["corpus_sqlite3_sha256"] = sha256_file(db_path)
    manifest["artifacts"]["ordered_chunk_records_sha256"] = ordered_digest
    unsigned = {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_sha256", "corpus_version", "built_at", "embedding_calls"}
    }
    digest = sha256_text(canonical_json(unsigned))
    new_version = f"v_{digest[:16]}"
    manifest["manifest_sha256"] = digest
    manifest["corpus_version"] = new_version
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    new_dir = settings.corpora_dir / new_version
    version_dir.rename(new_dir)
    pointer = {
        "corpus_id": manifest["corpus_id"],
        "corpus_version": new_version,
        "manifest_sha256": digest,
    }
    (settings.corpora_dir / "current.json").write_text(
        canonical_json(pointer) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="ID set"):
        CorpusStore(settings)


def test_current_pointer_digest_must_match_manifest(tmp_path):
    settings, _version_dir = copied_corpus(tmp_path)
    pointer_path = settings.corpora_dir / "current.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["manifest_sha256"] = "0" * 64
    pointer_path.write_text(canonical_json(pointer) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="pointer"):
        CorpusStore(settings)
