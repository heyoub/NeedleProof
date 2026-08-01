from __future__ import annotations

import json
from pathlib import Path

from needleproof_api.corpus import deterministic_chunk_id, load_current_manifest
from needleproof_api.util import canonical_json, sha256_text


def test_manifest_is_canonical_and_seeded(settings):
    manifest = load_current_manifest(settings)
    assert manifest["embedding_model"] == "text-embedding-3-small"
    assert manifest["embedding_dimensions"] == 768
    assert manifest["embedding_l2_normalized"] is True
    assert manifest["turbovec_bit_width"] == 4
    assert manifest["document_count"] == 1
    assert manifest["chunk_count"] == 5
    assert manifest["documents"][0]["included_pages"] == [3, 4, 5, 6]


def test_answer_key_is_not_in_seeded_source_selection():
    source = json.loads(Path("data/corpus-source.json").read_text(encoding="utf-8"))
    assert source["documents"][0]["include_pages"] == [3, 4, 5, 6]
    assert 7 not in source["documents"][0]["include_pages"]


def test_external_chunk_ids_are_unsigned_deterministic():
    first = deterministic_chunk_id("doc_example", 6, 0)
    second = deterministic_chunk_id("doc_example", 6, 0)
    changed = deterministic_chunk_id("doc_example", 6, 1)
    assert first == second
    assert first != changed
    assert 0 <= first <= (2**64 - 1)


def test_canonical_json_digest_is_key_order_independent():
    assert sha256_text(canonical_json({"b": 2, "a": 1})) == sha256_text(
        canonical_json({"a": 1, "b": 2})
    )
