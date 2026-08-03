from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from needleproof_api.chunk_ids import chunk_id_to_uint64
from needleproof_api.config import Settings
from needleproof_api.corpus import (
    CorpusBuilder,
    chunk_page,
    deterministic_chunk_id,
    l2_normalize,
    load_current_manifest,
)
from needleproof_api.util import canonical_json, estimate_tokens, sha256_text


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
    assert first.startswith("chk_")
    assert len(first) == 20
    numeric = chunk_id_to_uint64(first)
    assert numeric > 2**53
    assert 0 <= numeric <= (2**64 - 1)


def test_canonical_json_digest_is_key_order_independent():
    assert sha256_text(canonical_json({"b": 2, "a": 1})) == sha256_text(
        canonical_json({"a": 1, "b": 2})
    )


def test_oversized_paragraph_is_split_into_bounded_overlapping_chunks():
    paragraph = " ".join(f"token-{index}" for index in range(4000))
    chunks = chunk_page(paragraph)
    assert len(chunks) > 1
    assert all(estimate_tokens(chunk) <= 710 for chunk in chunks)
    assert set(chunks[0].split()[-20:]) & set(chunks[1].split()[:100])


def test_medium_paragraph_overlap_does_not_exceed_chunk_maximum():
    first = " ".join(f"alpha-{index}" for index in range(400))
    second = " ".join(f"bravo-{index}" for index in range(400))

    chunks = chunk_page(f"{first}\n\n{second}")

    assert len(chunks) == 2
    assert all(estimate_tokens(chunk) <= 700 for chunk in chunks)
    assert set(chunks[0].split()[-20:]) & set(chunks[1].split()[:100])


def configured_builder(tmp_path, documents):
    source_path = tmp_path / "source.json"
    source_path.write_text(
        json.dumps(
            {
                "corpus_id": "builder-regression",
                "display_name": "Builder regression",
                "documents": documents,
            }
        ),
        encoding="utf-8",
    )
    settings = Settings(
        data_dir=tmp_path,
        corpus_source=source_path,
        embedding_dimensions=8,
    )
    builder = CorpusBuilder(settings)

    def embed(texts):
        vectors = np.ones((len(texts), 8), dtype=np.float32)
        return l2_normalize(vectors), []

    builder._embed = embed  # type: ignore[method-assign]
    return builder, settings


def test_duplicate_source_selection_fails_clearly(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("A sufficiently useful source paragraph.", encoding="utf-8")
    definition = {"path": str(source)}
    builder, _settings = configured_builder(tmp_path, [definition, definition])
    with pytest.raises(ValueError, match="Duplicate corpus source"):
        builder.build()


def test_corrupt_existing_immutable_version_is_not_silently_reused(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text(
        "A deterministic source paragraph for immutable build testing.",
        encoding="utf-8",
    )
    builder, settings = configured_builder(tmp_path, [{"path": str(source)}])
    summary = builder.build()
    index_path = settings.corpora_dir / summary.corpus_version / "index.tvim"
    payload = bytearray(index_path.read_bytes())
    payload[-1] ^= 0x01
    index_path.write_bytes(payload)

    with pytest.raises(ValueError, match="immutable corpus version.*corrupt"):
        builder.build()


@pytest.mark.parametrize("artifact", ["manifest", "document"])
def test_corrupt_existing_manifest_or_document_is_not_silently_reused(tmp_path, artifact):
    source = tmp_path / "source.txt"
    source.write_text(
        "A deterministic source paragraph for complete artifact validation.",
        encoding="utf-8",
    )
    builder, settings = configured_builder(tmp_path, [{"path": str(source)}])
    summary = builder.build()
    version_dir = settings.corpora_dir / summary.corpus_version
    if artifact == "manifest":
        manifest_path = version_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["display_name"] = "Tampered but self-reported as valid"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        document_path = next((version_dir / "documents").iterdir())
        document_path.write_text("tampered source artifact", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable corpus version.*corrupt"):
        builder.build()
