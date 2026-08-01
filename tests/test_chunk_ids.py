from __future__ import annotations

import json

import pytest
from needleproof_api.chunk_ids import chunk_id_from_uint64, chunk_id_to_uint64
from needleproof_api.main import QuoteChallengeRequest
from pydantic import ValidationError


def test_seeded_uint64_chunk_id_round_trips_as_an_opaque_json_string():
    numeric_id = 6812131146285660789
    opaque_id = chunk_id_from_uint64(numeric_id)
    request = QuoteChallengeRequest(
        run_id="run_0123456789abcdef0123456789abcdef",
        corpus_version="v_0123456789abcdef",
        chunk_id=opaque_id,
        quote="altered quotation",
    )

    payload = json.loads(request.model_dump_json())

    assert payload["chunk_id"] == opaque_id
    assert isinstance(payload["chunk_id"], str)
    assert chunk_id_to_uint64(payload["chunk_id"]) == numeric_id
    assert numeric_id > 2**53


def test_quote_challenge_rejects_numeric_chunk_ids_at_the_contract_boundary():
    with pytest.raises(ValidationError):
        QuoteChallengeRequest.model_validate(
            {
                "run_id": "run_0123456789abcdef0123456789abcdef",
                "corpus_version": "v_0123456789abcdef",
                "chunk_id": 6812131146285660789,
                "quote": "altered quotation",
            }
        )
