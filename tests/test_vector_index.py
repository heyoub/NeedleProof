from __future__ import annotations

import numpy as np
from needleproof_api.chunk_ids import chunk_id_from_uint64
from needleproof_api.vector_index import TurboVecAdapter


def test_turbovec_persistence_allowlist_and_delete(tmp_path):
    vectors = np.eye(8, dtype=np.float32)[:3]
    ids = [chunk_id_from_uint64(value) for value in (101, 202, 303)]
    index = TurboVecAdapter.create(dimensions=8, bit_width=4)
    index.add(vectors, ids)

    nearest = index.search(vectors[[0]], top_k=3)
    assert nearest[0][0] == ids[0]
    assert {item[0] for item in nearest} == set(ids)

    allowed = index.search(vectors[[0]], top_k=2, allowlist=ids[1:])
    assert {item[0] for item in allowed} == set(ids[1:])

    path = tmp_path / "adapter.tvim"
    index.write(path)
    loaded = TurboVecAdapter.load(path, dimensions=8, bit_width=4)
    assert len(loaded) == 3
    assert loaded.search(vectors[[1]], top_k=1)[0][0] == ids[1]

    loaded.remove(ids[1])
    assert len(loaded) == 2
    assert ids[1] not in {item[0] for item in loaded.search(vectors[[1]], top_k=2)}
