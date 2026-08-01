from __future__ import annotations

import numpy as np
from needleproof_api.vector_index import TurboVecAdapter


def test_turbovec_persistence_allowlist_and_delete(tmp_path):
    vectors = np.eye(8, dtype=np.float32)[:3]
    ids = np.asarray([101, 202, 303], dtype=np.uint64)
    index = TurboVecAdapter.create(dimensions=8, bit_width=4)
    index.add(vectors, ids)

    nearest = index.search(vectors[[0]], top_k=3)
    assert nearest[0][0] == 101
    assert {item[0] for item in nearest} == {101, 202, 303}

    allowed = index.search(vectors[[0]], top_k=2, allowlist=np.asarray([202, 303], dtype=np.uint64))
    assert {item[0] for item in allowed} == {202, 303}

    path = tmp_path / "adapter.tvim"
    index.write(path)
    loaded = TurboVecAdapter.load(path, dimensions=8, bit_width=4)
    assert len(loaded) == 3
    assert loaded.search(vectors[[1]], top_k=1)[0][0] == 202

    loaded.remove(202)
    assert len(loaded) == 2
    assert 202 not in {item[0] for item in loaded.search(vectors[[1]], top_k=2)}
