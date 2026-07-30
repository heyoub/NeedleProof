from __future__ import annotations

from pathlib import Path

import numpy as np
from turbovec import IdMapIndex


class TurboVecAdapter:
    """Narrow, testable boundary around TurboVec's alpha Python API."""

    def __init__(self, index: IdMapIndex, *, dimensions: int, bit_width: int):
        self._index = index
        self.dimensions = dimensions
        self.bit_width = bit_width

    @classmethod
    def create(cls, *, dimensions: int, bit_width: int = 4) -> TurboVecAdapter:
        return cls(
            IdMapIndex(dim=dimensions, bit_width=bit_width),
            dimensions=dimensions,
            bit_width=bit_width,
        )

    @classmethod
    def load(cls, path: Path, *, dimensions: int, bit_width: int = 4) -> TurboVecAdapter:
        return cls(IdMapIndex.load(str(path)), dimensions=dimensions, bit_width=bit_width)

    def add(self, vectors: np.ndarray, external_ids: np.ndarray) -> None:
        self._index.add_with_ids(
            np.ascontiguousarray(vectors, dtype=np.float32),
            np.ascontiguousarray(external_ids, dtype=np.uint64),
        )

    def search(
        self, query: np.ndarray, *, top_k: int, allowlist: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        query = np.ascontiguousarray(query, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != self.dimensions:
            raise ValueError(f"Expected query shape (n, {self.dimensions}), received {query.shape}")
        kwargs = {}
        if allowlist is not None:
            kwargs["allowlist"] = np.ascontiguousarray(allowlist, dtype=np.uint64)
        scores, ids = self._index.search(query, top_k, **kwargs)
        return [
            (int(chunk_id), float(score)) for score, chunk_id in zip(scores[0], ids[0], strict=True)
        ]

    def remove(self, external_id: int) -> None:
        self._index.remove(external_id)

    def write(self, path: Path) -> None:
        self._index.write(str(path))

    def __len__(self) -> int:
        return len(self._index)
