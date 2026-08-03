from __future__ import annotations

import re
from typing import Annotated, TypeAlias

from pydantic import StringConstraints

CHUNK_ID_PATTERN = re.compile(r"^chk_[0-9a-f]{16}$")
ChunkId: TypeAlias = Annotated[
    str,
    StringConstraints(pattern=CHUNK_ID_PATTERN.pattern),
]


def chunk_id_from_uint64(value: int) -> str:
    if value < 0 or value > 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("Chunk ID must fit in an unsigned 64-bit integer")
    return f"chk_{value:016x}"


def chunk_id_to_uint64(chunk_id: str) -> int:
    if not CHUNK_ID_PATTERN.fullmatch(chunk_id):
        raise ValueError(f"Invalid opaque chunk ID: {chunk_id!r}")
    return int(chunk_id[4:], 16)
