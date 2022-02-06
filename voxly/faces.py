import enum
import itertools
from typing import List, Optional, Tuple, Union, Sequence, Iterator

import numpy as np
import numpy.typing as npt

from .typing import Index3, Vec3i


# Static face directions
CHUNK_DIRECTIONS: npt.NDArray[np.int_] = np.array(
    [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=int
)
CHUNK_DIRECTIONS.flags.writeable = False

IoS = int | slice

# Static face slices
__CSNONE = slice(None)
CHUNK_SLICE_DEFAULT: List[Tuple[IoS, IoS, IoS]] = [
    (-1, __CSNONE, __CSNONE),
    (0, __CSNONE, __CSNONE),
    (__CSNONE, -1, __CSNONE),
    (__CSNONE, 0, __CSNONE),
    (__CSNONE, __CSNONE, -1),
    (__CSNONE, __CSNONE, 0),
]


class ChunkFace(enum.IntEnum):
    """Chunk or Voxel Face"""

    NORTH = 0  # +X
    SOUTH = 1  # -X
    TOP = 2  # +Y
    BOTTOM = 3  # -Y
    EAST = 4  # +Z
    WEST = 5  # -Z

    def direction(self) -> Vec3i:
        return CHUNK_DIRECTIONS[self]  # type: ignore

    def _flip(self) -> int:
        return (self // 2) * 2 + ((self + 1) % 2)

    def flip(self) -> "ChunkFace":
        return ChunkFace(self._flip())

    def slice(self, width: int = -1, other: Optional[slice] = None) -> Tuple[IoS, IoS, IoS]:
        _other_none = other is None
        _width_default = width == -1
        if _other_none and _width_default:  # Fast default case
            return CHUNK_SLICE_DEFAULT[self]
        else:  # Slow special case
            s = slice(None) if _other_none else other
            s0: IoS = -1
            s1: IoS = 0
            if not _width_default:
                s0 = slice(-width, None)
                s1 = slice(None, width)
            return ((s0, s, s), (s1, s, s), (s, s0, s), (s, s1, s), (s, s, s0), (s, s, s1))[self]  # type: ignore

    def shape(self, size: int) -> Tuple[int, int, int]:
        return ((1, size, size), (1, size, size), (size, 1, size), (size, 1, size), (size, size, 1), (size, size, 1))[
            self
        ]

    def orthogonal(self) -> Sequence["ChunkFace"]:
        return (
            (ChunkFace.TOP, ChunkFace.BOTTOM, ChunkFace.EAST, ChunkFace.WEST),
            (ChunkFace.NORTH, ChunkFace.SOUTH, ChunkFace.EAST, ChunkFace.WEST),
            (ChunkFace.NORTH, ChunkFace.SOUTH, ChunkFace.TOP, ChunkFace.BOTTOM),
        )[self // 2]

    def __bool__(self) -> bool:
        return True

    @classmethod
    def edges(cls) -> Iterator[Tuple["ChunkFace", "ChunkFace"]]:
        return ((a, b) for a, b in itertools.permutations(ChunkFace, 2) if a // 2 != b // 2)

    @classmethod
    def corners(cls) -> Iterator[Tuple["ChunkFace", "ChunkFace", "ChunkFace"]]:
        return itertools.product(
            (ChunkFace.NORTH, ChunkFace.SOUTH), (ChunkFace.TOP, ChunkFace.BOTTOM), (ChunkFace.EAST, ChunkFace.WEST)
        )

    @classmethod
    def corner_slice(cls, x: "ChunkFace", y: "ChunkFace", z: "ChunkFace", width: int = -1) -> Tuple[IoS, IoS, IoS]:
        s = slice(None)
        s0: IoS = -1
        s1: IoS = 0
        if width >= 0:
            s0 = slice(-1 - width, -1)
            s1 = slice(0, width)
        u = x % 2 == 0
        v = y % 2 == 0
        w = z % 2 == 0
        return (s0 if u else s1, s0 if v else s1, s0 if w else s1)

    @classmethod
    def corner_direction(cls, x: "ChunkFace", y: "ChunkFace", z: "ChunkFace") -> Vec3i:
        return x.direction() + y.direction() + z.direction()
