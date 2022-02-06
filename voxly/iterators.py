from typing import Any, Protocol, TypeVar, Union, Tuple, Iterator, Optional, Iterable, Callable, overload

import numpy as np
import numpy.typing as npt

from .typing import Vec3i, Index3

SliceOpt = Union[int, slice, None]


def to_slice(s: SliceOpt = None) -> slice:
    if isinstance(s, slice):
        return s
    if s is None:
        return slice(s)
    return slice(s, s + 1)


class SlicedRangeIterator:
    """1D Slice Iterator"""

    @classmethod
    def _indices(cls, low: int, high: int, s: slice, clip: bool = True) -> Tuple[int, int, int]:
        step = s.step or 1
        start = low if s.start is None else s.start
        stop = high if s.stop is None else s.stop
        if clip:
            start = max(start, low + (start - low) % step)
            stop = min(stop, high)
        else:
            start = start
            stop = stop
        return start, stop, step

    def __init__(self, low: int, high: int, s: SliceOpt, clip: bool = True):
        self._low = int(low)
        self._high = int(high)
        self._slice = to_slice(s)
        self._start, self._stop, self._step = self._indices(low, high, self._slice, clip)
        self.clip = clip

    def range(self) -> range:
        return range(self._start, self._stop, self._step)

    def __contains__(self, item: Any) -> bool:
        if isinstance(item, int):
            return self._start <= item < self._stop and (item % self._step) == (self._start % self._step)
        return False

    def __iter__(self) -> Iterator[int]:
        yield from range(self._start, self._stop, self._step)

    def __len__(self) -> int:
        ds = self._stop - self._start
        return max(0, ds // self._step, ds % self._step > 0)

    @property
    def low(self) -> int:
        return self._low

    @property
    def high(self) -> int:
        return self._high

    @property
    def slice(self) -> slice:
        return self._slice

    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop

    @property
    def step(self) -> int:
        return self._step

    def __floordiv__(self, other: int) -> "SlicedRangeIterator":
        high = -(-self._high // other)
        slice_stop = -(-self._stop // other)
        return SlicedRangeIterator(
            self._low // other,
            high,
            slice(self._start // other, slice_stop, max(1, self._step // other)),
            self.clip,
        )


class VoxelGridIterator:
    """3D Slice Iterator"""

    @classmethod
    def require_bounded(cls, x: SliceOpt, y: SliceOpt, z: SliceOpt) -> "VoxelGridIterator":
        x = to_slice(x)
        y = to_slice(y)
        z = to_slice(z)
        assert x.start is not None and x.stop is not None
        assert y.start is not None and y.stop is not None
        assert z.start is not None and z.stop is not None
        return cls((x.start, y.start, z.start), (x.stop, y.stop, z.stop), x, y, z)  # type: ignore

    @classmethod
    def empty(cls) -> "VoxelGridIterator":
        return cls(np.zeros(3), np.zeros(3), 0, 0, 0)  # type: ignore

    def __init__(
        self, low: Vec3i, high: Vec3i, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None, clip: bool = True
    ):
        self._low: Vec3i = np.asarray(low, dtype=int)
        self._high: Vec3i = np.asarray(high, dtype=int)
        assert self._low.shape == (3,) and self._high.shape == (3,)
        self._x = SlicedRangeIterator(self._low[0], self._high[0], x, clip)
        self._y = SlicedRangeIterator(self._low[1], self._high[1], y, clip)
        self._z = SlicedRangeIterator(self._low[2], self._high[2], z, clip)
        self.clip = clip

    def __contains__(self, item: Vec3i) -> bool:
        if len(item) == 3:
            return item[0] in self._x and item[1] in self._y and item[2] in self._z
        return False

    def iter_with_indices(self) -> Iterator[Tuple[Index3, Index3]]:
        for i, u in enumerate(self._x.range()):
            for j, v in enumerate(self._y.range()):
                for k, w in enumerate(self._z.range()):
                    yield (i, j, k), (u, v, w)

    def iter(self) -> Iterator[Index3]:
        for u in self._x.range():
            for v in self._y.range():
                for w in self._z.range():
                    yield u, v, w

    def __iter__(self) -> Iterator[Index3]:
        return self.iter()

    def __len__(self) -> int:
        return len(self._x) * len(self._y) * len(self._z)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return len(self._x), len(self._y), len(self._z)

    @property
    def low(self) -> Vec3i:
        return self._low

    @property
    def high(self) -> Vec3i:
        return self._high

    @property
    def x(self) -> SlicedRangeIterator:
        return self._x

    @property
    def y(self) -> SlicedRangeIterator:
        return self._y

    @property
    def z(self) -> SlicedRangeIterator:
        return self._z

    @property
    def start(self) -> Vec3i:
        return np.asarray((self._x.start, self._y.start, self._z.start), dtype=int)

    @property
    def stop(self) -> Vec3i:
        return np.asarray((self._x.stop, self._y.stop, self._z.stop), dtype=int)

    @property
    def step(self) -> Vec3i:
        return np.asarray((self._x.step, self._y.step, self._z.step), dtype=int)

    def __floordiv__(self, other: int) -> "VoxelGridIterator":
        x = self._x.__floordiv__(other)
        y = self._y.__floordiv__(other)
        z = self._z.__floordiv__(other)
        high = -(-self._high // other)
        return VoxelGridIterator(self._low // other, high, x.slice, y.slice, z.slice, self.clip)
