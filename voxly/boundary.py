from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Optional, Protocol, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from .typing import SupportsDunderGT, SupportsDunderLT


class Comparable(Protocol):
    @abstractmethod
    def __lt__(self, __other: Any) -> bool:
        ...

    @abstractmethod
    def __gt__(self, __other: Any) -> bool:
        ...


_T = TypeVar("_T", bound=(SupportsDunderLT | SupportsDunderGT))
_Tup3 = Tuple[_T, _T, _T]


class Box(ABC, Generic[_T]):
    """3D box"""

    @property
    @abstractmethod
    def min(self) -> _Tup3[_T]:
        ...

    @property
    @abstractmethod
    def max(self) -> _Tup3[_T]:
        ...

    @property
    @abstractmethod
    def minmax(self) -> Tuple[_Tup3[_T], _Tup3[_T]]:
        ...

    def __contains__(self, other: _Tup3[_T]) -> bool:
        return self.min <= other <= self.max


class ImmutableBox(Box[_T], Generic[_T]):
    __slots__ = ("_min", "_max")

    def __init__(self, a: _Tup3[_T], b: _Tup3[_T]) -> None:
        self._min: _Tup3[_T] = (min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]))
        self._max: _Tup3[_T] = (max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]))

    @classmethod
    def of(cls, points: Iterable[_Tup3[_T]] | npt.NDArray[np.int_]) -> "ImmutableBox[_T]":
        pts: npt.NDArray[np.int_] = np.asarray(points, dtype=int)
        assert pts.ndim ==2 and pts.shape[1] == 3
        _min: _Tup3[_T] = tuple(np.min(pts, axis=0))  # type: ignore
        _max: _Tup3[_T] = tuple(np.max(pts, axis=0))  # type: ignore
        return cls(_min, _max)

    @classmethod
    def join(cls, *boxes: Box[_T]) -> "ImmutableBox[_T]":
        pts_min, pts_max = zip(*((b.min, b.max) for b in boxes))
        _min: _Tup3[_T] = tuple(np.min(pts_min, axis=0))  # type: ignore
        _max: _Tup3[_T] = tuple(np.max(pts_max, axis=0))  # type: ignore
        return cls(_min, _max)

    @property
    def min(self) -> _Tup3[_T]:
        return self._min

    @property
    def max(self) -> _Tup3[_T]:
        return self._max

    @property
    def minmax(self) -> Tuple[_Tup3[_T], _Tup3[_T]]:
        return self._min, self._max

    def merge(self, *other: Box[_T]) -> "ImmutableBox[_T]":
        if not other:
            return self
        return ImmutableBox.join(self, *other)


class UnsafeBox(Box[_T], Generic[_T]):
    """Unsafe 3D box that can have a dity state"""

    __slots__ = ("_min", "_max", "_dirty")

    def __init__(self) -> None:
        self._min: Optional[_Tup3[_T]] = None
        self._max: Optional[_Tup3[_T]] = None
        self._dirty = False

    def clear(self) -> None:
        self._min = None
        self._max = None
        self._dirty = False

    def update(self, other: Iterable[_Tup3[_T]] | npt.NDArray[np.int_]) -> None:
        indices: npt.NDArray[np.int_] = np.asarray(other)
        if len(indices):
            assert indices.ndim == 2 and indices.shape[1] == 3
            self._min = tuple(np.min(indices, axis=0))  # type: ignore
            self._max = tuple(np.max(indices, axis=0))  # type: ignore
            self._dirty = False

    def update_if_dirty(self, getter: Callable[[], Iterable[_Tup3[_T]] | npt.NDArray[np.int_]]) -> None:
        if self._dirty:
            self.update(getter())

    def add(self, index: _Tup3[_T]) -> None:
        if self._min is None:
            self._min = tuple(np.asarray(index, dtype=int))  # type: ignore
        else:
            self._min = tuple(np.min((self._min, index), axis=0)) # type: ignore
        if self._max is None:
            self._max = tuple(np.asarray(index, dtype=int)) # type: ignore
        else:
            self._max = tuple(np.max((self._max, index), axis=0)) # type: ignore

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_dirty(self) -> None:
        self._dirty = True

    @property
    def min(self) -> _Tup3[_T]:
        assert not self._dirty and self._min
        return self._min

    @property
    def max(self) -> _Tup3[_T]:
        assert not self._dirty and self._max
        return self._max

    @property
    def minmax(self) -> Tuple[_Tup3[_T], _Tup3[_T]]:
        assert not self._dirty and self._min and self._max
        return self._min, self._max

    def to_box(self) -> Box[_T]:
        assert not self._dirty and self._min and self._max
        return ImmutableBox(self._min, self._max)

    def to_safe(self, getter: Callable[[], Iterable[_Tup3[_T]] | npt.NDArray[np.int_]]) -> Box[_T]:
        self.update_if_dirty(getter)
        return self.to_box()
