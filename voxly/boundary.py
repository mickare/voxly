from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Optional, Protocol, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from .typing import SupportsRichComparison


class Comparable(Protocol):
    @abstractmethod
    def __lt__(self, __other: Any) -> bool:
        ...

    @abstractmethod
    def __gt__(self, __other: Any) -> bool:
        ...


T = TypeVar("T", bound=SupportsRichComparison)
Tup3 = Tuple[T, T, T]


class Box(ABC, Generic[T]):
    """3D box"""

    @property
    @abstractmethod
    def min(self) -> Tup3[T]:
        ...

    @property
    @abstractmethod
    def max(self) -> Tup3[T]:
        ...

    @property
    @abstractmethod
    def minmax(self) -> Tuple[Tup3[T], Tup3[T]]:
        ...

    def __contains__(self, other: Tup3[T]) -> bool:
        return self.min <= other <= self.max


class ImmutableBox(Box[T], Generic[T]):
    __slots__ = ("_min", "_max")

    def __init__(self, a: Tup3[T], b: Tup3[T]) -> None:
        self._min: Tup3[T] = (min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]))
        self._max: Tup3[T] = (max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]))

    @classmethod
    def of(cls, points: Iterable[Tup3[T]] | npt.NDArray[np.int_]) -> "ImmutableBox[T]":
        pts: npt.NDArray[np.int_] = np.asarray(points, dtype=int)
        assert pts.ndim ==2 and pts.shape[1] == 3
        _min: Tup3[T] = tuple(np.min(pts, axis=0))  # type: ignore
        _max: Tup3[T] = tuple(np.max(pts, axis=0))  # type: ignore
        return cls(_min, _max)

    @classmethod
    def join(cls, *boxes: Box[T]) -> "ImmutableBox[T]":
        pts_min, pts_max = zip(*((b.min, b.max) for b in boxes))
        _min: Tup3[T] = tuple(np.min(pts_min, axis=0))  # type: ignore
        _max: Tup3[T] = tuple(np.max(pts_max, axis=0))  # type: ignore
        return cls(_min, _max)

    @property
    def min(self) -> Tup3[T]:
        return self._min

    @property
    def max(self) -> Tup3[T]:
        return self._max

    @property
    def minmax(self) -> Tuple[Tup3[T], Tup3[T]]:
        return self._min, self._max

    def merge(self, *other: Box[T]) -> "ImmutableBox[T]":
        if not other:
            return self
        return ImmutableBox.join(self, *other)


class UnsafeBox(Box[T], Generic[T]):
    """Unsafe 3D box that can have a dity state"""

    __slots__ = ("_min", "_max", "_dirty")

    def __init__(self) -> None:
        self._min: Optional[Tup3[T]] = None
        self._max: Optional[Tup3[T]] = None
        self._dirty = False

    def clear(self) -> None:
        self._min = None
        self._max = None
        self._dirty = False

    def update(self, other: Iterable[Tup3[T]] | npt.NDArray[np.int_]) -> None:
        indices: npt.NDArray[np.int_] = np.asarray(other)
        if len(indices):
            assert indices.ndim == 2 and indices.shape[1] == 3
            self._min = tuple(np.min(indices, axis=0))  # type: ignore
            self._max = tuple(np.max(indices, axis=0))  # type: ignore
            self._dirty = False

    def update_if_dirty(self, getter: Callable[[], Iterable[Tup3[T]] | npt.NDArray[np.int_]]) -> None:
        if self._dirty:
            self.update(getter())

    def add(self, index: Tup3[T]) -> None:
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
    def min(self) -> Tup3[T]:
        assert not self._dirty and self._min
        return self._min

    @property
    def max(self) -> Tup3[T]:
        assert not self._dirty and self._max
        return self._max

    @property
    def minmax(self) -> Tuple[Tup3[T], Tup3[T]]:
        assert not self._dirty and self._min and self._max
        return self._min, self._max

    def to_box(self) -> Box[T]:
        assert not self._dirty and self._min and self._max
        return ImmutableBox(self._min, self._max)

    def to_safe(self, getter: Callable[[], Iterable[Tup3[T]] | npt.NDArray[np.int_]]) -> Box[T]:
        self.update_if_dirty(getter)
        return self.to_box()
