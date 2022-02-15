"""
Module that contains the bounding box classes
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, Iterable, Optional, Protocol, Sequence, Tuple, Type, TypeVar

import numpy as np
import numpy.typing as npt

from .typing import SupportsDunderGT, SupportsDunderLT


_T = TypeVar("_T", bound=(SupportsDunderLT | SupportsDunderGT))
_Tup3 = Tuple[_T, _T, _T]


class BoundingBox(ABC, Generic[_T]):
    """Bounding box for 3D space"""

    @property
    @abstractmethod
    def min(self) -> _Tup3[_T]:
        """Minimum point"""
        ...

    @property
    @abstractmethod
    def max(self) -> _Tup3[_T]:
        """Maximum point"""
        ...

    @property
    @abstractmethod
    def minmax(self) -> Tuple[_Tup3[_T], _Tup3[_T]]:
        """Tuple of both minimum and maximum point"""
        ...

    def __contains__(self, other: _Tup3[_T] | Sequence[_T] | npt.NDArray[Any]) -> bool:
        """Check if a point is inside this bounding box"""
        t: _Tup3[_T] = other if isinstance(other, tuple) else tuple(other)  # type: ignore
        return self.min <= t <= self.max


class ImmutableBoundingBox(BoundingBox[_T], Generic[_T]):
    """
    Bounding box for 3D space that is immutable.
    """
    __slots__ = ("_min", "_max")

    def __init__(self, a: _Tup3[_T], b: _Tup3[_T]) -> None:
        self._min: _Tup3[_T] = (min(a[0], b[0]), min(a[1], b[1]), min(a[2], b[2]))
        self._max: _Tup3[_T] = (max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]))

    @classmethod
    def create(cls, points: Iterable[_Tup3[_T]] | npt.NDArray[np.int_]) -> "ImmutableBoundingBox[_T]":
        """Create from a sequence of points"""
        pts: npt.NDArray[np.int_] = np.asarray(points, dtype=int)
        assert pts.ndim ==2 and pts.shape[1] == 3
        _min: _Tup3[_T] = tuple(np.min(pts, axis=0))  # type: ignore
        _max: _Tup3[_T] = tuple(np.max(pts, axis=0))  # type: ignore
        return cls(_min, _max)

    @classmethod
    def join(cls, *boxes: BoundingBox[_T]) -> "ImmutableBoundingBox[_T]":
        """Join multiple boxes to a new one"""
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

    def merge(self, *other: BoundingBox[_T]) -> "ImmutableBoundingBox[_T]":
        """Merge this bounding box with multiple"""
        if not other:
            return self
        return ImmutableBoundingBox.join(self, *other)


class UnsafeBoundingBox(BoundingBox[_T], Generic[_T]):
    """
    Unsafe bounding box for 3D that has a dirty state for lazy box calculation.
    """

    __slots__ = ("_min", "_max", "_dirty")

    def __init__(self, dtype: Type[_T] | npt.DTypeLike) -> None:
        self._dtype = np.dtype(dtype)
        self._min: Optional[_Tup3[_T]] = None
        self._max: Optional[_Tup3[_T]] = None
        self._dirty = False

    def clear(self) -> None:
        """Clear the bounding box"""
        self._min = None
        self._max = None
        self._dirty = False

    def set(self, points: Iterable[_Tup3[_T]] | npt.ArrayLike) -> None:
        """Set the bounding box to a box of the points"""
        indices: npt.NDArray[Any] = np.asarray(points, self._dtype)
        if len(indices):
            assert indices.ndim == 2 and indices.shape[1] == 3
            self._min = tuple(np.min(indices, axis=0))  # type: ignore
            self._max = tuple(np.max(indices, axis=0))  # type: ignore
            self._dirty = False
        else:
            self._min = None
            self._max = None
            self._dirty = False

    def update_if_dirty(self, callback: Callable[[], Iterable[_Tup3[_T]] | npt.NDArray[np.int_]]) -> None:
        """Update this box with the return value of the callback if the box is marked dirty."""
        if self._dirty:
            self.set(callback())

    def add(self, point: _Tup3[_T]) -> None:
        """Add a single point to the box"""
        if self._min is None:
            self._min = tuple(np.asarray(point, dtype=self._dtype))  # type: ignore
        else:
            self._min = tuple(np.min((self._min, point), axis=0)) # type: ignore
        if self._max is None:
            self._max = tuple(np.asarray(point, dtype=self._dtype)) # type: ignore
        else:
            self._max = tuple(np.max((self._max, point), axis=0)) # type: ignore

    @property
    def dirty(self) -> bool:
        """Flag if state is dirty"""
        return self._dirty

    def mark_dirty(self) -> None:
        """Mark the state dirty"""
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

    def to_box(self) -> BoundingBox[_T]:
        """Convert to immutable boundary box."""
        assert not self._dirty and self._min and self._max
        return ImmutableBoundingBox(self._min, self._max)

    """Safely get an immutable boundary box of this unsafe box."""
    def to_safe(self, getter: Callable[[], Iterable[_Tup3[_T]] | npt.NDArray[np.int_]]) -> BoundingBox[_T]:
        self.update_if_dirty(getter)
        return self.to_box()
