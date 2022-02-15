"""
Module that contains the index based hash map based on tuples.

A python tuple is immutable and hashable.
We can't use numpy arrays as index because they are mutable and not hashable.
"""

from typing import (
    Any,
    Dict,
    Tuple,
    Iterator,
    Sequence,
    Generic,
    TypeVar,
    Callable,
    List,
    ItemsView,
    KeysView,
    ValuesView,
    overload,
)


import numpy as np
import numpy.typing as npt

from .iterators import SliceOpt, VoxelGridIterator
from .typing import Index3, Arr3i
from .bounding import BoundingBox, UnsafeBoundingBox


_T = TypeVar("_T")
_VT = TypeVar("_VT")
# T_co = TypeVar("T_co", covariant=True)
IndexUnion = Index3 | Sequence[int] | Arr3i


def _index(item: IndexUnion) -> Index3:
    """Convert any index type to a tuple index"""
    item = np.asarray(item, dtype=np.int_)
    if item.shape != (3,):
        raise IndexError(f"Invalid index {item}")
    return tuple(item)  # type: ignore


class IndexDict(Generic[_VT]):
    """
    A dictionary that uses 3d integer position tuples as keys.

    It keeps track of the minimum and maximum positions, which enables a fast size query.
    """

    __slots__ = ("_data", "_box")

    def __init__(self) -> None:
        self._data: Dict[Index3, _VT] = dict()
        self._box: UnsafeBoundingBox[int] = UnsafeBoundingBox(dtype=np.int_)

    def clear(self) -> None:
        """Clear all elements"""
        self._data.clear()
        self._box.clear()

    def __delitem__(self, key: Any) -> None:
        """Delete an item by index"""
        index = _index(key)
        del self._data[index]
        self._box.mark_dirty()

    @overload
    def pop(self, index: IndexUnion) -> _VT:
        """Get and delete an item by index"""
        ...

    @overload
    def pop(self, index: IndexUnion, default: _VT | _T) -> _VT | _T:
        """Get and delete an item by index, or return the default argument"""
        ...

    def pop(self, index: IndexUnion, default: Any = ...) -> Any:
        l = len(self._data)
        _data = self._data
        idx = _index(index)
        if default is ...:
            c = _data.pop(idx)
        else:
            c = _data.pop(idx, default)
        if l != len(_data):
            self._box.mark_dirty()
        return c

    @property
    def box(self) -> BoundingBox[int]:
        """Bounding box"""
        return self._box.to_safe(self._data.keys)

    def size(self) -> Arr3i:
        """Size of the bounding box"""
        if self._data:
            b = self.box
            return np.array(b.max, dtype=int) - np.array(b.min, dtype=int) + 1  # type: ignore
        return np.zeros(3, dtype=int)

    @overload
    def get(self, index: IndexUnion) -> _VT | None:
        """Get an item by index"""
        ...

    @overload
    def get(self, index: IndexUnion, default: _VT | _T | None) -> _VT | _T | None:
        """Get an item by index, or return the default argument"""
        ...

    def get(self, index: IndexUnion, default: _VT | _T | None = None) -> _VT | _T | None:  # type: ignore
        return self._data.get(_index(index), default)

    def __getitem_from_numpy(self, item: npt.NDArray[np.int_], ignore_empty: bool = True) -> _VT | List[_VT]:
        item = np.asarray(item, dtype=np.int_)
        if item.shape == (3,):
            return self._data[_index(item)]
        else:
            assert item.ndim == 2 and item.shape[1] == 3
            _data = self._data
            if ignore_empty:
                return [_data[index] for index in (_index(i) for i in item) if index in _data]
            else:
                return [_data[_index(i)] for i in item]

    def sliced_iterator(self, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None) -> VoxelGridIterator:
        """Iterate over all positions in the bounding box"""
        if not self._data:
            return VoxelGridIterator.empty()
        k_min, k_max = np.array(self.box.minmax)
        return VoxelGridIterator(low=k_min, high=k_max + 1, x=x, y=y, z=z)

    def sliced(
        self, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None, ignore_empty: bool = True
    ) -> Iterator[_VT]:
        _data = self._data
        if not _data:
            return
        if x is None and y is None and z is None:
            yield from _data.values()
        else:
            it = self.sliced_iterator(x, y, z)
            if ignore_empty:
                for key in it:
                    try:
                        yield _data[key]
                    except KeyError:
                        pass  # ignored
            else:
                for key in it:
                    yield _data[key]

    @overload
    def __getitem__(self, item: IndexUnion) -> _VT:
        ...

    @overload
    def __getitem__(self, item: Tuple[slice, slice, slice]) -> List[_VT]:
        ...

    def __getitem__(self, item: IndexUnion | Tuple[slice, slice, slice]) -> _VT | List[_VT]:
        if isinstance(item, slice):
            return list(self.sliced(item))
        if isinstance(item, tuple):
            if 0 < len(item) <= 3 and any(isinstance(i, slice) for i in item):
                return list(self.sliced(*item))
            else:
                try:
                    index: npt.NDArray[np.int_] = np.asarray(item, dtype=np.int_)
                    if index.shape == (3,):
                        return self._data[_index(item)]
                    raise KeyError(f"invalid key {item}")
                except TypeError:
                    pass
            raise KeyError(f"invalid key {item}")
        elif isinstance(item, (list, np.ndarray)):
            return self.__getitem_from_numpy(np.array(item, dtype=int))
        else:
            raise KeyError(f"invalid key {item}")

    def insert(self, index: IndexUnion, value: _VT) -> None:
        idx = _index(index)
        self._data[idx] = value
        self._box.mark_dirty()

    def __setitem__(self, key: IndexUnion, value: _VT) -> None:
        self.insert(key, value)

    def __contains__(self, item: IndexUnion) -> bool:
        return _index(item) in self._data

    def setdefault(self, index: IndexUnion, default: _VT) -> None:
        idx = _index(index)
        _data = self._data
        if idx not in _data:
            self._data[idx] = default
            self._box.mark_dirty()

    def create_if_absent(self, index: IndexUnion, factory: Callable[[Index3], _VT], *, insert: bool = True) -> _VT:
        idx = _index(index)
        _data = self._data
        try:
            return _data[idx]
        except KeyError:
            c = factory(idx)
            if insert:
                _data[idx] = c
                self._box.mark_dirty()
            return c

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> ItemsView[Index3, _VT]:
        return self._data.items()

    def keys(self) -> KeysView[Index3]:
        return self._data.keys()

    def values(self) -> ValuesView[_VT]:
        return self._data.values()

    def __iter__(self) -> Iterator[_VT]:
        return iter(self._data.values())
