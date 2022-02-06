from typing import (
    Any,
    Dict,
    Union,
    Tuple,
    Iterator,
    Optional,
    Iterable,
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
from .typing import Index3, Vec3i
from .boundary import Box, UnsafeBox


T = TypeVar("T")
# T_co = TypeVar("T_co", covariant=True)
Index = Tuple[int, int, int]
IndexUnion = Index | Sequence[int] | npt.NDArray[np.int_]


class IndexDict(Generic[T]):
    """
    A dictionary that uses 3d integer position tuples as keys.

    It keeps track of the minimum and maximum positions, which enables a fast size query.
    """

    __slots__ = ("_data", "_box")

    def __init__(self) -> None:
        self._data: Dict[Index, T] = dict()
        self._box: UnsafeBox[int] = UnsafeBox()

    def clear(self) -> None:
        self._data.clear()
        self._box.clear()

    def __delitem__(self, key: Any) -> None:
        index = self.index(key)
        del self._data[index]
        self._box.mark_dirty()

    @overload
    def pop(self, index: Index3) -> T:
        ...

    @overload
    def pop(self, index: Index3, default: T):
        ...

    def pop(self, index: Index3, default: T_co | None = None) -> T | T_co:
        l = len(self._data)
        c = self._data.pop(index, default)
        if l != len(self._data):
            self._box.mark_dirty()
        return c

    @property
    def box(self) -> Box[int]:
        self._box.to_safe(self._data.keys)

    def size(self) -> Vec3i:
        if self._data:
            min, max = np.array(self.box.minmax)
            return max - min + 1
        return np.zeros(3, dtype=np.int_)

    @classmethod
    def index(cls, item: IndexUnion) -> Index:
        item = np.asarray(item, dtype=np.int_)
        if item.shape != (3,):
            raise IndexError(f"Invalid index {item}")
        return tuple(item)

    def get(self, index: Index, default=None) -> Optional[T]:
        return self._data.get(self.index(index), default)

    def __getitem_from_numpy(self, item: npt.NDArray[np.int_], ignore_empty=True) -> Union[T, List[T]]:
        item = np.asarray(item, dtype=np.int_)
        if item.shape == (3,):
            return self._data[tuple(item)]
        else:
            assert item.ndim == 2 and item.shape[1] == 3
            if ignore_empty:
                return [d for d in (self._data.get(tuple(i), None) for i in item) if d is not None]
            else:
                return [self._data[tuple(i)] for i in item]

    def sliced_iterator(self, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None) -> VoxelGridIterator:
        if not self._data:
            return VoxelGridIterator.empty()
        k_min, k_max = np.array(self.box.minmax)
        return VoxelGridIterator(low=k_min, high=k_max + 1, x=x, y=y, z=z)

    def sliced(self, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None, ignore_empty=True) -> Iterator[T]:
        if not self._data:
            return
        if x is None and y is None and z is None:
            yield from self._data.values()
        else:
            it = self.sliced_iterator(x, y, z)
            if ignore_empty:
                for key in it:
                    if key in self._data:
                        yield self._data[key]
            else:
                for key in it:
                    yield self._data.get(key)

    def __getitem__(self, item: Union[IndexUnion, slice, Tuple[slice, ...]]) -> Union[T, List[T]]:
        if isinstance(item, slice):
            return list(self.sliced(item))
        if isinstance(item, tuple):
            try:
                index = np.asarray(item, dtype=np.int_)
                if index.shape == (3,):
                    return self._data[self.index(item)]
                raise KeyError(f"invalid key {item}")
            except TypeError:
                pass
            if 0 < len(item) <= 3 and any(isinstance(i, slice) for i in item):
                return list(self.sliced(*item))
            raise KeyError(f"invalid key {item}")
        elif isinstance(item, (list, np.ndarray)):
            return self.__getitem_from_numpy(np.array(item, dtype=int))
        else:
            raise KeyError(f"invalid key {item}")

    def insert(self, index: Index, value: T):
        index = self.index(index)
        self._data[index] = value
        self._box.add(index)

    def __setitem__(self, key: Index, value: T):
        self.insert(key, value)

    def __contains__(self, item: Index) -> bool:
        return self.index(item) in self._data

    def setdefault(self, index: Index, default: T) -> T:
        index = self.index(index)
        c: T | None = self._data.get(index, None)
        if c is None:
            self._data[index] = default
            self._box.add(index)
            return default
        return c

    def create_if_absent(self, index: Index, factory: Callable[[Index], T], *, insert=True) -> T:
        index = self.index(index)
        _data = self._data
        c: T | None = _data.get(index, None)
        if c is None:
            c = factory(index)
            if insert:
                _data[index] = c
                self._box.add(index)
        return c

    def __len__(self) -> int:
        return len(self._data)

    def items(self) -> ItemsView[Index, T]:
        return self._data.items()

    def keys(self) -> KeysView[Index]:
        return self._data.keys()

    def values(self) -> ValuesView[T]:
        return self._data.values()

    def __iter__(self) -> Iterator[T]:
        return iter(self._data.values())
