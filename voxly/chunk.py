import functools
import operator
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Sequence,
    SupportsAbs,
    Tuple,
    Type,
    TypeVar,
    overload,
)

import numpy as np
import numpy.typing as npt

from .typing import Index3, SupportsDunderMul, Vec3i


class ChunkHelper:
    class _IndexMeshGrid:
        def __getitem__(self, item: slice) -> Iterator[Index3]:
            assert isinstance(item, slice)
            xs, ys, zs = np.mgrid[item, item, item]
            return zip(xs.flat, ys.flat, zs.flat)

    indexGrid = _IndexMeshGrid()

    @staticmethod
    def iter_grid(stop: int, start: int = 0) -> Iterator[Index3]:
        xs, ys, zs = np.mgrid[start:stop, start:stop, start:stop]
        return zip(xs.flat, ys.flat, zs.flat)

    @staticmethod
    @functools.cache
    def get_index_grid(size: int) -> npt.NDArray[np.int_]:
        xs, ys, zs = np.mgrid[0:size, 0:size, 0:size]
        result: npt.NDArray[np.int_] = np.vstack((xs, ys, zs)).T
        result.flags.writeable = False
        return result

    @staticmethod
    def _ensure_slice(low: int, high: int, s: int | slice | None, clip: bool = True) -> slice:
        if s is None:
            return slice(low, high, 1)
        _s: slice = s if isinstance(s, slice) else slice(s, s + 1)
        step = _s.step or 1
        start = low if _s.start is None else _s.start
        stop = high if _s.stop is None else _s.stop
        if clip:
            start = max(start, low + (start - low) % step)
            stop = min(stop, high)
        else:
            start = start
            stop = stop
        return slice(start, stop, step)

    @staticmethod
    def clip_index_slice(
        low: Vec3i, high: Vec3i, idx: Tuple[int | slice, ...]
    ) -> Tuple[bool, None | npt.NDArray[np.int_]]:
        assert 0 < len(idx) <= 3
        xss = ChunkHelper._ensure_slice(low[0], high[0], idx[0])
        yss = ChunkHelper._ensure_slice(low[1], high[1], idx[1] if len(idx) >= 2 else None)
        zss = ChunkHelper._ensure_slice(low[2], high[2], idx[2] if len(idx) == 3 else None)
        if (
            low == (xss.start, yss.start, zss.start)
            and high == (xss.stop, yss.stop, zss.stop)
            and (1, 1, 1) == (xss.step, yss.step, zss.step)
        ):
            return True, None
        xs, ys, zs = np.mgrid[xss, yss, zss]
        return False, np.vstack((xs, ys, zs)).T


_VT_co = TypeVar("_VT_co", covariant=True, bound=np.generic)
_OT_co = TypeVar("_OT_co", covariant=True, bound=np.generic)
_WT_co = TypeVar("_WT_co", covariant=True, bound=np.generic)


class Chunk(Generic[_VT_co]):
    __slots__ = ("_index", "_size", "_dtype", "_default", "_fill", "_array")

    def __init__(
        self,
        index: Index3,
        size: int,
        dtype: np.dtype[Any] | Type[Any],
        fill: Any | ellipsis = ...,
    ) -> None:
        self._index: Index3 = index
        self._size: int = size
        self._dtype: np.dtype[Any] = np.dtype(dtype)
        self._fill: _VT_co = self._dtype.type() if fill is ... else self._dtype.type(fill)
        self._array: npt.NDArray[_VT_co] | None = None

    @property
    def index(self) -> Index3:
        return self._index

    @property
    def size(self) -> int:
        return self._size

    @property
    def dtype(self) -> npt.DTypeLike:
        return self._dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        cs = self._size
        return cs, cs, cs

    @property
    def array_shape(self) -> Tuple[int, ...]:
        cs = self._size
        return cs, cs, cs, *self._dtype.shape

    @property
    def value(self) -> _VT_co | npt.NDArray[_VT_co]:
        return self._fill if self._array is None else self._array

    @property
    def fill(self) -> _VT_co:
        assert self._array is None
        return self._fill

    @property
    def array(self) -> npt.NDArray[_VT_co]:
        assert self._array is not None
        return self._array

    @property
    def position_low(self) -> Vec3i:
        return np.array(self._index, dtype=np.int_) * self._size  # type: ignore

    @property
    def position_high(self) -> Vec3i:
        return self.position_low + self._size

    def is_filled(self) -> bool:
        return self._array is None

    def is_filled_with(self, value: Any) -> np.bool8:
        _array = self._array
        if _array is None:
            return np.all(self._fill == value)
        return np.all(_array == value)

    def is_array(self) -> bool:
        return self._array is not None

    def index_inner(self, pos: Vec3i) -> Vec3i:
        return np.asarray(pos, dtype=np.int_) % self._size

    def to_array(self) -> npt.NDArray[_VT_co]:
        _array = self._array
        if _array is None:
            return np.full(self.shape, self._fill, dtype=self._dtype)
        else:
            return np.copy(_array)

    def set_array(self, data: npt.ArrayLike) -> "Chunk[_VT_co]":
        if isinstance(data, np.ndarray):
            data = data.astype(self._dtype)
        else:
            data = np.asarray(data, dtype=self._dtype)
        assert self.array_shape == data.shape, f"shape mismatch {self.array_shape} != {data.shape}"
        self._array = data
        return self

    def get(self, pos: Vec3i) -> _VT_co:
        _array = self._array
        if _array is None:
            return self._fill
        else:
            return _array[tuple(self.index_inner(pos))]  # type: ignore

    def set(self, pos: Vec3i, value: Any) -> None:
        _array = self._array
        if _array is None and self._fill == value:
            return  # no-op
        idx = tuple(self.index_inner(pos))
        _array = self.to_array()
        _array[idx] = value
        self._array = _array

    def set_or_fill(self, pos: Vec3i, value: Any) -> None:
        _array = self._array
        if _array is None:
            self.set_fill(value)
        else:
            self.set(pos, value)

    def _ensure_dtype(self, value: _VT_co | Any) -> _VT_co:
        _dtype = self._dtype
        if np.can_cast(value, _dtype):
            return value
        return self._dtype.type(value)  # type: ignore

    def set_fill(self, value: _VT_co | Any) -> None:
        # Need to ensure that the fill field is the dtype.
        self._fill = self._ensure_dtype(value)
        self._array = None

    def cleanup(self) -> "Chunk[_VT_co]":
        """Try to reduce memory footprint"""
        _array = self._array
        if _array is not None:
            u: npt.NDArray[_VT_co] = np.unique(_array)
            if len(u) == 1:
                self.set_fill(u.item())
        return self

    def __bool__(self) -> bool:
        raise ValueError(
            f"The truth value of {self.__class__} is ambiguous. "
            "Use a.any(), or a.all(), or wrap the comparison (0 < a) & (a < 0)"
        )

    def all(self) -> np.bool8:
        return np.all(self.value)

    def any(self) -> np.bool8:
        return np.any(self.value)

    def any_if_filled(self) -> bool:
        if self.is_filled():
            return bool(self.value)
        return True

    def astype(self, dtype: Type[_OT_co] | np.dtype[_OT_co], copy: bool = False) -> "Chunk[_OT_co]":
        _dtype = np.dtype(dtype)
        if not copy and self._dtype == _dtype:
            return self  # type: ignore
        c: "Chunk[_OT_co]" = Chunk(self._index, self._size, dtype=_dtype, fill=self._fill)
        _array = self._array
        if _array is not None:
            c.set_array(_array.astype(_dtype))
        return c

    def copy(self, empty: bool = False) -> "Chunk[_VT_co]":
        c: "Chunk[_VT_co]" = Chunk(self._index, self._size, self._dtype, self._fill)
        if not empty:
            _array = self._array
            if _array is not None:
                c.set_array(self.to_array())
        return c

    def filter(self, mask: "Chunk[np.bool8]") -> "Chunk[_VT_co]":
        _mask: Chunk[np.bool8] = mask.astype(np.bool8)
        c = self.copy(empty=True)
        if _mask.is_filled():
            if _mask.fill:
                if self.is_filled():
                    c.set_fill(self.fill)
                else:
                    c.set_array(self.array)
        else:
            arr = c.to_array()
            _mask_arr = _mask.array
            arr[_mask_arr] = self.to_array()[_mask_arr]
            c.set_array(arr)
        return c

    def items(
        self, mask: "Chunk[np.bool8]" | npt.NDArray[np.bool8] | None = None
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[_VT_co]]:
        """
        Get positions and voxels in this chunk in two lists. Optionally apply a filtering mask.
        """
        if mask is None:
            grid: npt.NDArray[np.int_] = ChunkHelper.get_index_grid(self._size)
        elif isinstance(mask, Chunk):
            assert mask.shape == self.shape
            grid = mask.argwhere()
        elif isinstance(mask, np.ndarray):
            assert mask.shape == self.shape
            grid = mask % self.size
        else:
            raise ValueError(f"invalid mask of type {type(mask)}")

        cps = grid + self.position_low
        _array = self._array
        if _array is None:
            return cps, np.full((len(cps),), self._fill, dtype=self._dtype)
        else:
            return cps, _array[tuple(grid.T)]

    def _argwhere(self) -> npt.NDArray[np.int_]:
        _array = self._array
        if _array is None:
            if self._fill:
                return ChunkHelper.get_index_grid(self._size)
            else:
                return np.empty((0, 3), dtype=int)
        return np.argwhere(_array.astype(np.bool8))

    def argwhere(self, mask: "Chunk[np.bool8]" | npt.NDArray[np.bool8] | None = None) -> npt.NDArray[np.int_]:
        if isinstance(mask, Chunk):
            if mask.is_filled() and not bool(mask.value):
                return np.empty((0, 3), dtype=int)
            mask_arr = mask.astype(np.bool8)
            idx = self._argwhere()
            return idx[mask_arr[idx]]  # type: ignore
        if isinstance(mask, np.ndarray):
            assert self.shape == mask.shape, f"shape mismatch {self.shape} != {mask.shape}"
            idx = self._argwhere()
            return idx[mask[idx]]  # type: ignore
        if mask is None:
            return self._argwhere()
        raise ValueError(f"invalid mask of type {type(mask)}")

    @overload
    def __getitem__(self, item: Index3) -> _VT_co:
        ...

    @overload
    def __getitem__(self, item: Sequence[Index3]) -> npt.NDArray[_VT_co]:
        ...

    @overload
    def __getitem__(self, item: npt.NDArray[np.int_]) -> npt.NDArray[_VT_co]:
        ...

    def __getitem__(self, item: Index3 | npt.ArrayLike) -> Any:
        if isinstance(item, Chunk):
            return self.filter(item)
        return self.to_array()[item]

    def __setitem__(
        self,
        key: npt.NDArray[np.int_] | "Chunk[np.bool8]" | Tuple[int | slice, int | slice, int | slice],
        value: _VT_co | npt.NDArray[_VT_co] | "Chunk[_VT_co]",
    ) -> None:

        _all: bool | np.bool8 = False
        _index: npt.NDArray[np.int_] | None = None
        _mask: npt.NDArray[np.bool8] | None = None

        if isinstance(key, Chunk):
            assert key.size == self.size, f"invalid key: chunk size {key.size} != {self.size}"
            _key = key.astype(np.bool8)
            if _key.is_filled():
                if bool(_key.fill):
                    _all = True
                else:
                    return
            else:
                _mask = _key.array
        elif isinstance(key, np.ndarray):
            if key.dtype == np.bool8:
                assert key.shape == self.shape, f"invalid key: mask shape {key.shape} != {self.shape}"
                _mask = key.astype(np.bool8)
                _all = np.all(key)
            else:
                assert key.ndim == 2 and key.shape[1] == 3, f"invalid key: index list {key.shape} != (N,3)"
                _index = key % self._size
        elif isinstance(key, tuple) and len(key) == 3:
            _mask = np.zeros(self.shape, np.bool8)
            _mask[key] = True
            _all = np.all(_mask)
        else:
            try:
                idx_list: npt.NDArray[np.int_] = np.asarray(key, dtype=np.int_)
                assert (
                    idx_list.ndim == 2 and idx_list.shape[1] == 3
                ), f"invalid key: index list {idx_list.shape} != (N,3)"
                _index = idx_list
            except ValueError as exc:
                raise ValueError(f"invalid key: unkown '{key}'") from exc

        if _all:
            if isinstance(value, Chunk):
                assert self.size == value.size, f"invalid value: size {value.size} != {self.size}"
                _value_array = value._array
                if _value_array is None:
                    self.set_fill(value._fill)
                else:
                    self.set_array(_value_array)
            elif isinstance(value, np.ndarray):
                assert self.shape == value.shape, f"invalid value: shape {value.shape} != {self.shape}"
                self.set_array(value)
            else:
                self.set_fill(value)
        elif _mask is not None:
            _self_array = self.to_array()
            if isinstance(value, Chunk):
                assert self.size == value.size, f"invalid value: size {value.size} != {self.size}"
                _self_array[_mask] = value.to_array()[_mask]
            elif isinstance(value, np.ndarray):
                assert self.shape == value.shape, f"invalid value: shape {value.shape} != {self.shape}"
                _self_array[_mask] = value[_mask]
            else:
                _self_array[_mask] = value
            self.set_array(_self_array)
        elif _index is not None:
            _self_array = self.to_array()
            if isinstance(value, Chunk):
                assert self.size == value.size, f"invalid value: size {value.size} != {self.size}"
                _self_array[_index] = value.to_array()[_index]
            elif isinstance(value, (np.ndarray, list, tuple)):
                assert len(_index) == len(value), f"invalid value: length {len(value)} != {len(_index)}"
                _self_array[_index] = value
            else:
                _self_array[_index] = value
            self.set_array(_self_array)

    def split(self, splits: int, chunk_size: int | None = None) -> Iterator["Chunk[_VT_co]"]:
        splits = int(splits)
        assert splits > 0 and self._size % splits == 0
        split_size = self._size // splits

        # New chunk size
        chunk_size = int(chunk_size or self._size)
        assert chunk_size > 0

        dtype = self._dtype

        # Voxel repeats to achieve chunk size
        repeats = chunk_size / split_size
        assert repeats > 0 and repeats % 1 == 0  # Chunk size must be achieved by duplication of voxels
        repeats = int(repeats)

        for offset in ChunkHelper.iter_grid(splits):
            new_index = np.add(self._index * splits, offset)
            c: "Chunk[_VT_co]" = Chunk(new_index, size=chunk_size, dtype=dtype, fill=self._fill)
            _self_array = self._array
            if _self_array is not None:
                u, v, w = np.asarray(offset, dtype=np.int_) * split_size
                tmp = _self_array[u : u + split_size, v : v + split_size, w : w + split_size]
                if repeats == 1:
                    val = tmp.copy()
                else:
                    val = np.repeat(np.repeat(np.repeat(tmp, repeats, axis=0), repeats, axis=1), repeats, axis=2)
                c.set_array(val)
            yield c

    def convert(
        self,
        dtype: Type[_OT_co] | np.dtype[_OT_co],
        func: Callable[[Any], _OT_co],
        func_vec: Callable[[npt.NDArray[_VT_co]], npt.NDArray[_OT_co]] | None = None,
    ) -> "Chunk[_OT_co]":
        c: "Chunk[_OT_co]" = Chunk(self._index, self._size, dtype, fill=func(self._fill))
        _array = self._array
        if _array is None:
            c.set_fill(func(self.fill))
        else:
            func_vec = func_vec or np.vectorize(func)
            c.set_array(func_vec(_array))
        return c

    # @overload
    # def apply(
    #     self,
    #     func: Callable[[npt.NDArray[_VT_co] | _VT_co], npt.NDArray[_OT_co] | _OT_co],
    #     dtype: Type[_OT_co] | npt.DTypeLike = None,
    # ) -> "Chunk[_OT_co]":
    #     ...

    # @overload
    # def apply(
    #     self,
    #     func: Callable[[npt.NDArray[_VT_co] | _VT_co], npt.NDArray[_VT_co] | _VT_co],
    #     dtype: Type[_VT_co] | npt.DTypeLike = None,
    #     inplace: Literal[True] = True,
    # ) -> "Chunk[_VT_co]":
    #     ...

    # @overload
    # def apply(
    #     self,
    #     func: Callable[[npt.NDArray[_VT_co] | _VT_co], npt.NDArray[_OT_co] | _OT_co],
    #     dtype: Type[_OT_co] | npt.DTypeLike = None,
    #     inplace: Literal[False] = False,
    # ) -> "Chunk[_OT_co]":
    #     ...

    def apply(
        self,
        func: Callable[[npt.NDArray[_VT_co] | _VT_co], npt.NDArray[_OT_co] | _OT_co],
        dtype: Type[_OT_co] | npt.DTypeLike = None,
        into: "Chunk[_OT_co]" | None = None,
    ) -> "Chunk[_OT_co]":
        _fill = func(self._fill)

        if into is None:
            if dtype is None:
                _dtype: np.dtype[_OT_co] = np.asarray(_fill).dtype
            else:
                _dtype = np.dtype(dtype)  # type: ignore
            c = self.astype(dtype=_dtype, copy=True)
        else:
            c = into

        c.set_fill(_fill)
        _array = self._array
        if _array is not None:
            c.set_array(func(_array))
        return c

    def join(
        self,
        rhs: "Chunk[_WT_co]",
        func: Callable[[_VT_co | npt.NDArray[_VT_co], _WT_co | npt.NDArray[_WT_co]], _OT_co | npt.NDArray[_OT_co]],
        dtype: Type[_OT_co] | np.dtype[_OT_co] | None = None,
        into: "Chunk[_OT_co]" | None = None,
    ) -> "Chunk[_OT_co]":
        _fill = func(self._fill, rhs._fill)

        if into is None:
            if dtype is None:
                _dtype: np.dtype[_OT_co] = np.asarray(_fill).dtype
            else:
                _dtype = np.dtype(dtype)  # type: ignore
            c = self.astype(dtype=_dtype, copy=True)
        else:
            c = into

        c.set_fill(_fill)
        _array = self._array
        if _array is not None:
            c.set_array(func(_array, rhs.value))
        return c

    # @overload
    # def join(
    #     self,
    #     rhs: _VT_co | npt.NDArray[_VT_co] | "Chunk[_VT_co]",
    #     func: Callable[[_VT_co | npt.NDArray[_VT_co], _VT_co | npt.NDArray[_VT_co]], _OT_co | npt.NDArray[_OT_co]],
    #     dtype: Type[_OT_co] | npt.DTypeLike = None,
    # ) -> "Chunk[_OT_co]":
    #     ...

    # @overload
    # def join(
    #     self,
    #     rhs: _VT_co | npt.NDArray[_VT_co] | "Chunk[_VT_co]",
    #     func: Callable[[_VT_co | npt.NDArray[_VT_co], _VT_co | npt.NDArray[_VT_co]], _VT_co | npt.NDArray[_VT_co]],
    #     dtype: Type[_VT_co] | npt.DTypeLike = None,
    #     inplace: Literal[True] = True,
    # ) -> "Chunk[_VT_co]":
    #     ...

    # @overload
    # def join(
    #     self,
    #     rhs: _VT_co | npt.NDArray[_VT_co] | "Chunk[_VT_co]",
    #     func: Callable[[_VT_co | npt.NDArray[_VT_co], _VT_co | npt.NDArray[_VT_co]], _OT_co | npt.NDArray[_OT_co]],
    #     dtype: Type[_OT_co] | npt.DTypeLike = None,
    #     inplace: Literal[False] = False,
    # ) -> "Chunk[_OT_co]":
    #     ...

    # def join(
    #     self,
    #     rhs: _VT_co | npt.NDArray[_VT_co] | "Chunk[_VT_co]",
    #     func: Callable[[_VT_co | npt.NDArray[_VT_co], _VT_co | npt.NDArray[_VT_co]], _OT_co | npt.NDArray[_OT_co]],
    #     dtype: Type[_OT_co] | npt.DTypeLike = None,
    #     inplace: bool = False,
    # ) -> "Chunk[_OT_co]" | "Chunk[_VT_co]":
    #     _dtype = self._dtype if dtype is None else np.dtype(dtype)
    #     _self_array = self._array

    #     if isinstance(rhs, Chunk):
    #         _rhs_array: Any = rhs._array
    #         _rhs_fill = rhs._fill
    #     else:
    #         _rhs_array = rhs  # assume array covers value-case
    #         _rhs_fill = None

    #     if inplace:
    #         assert self._dtype == _dtype
    #         self.default = func(self._default, rhs._default)  # type: ignore
    #         if _self_array is None and _rhs_array is None:
    #             self.set_fill(func(self.fill, _rhs_fill))   # type: ignore
    #         else:
    #             if _rhs_array is None:
    #                 self.set_array(func(self.to_array(), _rhs_fill)) # type: ignore
    #             else:
    #                 self.set_array(func(self.to_array(), _rhs_array)) # type: ignore
    #         return self
    #     else:
    #         _default: _VT_co | _OT_co = func(self._default, rhs._default)
    #         c = self.astype(dtype=_dtype, default=_default, copy=True)
    #         if _self_array is None and _rhs_array is None:
    #             c.set_fill(func(self._fill, _rhs_fill))
    #         else:
    #             c.set_array(func(self.value, rhs.value))
    #         return c

    # Comparison Operator

    def __eq__(self, rhs: Any) -> "Chunk[np.bool8]":  # type: ignore[override]
        return self.join(rhs, func=operator.eq, dtype=np.bool8).cleanup()

    def __ne__(self, rhs: Any) -> "Chunk[np.bool8]":  # type: ignore[override]
        return self.join(rhs, func=operator.ne, dtype=np.bool8).cleanup()

    def __lt__(self, rhs: Any) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.lt, dtype=np.bool8).cleanup()

    def __le__(self, rhs: Any) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.le, dtype=np.bool8).cleanup()

    def __gt__(self, rhs: Any) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.gt, dtype=np.bool8).cleanup()

    def __ge__(self, rhs: Any) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.ge, dtype=np.bool8).cleanup()

    eq = __eq__
    ne = __ne__
    lt = __lt__
    le = __le__
    gt = __gt__
    ge = __ge__

    equals = __eq__

    # Single Operator

    def __abs__(self: "Chunk[SupportsAbs[_VT_co]]") -> "Chunk[_VT_co]":  # type: ignore[type-var]
        return self.apply(func=operator.abs)  # type: ignore

    def __invert__(self) -> "Chunk[_VT_co]":
        return self.apply(func=operator.inv)

    def __neg__(self) -> "Chunk[_VT_co]":
        return self.apply(func=operator.neg)

    abs = __abs__
    invert = __invert__
    neg = __neg__

    # Logic Operator

    def __and__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.and_)

    def __or__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.or_)

    def __xor__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.xor)

    def __iand__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.iand, into=self)

    def __ior__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.ior, into=self)

    def __ixor__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.ixor, into=self)

    # Math Operator

    def __add__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.add)

    def __sub__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.sub)

    def __mul__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.mul)

    def __matmul__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.matmul)

    def __mod__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.mod)

    def __pow__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.pow)

    def __floordiv__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.floordiv)

    def __iadd__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.iadd, into=self)

    def __isub__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.isub, into=self)

    def __imul__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.imul, into=self)

    def __imatmul__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.imatmul, into=self)

    def __imod__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.imod, into=self)

    def __ifloordiv__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.ifloordiv, into=self)

    # TrueDiv Operator

    def __truediv__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.truediv)

    def __itruediv__(self, rhs: Any) -> "Chunk[_VT_co]":
        return self.join(rhs, func=operator.itruediv, into=self)

    # Reflected Operators
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__
    __rtruediv__ = __truediv__
    __rfloordiv__ = __floordiv__
    __rmod__ = __mod__
    __rpow__ = __pow__
    __rand__ = __and__
    __rxor__ = __xor__
    __ror__ = __or__

    @overload
    def sum(self) -> _VT_co:
        ...

    @overload
    def sum(self, dtype: Type[_OT_co]) -> _OT_co:
        ...

    @overload
    def sum(self, dtype: npt.DTypeLike) -> Any:
        ...

    def sum(self, dtype: Type[_OT_co] | npt.DTypeLike = None) -> _VT_co | _OT_co | Any:
        _dtype = self._dtype if dtype is None else np.dtype(dtype)
        _array = self._array
        if _array is None:
            assert isinstance(self._fill, SupportsDunderMul)
            return self._fill * (self._size**3)
        else:
            return np.sum(_array, dtype=_dtype)

    def unique(self) -> npt.NDArray[_VT_co]:
        _array = self._array
        if _array is None:
            return np.asanyarray([self._fill])
        else:
            return np.unique(_array)

    # @classmethod
    # def _stack(
    #     cls, chunks: Sequence["Chunk[_VT_co]"], dtype: npt.DTypeLike, fill_value: npt.NDArray[_VT_co] | ellipsis = ...
    # ) -> "Chunk[npt.NDArray[_VT_co]]":
    #     _dtype = np.dtype(dtype)
    #     index = chunks[0]._index
    #     size = chunks[0]._size

    #     # _fill_value: npt.NDArray[_VT_co] = np.array([c._fill_value for c in chunks], dtype=dtype) if fill_value is ... else fill_value

    #     new_chunk: "Chunk[npt.NDArray[_VT_co]]" = Chunk(index, size, _dtype, fill_value)
    #     if all(c._is_filled for c in chunks):
    #         new_chunk.fill(np.array([c.value for c in chunks], dtype=_dtype.base))
    #     else:
    #         arr = np.array([c.to_array() for c in chunks], dtype=_dtype.base).transpose((1, 2, 3, 0))
    #         new_chunk.set_array(arr)
    #     return new_chunk
