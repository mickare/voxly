import functools
import operator
from tkinter.messagebox import NO
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)
import warnings

import numpy as np
import numpy.typing as npt

from voxly.iterators import VoxelGridIterator

from .typing import BoolType, Index3, SupportsDunderMul, Vec3i

_VT = TypeVar("_VT", bound=np.generic)
_OT = TypeVar("_OT", bound=np.generic)


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
    def get_grid(size: int) -> npt.NDArray[np.int_]:
        xs, ys, zs = np.mgrid[0:size, 0:size, 0:size]
        result: npt.NDArray[np.int_] = np.vstack((xs, ys, zs)).T
        result.flags.writeable = False
        return result



class Chunk(Generic[_VT]):
    __slots__ = ("_index", "_size", "_dtype", "_default", "_fill", "_array")

    def __init__(
        self,
        index: Index3,
        size: int,
        dtype: npt.DTypeLike = None,
        default: _VT | ellipsis = ...,
    ) -> None:
        self._index: Index3 = index
        self._size: int = size
        self._dtype: np.dtype[Any] = np.dtype(dtype)
        self._default: _VT = self._dtype.base.type() if default is ... else self._dtype.base.type(default)
        self._fill: _VT | None = self._default
        self._array: npt.NDArray[_VT] | None = None

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
    def value(self) -> _VT | npt.NDArray[_VT]:
        return self._fill if self._array is None else self._array  # type: ignore

    @property
    def position_low(self) -> Vec3i:
        return np.array(self._index, dtype=int) * self._size  # type: ignore

    @property
    def position_high(self) -> Vec3i:
        return self.position_low + self._size

    def is_filled(self) -> bool:
        return self._array is None

    def is_filled_with(self, value: _VT) -> bool:
        _array = self._array
        if _array is None:
            return np.all(self._fill == value)  # type: ignore
        return np.all(_array == value)  # type: ignore

    def is_array(self) -> bool:
        return self._array is not None

    def index_inner(self, pos: Vec3i) -> Vec3i:
        return np.asarray(pos, dtype=np.int_) % self._size

    def to_array(self) -> npt.NDArray[_VT]:
        _array = self._array
        if _array is None:
            return np.full(self.shape, self._fill, dtype=self._dtype)
        else:
            return _array

    def set_array(self, data: npt.NDArray[_VT]) -> "Chunk[_VT]":
        if isinstance(data, np.ndarray):
            data = data.astype(self._dtype)
        else:
            data = np.asarray(data, dtype=self._dtype)
        assert self.array_shape == data.shape, f"shape mismatch {self.array_shape} != {data.shape}"
        self._array = data
        self._fill = None
        return self

    def get(self, pos: Vec3i) -> _VT:
        _array = self._array
        if _array is None:
            return self._fill  # type: ignore
        else:
            return _array[tuple(self.index_inner(pos))]  # type: ignore

    def put(self, pos: Vec3i, value: _VT) -> None:
        _array = self._array
        if _array is None and self._fill == value:
            return  # no-op
        idx = tuple(self.index_inner(pos))
        _array = self.to_array()
        _array[idx] = value
        self._array = _array
        self._fill = None

    def put_or_fill(self, pos: Vec3i, value: _VT) -> None:
        _array = self._array
        if _array is None:
            self.fill(value)
        else:
            self.put(pos, value)

    def _ensure_dtype(self, value: _VT) -> _VT:
        _dtype = self._dtype
        if not _dtype.subdtype:
            return _dtype.type(value)  # type: ignore
        else:
            try:
                return _dtype.base.type(value)
            except Exception as e:
                warnings.warn(f"Failed to convert type to subtype: {e}")
                pass
        return value

    def fill(self, value: _VT) -> None:
        # Need to ensure that the fill field is the dtype.
        self._fill = self._ensure_dtype(value)
        self._array = None

    def cleanup(self) -> "Chunk[_VT]":
        """Try to reduce memory footprint"""
        _array = self._array
        if _array is not None:
            u: npt.NDArray[_VT] = np.unique(_array)
            if len(u) == 1:
                self.fill(u.item())
        return self

    def __bool__(self) -> bool:
        raise ValueError(
            f"The truth value of {self.__class__} is ambiguous. "
            "Use a.any(), or a.all(), or wrap the comparison (0 < a) & (a < 0)"
        )

    def all(self) -> bool:
        return np.all(self.value)  # type: ignore

    def any(self) -> bool:
        return np.any(self.value)  # type: ignore

    def any_if_filled(self) -> bool:
        if self.is_filled():
            return bool(self.value)
        return True

    def astype(self, dtype: Type[_OT], default: _OT | ellipsis = ...) -> "Chunk[_OT]":
        _dtype = np.dtype(dtype)
        if self._dtype == _dtype:
            return self # type: ignore
        _default = _dtype.type(self._default) if default is ... else default
        c: "Chunk[_OT]" = Chunk(self._index, self._size, dtype=_dtype, default=_default)
        _array = self._array
        if _array is None:
            c.fill(_dtype.type(self._fill))
        else:
            c.set_array(_array.astype(_dtype))
        return c

    @overload
    def copy(self) -> "Chunk[_VT]":
        ...

    @overload
    def copy(self, empty: bool) -> "Chunk[_VT]":
        ...

    @overload
    def copy(self, empty: bool, default: _VT) -> "Chunk[_VT]":
        ...

    @overload
    def copy(self, empty: bool, default: _OT, dtype: np.dtype[_OT] | npt.DTypeLike) -> "Chunk[_OT]":
        ...

    def copy(self, empty=False, default=None, dtype=None) -> "Chunk[_VT]" | "Chunk[_OT]":
        dtype = self._dtype if dtype is None else np.dtype(dtype)
        default = self._default if default is None else default
        c = Chunk(self._index, self._size, dtype=dtype, default=default)

        if not empty:
            if self.is_filled():
                fill = self._fill
                if dtype.subdtype:
                    fill = np.copy(fill)
                c.fill(fill)
            else:
                c.set_array(self._value.copy())
        return c

    def filter(self, mask: "Chunk[np.bool_]", default: Optional[_VT] = None) -> "Chunk[_VT]":
        mask: Chunk[bool] = mask.astype(bool)
        c = self.copy(empty=True, default=default)
        if mask.is_filled():
            if mask._fill:
                c._array = None
                c._fill = self._fill
        else:
            arr = c.to_array()
            mask_arr = mask.to_array()
            arr[mask_arr] = self._value[mask_arr]
            c.set_array(arr)
        return c

    def items(self, mask: "Chunk[np.bool_]" = None) -> Iterator[Tuple[Vec3i, _VT]]:
        it = VoxelGridIterator(np.zeros(3), self.shape)
        if mask is None:
            ps = np.asarray(list(it))
        elif isinstance(mask, Chunk):
            m = mask.to_array()
            ps = np.array([p for p in it if m[p]])
        else:
            raise ValueError(f"invalid mask of type {type(mask)}")
        if len(ps) > 0:
            cps = ps + self.position_low
            if self.is_filled():
                yield from ((p, self._value) for p in cps)
            else:
                yield from zip(cps, self.to_array()[tuple(ps.T)])

    def _argwhere(self) -> npt.NDArray[np.int_]:
        if self._is_filled:
            if self._value:
                return ChunkHelper.get_grid(self._size)
            else:
                return np.empty((0, 3), dtype=int)
        return np.argwhere(self.to_array().astype(bool))

    def argwhere(self, mask: "Chunk[np.bool_]" | npt.NDArray[np.bool_] = None) -> npt.NDArray[np.int_]:
        if isinstance(mask, Chunk):
            if mask.is_filled() and not bool(mask.value):
                return np.empty((0, 3), dtype=int) # type: ignore
            mask_arr = mask.astype(bool)
            idx = self._argwhere()
            return idx[mask_arr[idx]]
        if isinstance(mask, np.ndarray):
            assert self.shape == mask.shape, f"shape mismatch {self.shape} != {mask.shape}"
            idx = self._argwhere()
            return idx[mask[idx]] # type: ignore
        if mask is None:
            return self._argwhere()
        raise ValueError(f"invalid mask of type {type(mask)}")

    @overload
    def __getitem__(self, item: Index3) -> _VT:
        ...

    @overload
    def __getitem__(self, item: Sequence[Index3]) -> npt.NDArray[_VT]:
        ...

    @overload
    def __getitem__(self, item: npt.NDArray[np.int_]) -> npt.NDArray[_VT]:
        ...

    def __getitem__(self, item):
        if isinstance(item, Chunk):
            return self.filter(item)
        return self.to_array()[item]

    def __setitem__(
        self, key: npt.NDArray[np.int_] | "Chunk[bool]", value: _VT | npt.NDArray[_VT] | "Chunk[_VT]"
    ) -> None:
        is_value_chunk = isinstance(value, Chunk)
        if is_value_chunk:
            assert value._size == self._size

        if isinstance(key, Chunk):
            assert self.size == key.size, f"size mismatch  {self.size} != {key.size}"
            if key.all():
                # Fast set
                if is_value_chunk:
                    self._is_filled = value._is_filled
                    self._value = value._value
                    return
            key = key.to_array().astype(np.bool8)

        arr = self.to_array()
        if is_value_chunk:  # Masked
            arr[key] = value.to_array()[key]
        else:
            arr[key] = value
        self.set_array(arr)

    def split(self, splits: int, chunk_size: int = None) -> Iterator["Chunk"]:
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
            c = Chunk(new_index, size=chunk_size, dtype=dtype, default=self._default)
            if self.is_filled():
                c.fill(self._value)
            else:
                u, v, w = np.asarray(offset, dtype=np.int_) * split_size
                tmp = self._value[u : u + split_size, v : v + split_size, w : w + split_size]
                if repeats == 1:
                    val = tmp.copy()
                else:
                    val = np.repeat(np.repeat(np.repeat(tmp, repeats, axis=0), repeats, axis=1), repeats, axis=2)
                c.set_array(val)
            yield c

    def convert(
        self, func: Callable[[_VT], _OT], func_vec: Callable[[npt.NDArray[_VT]], npt.NDArray[_OT]] = None
    ) -> "Chunk[_OT]":
        c = Chunk(self._index, self._size, default=func(self._default))
        if self.is_filled():
            val: _OT = func(self._value)
            c.fill()
        else:
            func_vec = func_vec or np.vectorize(func)
            c.set_array(func_vec(self._value))
        return c

    @overload
    def apply(self, func: Callable[[npt.NDArray[_VT] | _VT], npt.NDArray[_OT] | _OT], 
    dtype: Type[_OT] | npt.DTypeLike = None) -> "Chunk[_OT]":
        ...
    @overload
    def apply(self, func: Callable[[npt.NDArray[_VT] | _VT], npt.NDArray[_VT] | _VT], 
    dtype: Type[_VT] | npt.DTypeLike = None, inplace: Literal[True] = True) -> "Chunk[_VT]":
        ...
    @overload
    def apply(self, func: Callable[[npt.NDArray[_VT] | _VT], npt.NDArray[_OT] | _OT], 
    dtype: Type[_OT] | npt.DTypeLike = None, inplace: Literal[False] = False) -> "Chunk[_OT]":
        ...

    def apply(
        self,
        func: Callable[[npt.NDArray[_VT] | _VT], npt.NDArray[_OT] | _OT],
        dtype: Type[_OT] | npt.DTypeLike = None,
        inplace=False,
    ) -> "Chunk[_OT]":
        _dtype = self._dtype if dtype is None else np.dtype(dtype)
        _self_array = self._array
        if inplace:
            assert _dtype == self._dtype
            self._default = func(self._default)
            if _self_array is None:
                self.fill(func(self._fill))
            else:
                self.set_array(func(_self_array))
        else:
            _default = func(self._default)
            c = self.copy(empty=True, dtype=_dtype, default=_default)
            if _self_array is None:
                c.fill(func(self._fill))
            else:
                c.set_array(func(_self_array))

    @overload
    def join(self, rhs: _VT | npt.NDArray[_VT] | "Chunk[_VT]", 
    func: Callable[[_VT | npt.NDArray[_VT], _VT | npt.NDArray[_VT]], _OT | npt.NDArray[_OT]],
    dtype: Type[_OT] | npt.DTypeLike = None) -> "Chunk[_OT]":
        ...

    @overload
    def join(self, rhs: _VT | npt.NDArray[_VT] | "Chunk[_VT]",
    func: Callable[[_VT | npt.NDArray[_VT], _VT | npt.NDArray[_VT]], _VT | npt.NDArray[_VT]],
    dtype: Type[_VT] | npt.DTypeLike = None, inplace: Literal[True] = True) -> "Chunk[_VT]":
        ...

    @overload
    def join(self, rhs: _VT | npt.NDArray[_VT] | "Chunk[_VT]",
    func: Callable[[_VT | npt.NDArray[_VT], _VT | npt.NDArray[_VT]], _OT | npt.NDArray[_OT]],
    dtype: Type[_OT] | npt.DTypeLike = None, inplace: Literal[False] = False) -> "Chunk[_OT]":
        ...

    def join(
        self,
        rhs: _VT | npt.NDArray[_VT] | "Chunk[_VT]",
        func: Callable[[_VT | npt.NDArray[_VT], _VT | npt.NDArray[_VT]], _OT | npt.NDArray[_OT]],
        dtype: Type[_OT] | npt.DTypeLike = None,
        inplace:bool=False,
    ) -> "Chunk[_OT]":
        _dtype = self._dtype if dtype is None else np.dtype(dtype)
        _self_array = self._array

        if isinstance(rhs, Chunk):
            _rhs_array = rhs._array
            _rhs_fill = rhs._fill
        else:
            _rhs_array = rhs  # assume array covers value-case
            _rhs_fill = None

        if inplace:
            assert self._dtype == _dtype
            self._default = func(self._default, rhs._default)
            if _self_array is None and _rhs_array is None:
                self.fill(func(self._fill, _rhs_fill))
            else:
                self.set_array(func(self.value, rhs.value))
            return self
        else:
            _default = func(self._default)
            c = self.copy(empty=True, dtype=_dtype, default=_default)
            if _self_array is None and _rhs_array is None:
                c.fill(func(self._fill, _rhs_fill))
            else:
                c.set_array(func(self.value, rhs.value))
            return c

    # Comparison Operator

    def __eq__(self, rhs: Any) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.eq, dtype=np.bool8).cleanup()

    def __ne__(self, rhs: Any) -> "Chunk[np.bool8]":
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

    def __abs__(self) -> "Chunk[_VT]":
        return self.apply(func=operator.abs)

    def __invert__(self) -> "Chunk[_VT]":
        return self.apply(func=operator.inv)

    def __neg__(self) -> "Chunk[_VT]":
        return self.apply(func=operator.neg)

    abs = __abs__
    invert = __invert__
    neg = __neg__

    # Logic Operator

    def __and__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.and_)

    def __or__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.or_)

    def __xor__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.xor)

    def __iand__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.iand, inplace=True)

    def __ior__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.ior, inplace=True)

    def __ixor__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.ixor, inplace=True)

    # Math Operator

    def __add__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.add)

    def __sub__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.sub)

    def __mul__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.mul)

    def __matmul__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.matmul)

    def __mod__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.mod)

    def __pow__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.pow)

    def __floordiv__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.floordiv)

    def __iadd__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.iadd, inplace=True)

    def __isub__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.isub, inplace=True)

    def __imul__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.imul, inplace=True)

    def __imatmul__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.imatmul, inplace=True)

    def __imod__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.imod, inplace=True)

    def __ifloordiv__(self, rhs: Any) -> "Chunk[_VT]":
        return self.join(rhs, func=operator.ifloordiv, inplace=True)

    # TrueDiv Operator

    def __truediv__(self, rhs: Any) -> "Chunk[np.float_]":
        return self.join(rhs, func=operator.truediv, dtype=np.float_)

    def __itruediv__(self, rhs: Any) -> "Chunk[np.float_]":
        return self.join(rhs, func=operator.itruediv, dtype=np.float_, inplace=True)

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
    def sum(self) -> _VT:
        ...

    @overload
    def sum(self, dtype: Type[_OT]) -> _OT:
        ...

    @overload
    def sum(self, dtype: npt.DTypeLike) -> Any:
        ...

    def sum(self, dtype: Type[_OT] | npt.DTypeLike = None) -> _VT | _OT | Any:
        if self._is_filled:
            assert isinstance(self._value, SupportsDunderMul)
            val = self._value * (self._size**3)
            if dtype is None:
                return val
            else:
                _dtype = np.dtype(dtype)
                return _dtype.type(val)
        else:
            return np.sum(self._value, dtype=dtype)

    def unique(self) -> npt.NDArray[_VT]:
        if self._is_filled:
            return np.asanyarray([self._value])
        else:
            return np.unique(self._value)

    @classmethod
    def _stack(
        cls, chunks: Sequence["Chunk[_VT]"], dtype: npt.DTypeLike, fill_value: npt.NDArray[_VT] | ellipsis = ...
    ) -> "Chunk[npt.NDArray[_VT]]":
        _dtype = np.dtype(dtype)
        index = chunks[0]._index
        size = chunks[0]._size

        # _fill_value: npt.NDArray[_VT] = np.array([c._fill_value for c in chunks], dtype=dtype) if fill_value is ... else fill_value

        new_chunk: "Chunk[npt.NDArray[_VT]]" = Chunk(index, size, _dtype, fill_value)
        if all(c._is_filled for c in chunks):
            new_chunk.fill(np.array([c.value for c in chunks], dtype=_dtype.base))
        else:
            arr = np.array([c.to_array() for c in chunks], dtype=_dtype.base).transpose((1, 2, 3, 0))
            new_chunk.set_array(arr)
        return new_chunk
