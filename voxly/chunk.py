import functools
import operator
from typing import (
    Callable,
    Generic,
    Iterator,
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

from .typing import BoolType, Index3, Vec3i

V = TypeVar("V", bound=np.generic, covariant=True)
OtherV = TypeVar("OtherV", bound=np.generic, covariant=True)


class ChunkHelper:
    class _IndexMeshGrid:
        def __getitem__(self, item) -> Iterator[Index3]:
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
        result = np.vstack((xs, ys, zs)).T
        result.flags.writeable = False
        return result


class Chunk(Generic[V]):
    __slots__ = ("_index", "_size", "_dtype", "_fill_value", "_is_filled", "_value")

    def __init__(
        self,
        index: Vec3i,
        size: int,
        dtype: np.dtype[V] = None,
        fill_value: Optional[V] = None,
    ) -> None:
        self._index = np.asarray(index, dtype=np.int)
        self._size = size
        self._dtype = np.dtype(dtype)
        self._fill_value = self._dtype.base.type(fill_value)
        self._is_filled = True
        self._value: Union[V, np.ndarray] = self._fill_value

    @property
    def index(self) -> Vec3i:
        return self._index

    @property
    def value(self) -> Union[V, npt.NDArray[V]]:
        return self._value

    @property
    def size(self) -> int:
        return self._size

    @property
    def dtype(self) -> np.dtype:
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
    def position_low(self) -> Vec3i:
        return np.multiply(self._index, self._size)

    @property
    def position_high(self) -> Vec3i:
        return self.position_low + self._size

    def is_filled(self) -> bool:
        return self._is_filled

    def is_filled_with(self, value: V):
        return np.all(self._value == value)

    def is_array(self) -> bool:
        return not self._is_filled

    def index_inner(self, pos: Vec3i) -> Vec3i:
        return np.asarray(pos, dtype=np.int) % self._size

    def to_array(self) -> npt.NDArray[V]:
        if self._is_filled:
            return np.full(self.shape, self._value, dtype=self._dtype)
        else:
            return self._value

    def set_array(self, value: npt.NDArray[V]) -> "Chunk[V]":
        if isinstance(value, np.ndarray):
            value = value.astype(self._dtype.base)
        else:
            value = np.asarray(value, dtype=self._dtype.base)
        assert self.array_shape == value.shape, f"shape mismatch {self.array_shape} != {value.shape}"
        self._value = value
        self._is_filled = False
        return self

    def get(self, pos: Vec3i) -> V:
        if self._is_filled:
            return self._value
        return self.to_array()[tuple(self.index_inner(pos))]

    def put(self, pos: Vec3i, value: V):
        inner = self.index_inner(pos)
        arr = self.to_array()
        arr[tuple(inner)] = value
        self.set_array(arr)

    def put_or_fill(self, pos: Vec3i, value: V):
        if self._is_filled:
            self.fill(value)
        else:
            self.put(pos, value)

    def fill(self, value: V) -> "Chunk[V]":
        dtype = self._dtype
        if not dtype.subdtype:
            value = dtype.type(value)
        else:
            try:
                value = dtype.base.type(value)
            except Exception:
                pass
        self._value = value
        self._is_filled = True
        return self

    def cleanup(self):
        """Try to reduce memory footprint"""
        if self.is_array():
            u = np.unique(self._value)
            if len(u) == 1:
                self.fill(u.item())
        return self

    def astype(self, dtype: Type[OtherV]) -> "Chunk[OtherV]":
        if self._dtype == dtype:
            return self
        c = Chunk(self._index, self._size, dtype=dtype, fill_value=dtype(self._fill_value))
        if self.is_filled():
            c.fill(dtype(self._value))
        else:
            c.set_array(self._value.astype(dtype))
        return c

    def __bool__(self):
        raise ValueError(
            f"The truth value of {__class__} is ambiguous. "
            "Use a.any(), or a.all(), or wrap the comparison (0 < a) & (a < 0)"
        )

    def all(self) -> bool:
        return np.all(self._value)

    def any(self) -> bool:
        return np.any(self._value)

    def any_if_filled(self) -> bool:
        if self.is_filled():
            return bool(self._value)
        return True

    @overload
    def copy(self) -> "Chunk[V]":
        ...

    @overload
    def copy(self, empty: bool) -> "Chunk[V]":
        ...

    @overload
    def copy(self, empty: bool, fill_value: V) -> "Chunk[V]":
        ...

    @overload
    def copy(self, empty: bool, fill_value: OtherV, dtype: np.dtype[OtherV]) -> "Chunk[OtherV]":
        ...

    def copy(self, empty=False, fill_value=None, dtype=None):
        dtype = self._dtype if dtype is None else np.dtype(dtype)
        fill_value = self._fill_value if fill_value is None else fill_value
        
        c = Chunk(self._index, self._size, dtype=dtype, fill_value=fill_value)
        if not empty:
            if self.is_filled():
                value = self._value
                if dtype.subdtype:
                    value = np.copy(value)
                c.fill(value)
            else:
                c.set_array(self._value.copy())
        return c

    def filter(self, mask: "Chunk[bool]", fill_value: Optional[V] = None) -> "Chunk[V]":
        mask: Chunk[bool] = mask.astype(bool)
        c = self.copy(empty=True, fill_value=fill_value)
        if mask._is_filled and mask._value:
            c.set_fill(self._value)
        else:
            arr = np.full(self.shape, c._fill_value, dtype=self._dtype)
            arr[mask._value] = self._value[mask._value]
            c.set_array(arr)
        return c

    def items(self, mask: "Chunk[bool]" = None) -> Iterator[Tuple[Vec3i, V]]:
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

    def argwhere(self, mask: "Chunk[bool]" | npt.NDArray[np.bool_] = None) -> npt.NDArray[np.int_]:
        if isinstance(mask, Chunk):
            if mask.is_filled() and not bool(mask.value):
                return np.empty((0, 3), dtype=int)
            mask_arr = mask.astype(bool)
            idx = self._argwhere()
            return idx[mask_arr[idx]]
        if isinstance(mask, np.ndarray):
            assert self.shape == mask.shape, f"shape mismatch {self.shape} != {mask.shape}"
            idx = self._argwhere()
            return idx[mask[idx]]
        if mask is None:
            return self._argwhere()
        raise ValueError(f"invalid mask of type {type(mask)}")

    @overload
    def __getitem__(self, item: Index3) -> V:
        ...

    @overload
    def __getitem__(self, item: Sequence[Index3]) -> npt.NDArray[V]:
        ...

    @overload
    def __getitem__(self, item: npt.NDArray[np.int_]) -> npt.NDArray[V]:
        ...

    def __getitem__(self, item):
        if isinstance(item, Chunk):
            return self.filter(item)
        return self.to_array()[item]

    def __setitem__(self, key: npt.NDArray[np.int_] | "Chunk[bool]", value: V | npt.NDArray[V] | "Chunk[V]") -> None:
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
            c = Chunk(new_index, size=chunk_size, dtype=dtype, fill_value=self._fill_value)
            if self.is_filled():
                c.fill(self._value)
            else:
                u, v, w = np.asarray(offset, dtype=np.int) * split_size
                tmp = self._value[u : u + split_size, v : v + split_size, w : w + split_size]
                if repeats == 1:
                    val = tmp.copy()
                else:
                    val = np.repeat(np.repeat(np.repeat(tmp, repeats, axis=0), repeats, axis=1), repeats, axis=2)
                c.set_array(val)
            yield c

    def convert(
        self, func: Callable[[V], OtherV], 
        func_vec: Callable[[npt.NDArray[V]], npt.NDArray[OtherV]] = None
    ) -> "Chunk[OtherV]":
        c = Chunk(self._index, self._size, fill_value=func(self._fill_value))
        if self.is_filled():
            c.fill(func(self._value))
        else:
            func_vec = func_vec or np.vectorize(func)
            c.set_array(func_vec(self._value))
        return c

    def apply(
        self, 
        func: Callable[[np.ndarray | V], np.ndarray | OtherV], 
        dtype: Type[OtherV] = None, 
        inplace=False
    ) -> "Chunk[OtherV]":
        dtype = self._dtype if dtype is None else np.dtype(dtype)

        # Fill value
        fill_value = self._fill_value
        try:
            fill_value = func(fill_value)
        except Exception as e:
            handling = np.geterr()["invalid"]
            if handling == "raise":
                raise e
            elif handling == "ignore":
                pass
            else:
                warnings.warn("Fill value operand", RuntimeWarning, source=e)

        # Inplace selection
        if inplace:
            assert self._dtype == dtype
            c = self
            c._fill_value = fill_value
        else:
            c = self.copy(empty=True, dtype=dtype, fill_value=fill_value)

        # Func on value
        if self._is_filled:
            c.set_fill(func(self._value))
        else:
            c.set_array(func(self._value))
        return c

    def join(
        self, rhs,
        func: Callable[[Union[np.ndarray, V], Union[np.ndarray, V]], Union[np.ndarray, OtherV]],
        dtype: Optional[Type[OtherV]] = None,
        inplace=False,
    ) -> "Chunk[OtherV]":
        dtype = self._dtype if dtype is None else np.dtype(dtype)

        rhs_is_chunk = isinstance(rhs, Chunk)

        # Fill value
        rhs_fill_value = rhs._fill_value if rhs_is_chunk else rhs
        fill_value = self._fill_value
        try:
            fill_value = func(self._fill_value, rhs_fill_value)
        except Exception as e:
            handling = np.geterr()["invalid"]
            if handling == "raise":
                raise e
            elif handling == "ignore":
                pass
            else:
                warnings.warn("Fill value operand", RuntimeWarning, source=e)

        # Inplace selection
        if inplace:
            assert self._dtype == dtype
            c = self
            c._fill_value = fill_value
        else:

            c = self.copy(empty=True, dtype=dtype, fill_value=fill_value)

        # Func on values
        if rhs_is_chunk:
            val = func(self._value, rhs._value)
            if self._is_filled and rhs._is_filled:
                c.set_fill(val)
            else:
                c.set_array(val)
        else:
            val = func(self._value, rhs)
            if self._is_filled:
                c.set_fill(func(self._value, rhs))
            else:
                c.set_array(func(self._value, rhs))

        return c

    # Comparison Operator

    def __eq__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.eq, dtype=np.bool8).cleanup()

    def __ne__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.ne, dtype=np.bool8).cleanup()

    def __lt__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.lt, dtype=np.bool8).cleanup()

    def __le__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.le, dtype=np.bool8).cleanup()

    def __gt__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.gt, dtype=np.bool8).cleanup()

    def __ge__(self, rhs) -> "Chunk[np.bool8]":
        return self.join(rhs, func=operator.ge, dtype=np.bool8).cleanup()

    eq = __eq__
    ne = __ne__
    lt = __lt__
    le = __le__
    gt = __gt__
    ge = __ge__

    equals = __eq__

    # Single Operator

    def __abs__(self) -> "Chunk[V]":
        return self.apply(func=operator.abs)

    def __invert__(self) -> "Chunk[V]":
        return self.apply(func=operator.inv)

    def __neg__(self) -> "Chunk[V]":
        return self.apply(func=operator.neg)

    abs = __abs__
    invert = __invert__
    neg = __neg__

    # Logic Operator

    def __and__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.and_)

    def __or__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.or_)

    def __xor__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.xor)

    def __iand__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.iand, inplace=True)

    def __ior__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.ior, inplace=True)

    def __ixor__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.ixor, inplace=True)

    # Math Operator

    def __add__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.add)

    def __sub__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.sub)

    def __mul__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.mul)

    def __matmul__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.matmul)

    def __mod__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.mod)

    def __pow__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.pow)

    def __floordiv__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.floordiv)

    def __iadd__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.iadd, inplace=True)

    def __isub__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.isub, inplace=True)

    def __imul__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.imul, inplace=True)

    def __imatmul__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.imatmul, inplace=True)

    def __imod__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.imod, inplace=True)

    def __ifloordiv__(self, rhs) -> "Chunk[V]":
        return self.join(rhs, func=operator.ifloordiv, inplace=True)

    # TrueDiv Operator

    def __truediv__(self, rhs) -> "Chunk[float]":
        return self.join(rhs, func=operator.truediv, dtype=np.float)

    def __itruediv__(self, rhs) -> "Chunk[float]":
        return self.join(rhs, func=operator.itruediv, dtype=np.float, inplace=True)

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
    def sum(self) -> V: ...
    @overload
    def sum(self, dtype: Type[OtherV]) -> OtherV: ...

    def sum(self, dtype: Type[OtherV] = None):
        if self._is_filled:
            val = self._value * (self._size ** 3)
            return val if dtype is None else dtype(val)
        else:
            return np.sum(self._value, dtype=dtype)

    def unique(self) -> np.ndarray:
        if self._is_filled:
            return np.asanyarray([self._value])
        else:
            return np.unique(self._value)

    @classmethod
    def _stack(cls, chunks: Sequence["Chunk[V]"], dtype: np.dtype, fill_value=None) -> "Chunk[np.ndarray]":
        assert dtype.shape
        index = chunks[0]._index
        size = chunks[0]._size
        new_chunk = Chunk(index, size, dtype, fill_value)
        if all(c._is_filled for c in chunks):
            new_chunk.set_fill(np.array([c.value for c in chunks], dtype=dtype.base))
        else:
            arr = np.array([c.to_array() for c in chunks], dtype=dtype.base).transpose((1, 2, 3, 0))
            new_chunk.set_array(arr)
        return new_chunk