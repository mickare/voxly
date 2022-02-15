"""
Module that contains the ChunkGrid class
"""

import functools
import operator
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    overload,
    SupportsAbs,
)

import numpy as np
import numpy.typing as npt

from voxly.iterators import SliceOpt, VoxelGridIterator

from .faces import ChunkFace
from .index_dict import IndexDict
from .typing import Index3, LenType, Arr3i, Vec3i
from .chunk import Chunk, stack_chunks


ChunkIndex = Index3

_VT_co = TypeVar("_VT_co", covariant=True, bound=np.generic)
_OT_co = TypeVar("_OT_co", covariant=True, bound=np.generic)


class ChunkGrid(Generic[_VT_co]):
    __slots__ = ("_chunk_size", "_dtype", "_fill", "chunks")

    _chunk_size: int
    _dtype: np.dtype[Any]
    _fill: _VT_co
    chunks: IndexDict[Chunk[_VT_co]]

    def __init__(self, chunk_size: int, dtype: np.dtype[_VT_co] | Type[_VT_co], fill: Any = ...) -> None:
        assert chunk_size > 0
        self._chunk_size = chunk_size
        self._dtype = np.dtype(dtype)
        self.fill = self._dtype.base.type() if fill is ... else fill
        self.chunks = IndexDict()

    @property
    def dtype(self) -> np.dtype[Any]:
        return self._dtype

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        s = self._chunk_size
        return s, s, s

    @property
    def fill(self) -> _VT_co:
        return self._fill

    @fill.setter
    def fill(self, value: Any) -> None:
        _dtype = self._dtype
        if _dtype.subdtype:
            self._fill = np.array((value,), _dtype)[0]  # Hacky way to ensure the data type matches
        else:
            self._fill = _dtype.base.type(value)

    def size(self) -> Arr3i:
        return self.chunks.size() * self._chunk_size

    def _new_chunk_factory(self, index: Index3) -> Chunk[_VT_co]:
        return Chunk(index, self._chunk_size, self._dtype, self._fill)

    def chunk_index(self, pos: Vec3i) -> ChunkIndex:
        res = np.asarray(pos, dtype=np.int_) // self._chunk_size
        assert res.shape == (3,)
        return tuple(res)  # type: ignore

    def chunk_at_pos(self, pos: Arr3i) -> Chunk[_VT_co] | None:
        return self.chunks.get(self.chunk_index(pos))

    def ensure_chunk_at_index(self, index: ChunkIndex, *, insert: bool = True) -> Chunk[_VT_co]:
        return self.chunks.create_if_absent(index, self._new_chunk_factory, insert=insert)

    def ensure_chunk_at_pos(self, pos: Vec3i, insert: bool = True) -> Chunk[_VT_co]:
        return self.ensure_chunk_at_index(self.chunk_index(pos), insert=insert)

    def new_empty_mask(self, default: bool = False) -> npt.NDArray[np.bool8]:
        return np.full(self.chunk_shape, default, dtype=np.bool8)

    @classmethod
    def iter_neighbors_indices(cls, index: ChunkIndex) -> Iterator[Tuple[ChunkFace, Index3]]:
        yield from ((f, tuple(np.add(index, f.direction()))) for f in ChunkFace)  # type: ignore

    def iter_neighbors(
        self, index: ChunkIndex, flatten: bool = False
    ) -> Iterator[Tuple[ChunkFace, Optional[Chunk[_VT_co]]]]:
        if flatten:
            yield from ((f, c) for f, c in self.iter_neighbors(index, False) if c is not None)
        else:
            yield from ((f, self.chunks.get(i, None)) for f, i in self.iter_neighbors_indices(index))

    def __bool__(self) -> bool:
        raise ValueError(
            f"The truth value of {self.__class__} is ambiguous. "
            "Use a.any(), or a.all(), or wrap the comparison (0 < a) & (a < 0)"
        )

    def all(self) -> bool:
        """True if all chunks contain only True values"""
        return all(c.all() for c in self.chunks.values())

    def any(self) -> bool:
        """True if any chunk contains any True value"""
        return any(c.any() for c in self.chunks.values())

    def to_dense(self, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None) -> npt.NDArray[_VT_co]:
        return self.to_dense_with_offset(x, y, z)[0]

    def to_dense_with_offset(
        self, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None
    ) -> Tuple[npt.NDArray[_VT_co], Arr3i]:
        """Convert the grid to a dense numpy array"""
        if len(self.chunks) == 0:
            return np.empty((0, 0, 0), dtype=self._dtype), np.zeros(3, dtype=np.int_)

        # Variable cache
        cs = self._chunk_size

        index_min, index_max = self.chunks.box.minmax
        pos_min = np.multiply(index_min, cs)
        pos_max = np.multiply(index_max, cs) + cs
        voxel_it = VoxelGridIterator(pos_min, pos_max, x, y, z, clip=False)

        chunk_it = voxel_it // cs
        chunk_min: Arr3i = np.asarray(chunk_it.start)
        chunk_max: Arr3i = np.asarray(chunk_it.stop)
        chunk_len: Arr3i = chunk_max - chunk_min

        res: npt.NDArray[_VT_co] = np.full(tuple(chunk_len * cs), self.fill, dtype=self.dtype)

        # Method cache (prevent lookup in loop)
        __self_chunks_get = self.chunks.get
        __chunk_to_array = Chunk.to_array

        u: int
        v: int
        w: int

        for index in chunk_it:
            c = __self_chunks_get(index, None)
            if c is not None:
                u, v, w = np.subtract(index, chunk_min) * cs
                res[u : u + cs, v : v + cs, w : w + cs] = __chunk_to_array(c)

        start = voxel_it.start - chunk_min * cs
        stop = voxel_it.stop - chunk_min * cs
        step = voxel_it.step

        return (
            res[start[0] : stop[0] : step[0], start[1] : stop[1] : step[1], start[2] : stop[2] : step[2]],
            chunk_min * cs,
        )

    def astype(self, dtype: Type[_OT_co] | np.dtype[_OT_co], copy: bool = False) -> "ChunkGrid[_OT_co]":
        _dtype = np.dtype(dtype)
        if not copy and self._dtype == _dtype:
            return self  # type: ignore
        new_grid: ChunkGrid[_OT_co] = ChunkGrid(self._chunk_size, dtype, fill=self._fill)
        __new_grid_chunks_insert = new_grid.chunks.insert  # Cache method lookup
        for src in self.chunks.values():
            __new_grid_chunks_insert(src.index, src.astype(dtype))
        return new_grid

    def copy(self, empty: bool = False) -> "ChunkGrid[_VT_co]":
        new_grid = ChunkGrid(self._chunk_size, self._dtype, self._fill)
        if not empty:
            __new_grid_chunks_insert = new_grid.chunks.insert  # Cache method lookup
            for src in self.chunks.values():
                __new_grid_chunks_insert(src.index, src.copy())
        return new_grid

    def split(self, splits: int, chunk_size: int | None = None) -> "ChunkGrid[_VT_co]":
        assert splits > 0 and self._chunk_size % splits == 0
        chunk_size = chunk_size or self._chunk_size
        new_grid: ChunkGrid[_VT_co] = ChunkGrid(chunk_size, self._dtype, self.fill)

        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert

        for c in self.chunks.values():
            for c_new in c.split(splits, chunk_size):
                __new_grid_chunks_insert(c_new.index, c_new)
        return new_grid

    def items(self, mask: "ChunkGrid[np.bool8]" | None = None) -> Iterator[Tuple[Arr3i, _VT_co]]:
        if mask is None:
            for i, c in self.chunks.items():
                yield from c.items()
        else:
            __mask_ensure_chunk_at_index = mask.ensure_chunk_at_index  # Cache method
            for i, c in self.chunks.items():
                m = __mask_ensure_chunk_at_index(i, insert=False)
                if m.any_if_filled():
                    yield from c.items(m)

    def filter(self, other: "ChunkGrid[np.bool8]") -> "ChunkGrid[_VT_co]":
        """Apply a filter mask to this grid and return the masked values"""
        result = self.copy(empty=True)

        # Method cache (prevent lookup in loop)
        __self_chunks_get = self.chunks.get
        __chunk_any = Chunk.any
        __result_chunks_insert = result.chunks.insert

        for i, o in other.chunks.items():
            c = __self_chunks_get(i, None)
            if c is not None and __chunk_any(c):
                __result_chunks_insert(i, c.filter(o))
        return result

    def _argwhere_iter_arrays(self, mask: Optional["ChunkGrid[np.bool8]"] = None) -> Iterator[npt.NDArray[np.int_]]:
        # Method cache (prevent lookup in loop)
        __chunk_where = Chunk.argwhere
        if mask is None:
            for i, c in self.chunks.items():
                yield __chunk_where(c)
        else:
            # Method cache (prevent lookup in loop)
            __chunk_any_fast = Chunk.any_if_filled
            __mask_ensure_chunk_at_index = mask.ensure_chunk_at_index
            for i, c in self.chunks.items():
                m = __mask_ensure_chunk_at_index(i, insert=False)
                if __chunk_any_fast(m):
                    yield __chunk_where(c, mask=m)

    def argwhere(self, mask: Optional["ChunkGrid[np.bool8]"] = None) -> Iterator[Arr3i]:
        # Method cache (prevent lookup in loop)
        __chunk_where = Chunk.argwhere
        if mask is None:
            for i, c in self.chunks.items():
                yield from __chunk_where(c)
        else:
            # Method cache (prevent lookup in loop)
            __chunk_any_fast = Chunk.any_if_filled
            __mask_ensure_chunk_at_index = mask.ensure_chunk_at_index
            for i, c in self.chunks.items():
                m = __mask_ensure_chunk_at_index(i, insert=False)
                if __chunk_any_fast(m):
                    yield from __chunk_where(c, mask=m)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            return self.to_dense(item)
        elif isinstance(item, tuple) and len(item) <= 3:
            return self.to_dense(*item)
        elif isinstance(item, ChunkGrid):
            return self.filter(item)
        elif isinstance(item, np.ndarray):
            return self.get_values(item)
        else:
            raise IndexError("Invalid get")

    def get_values(self, pos: Iterable[Arr3i] | npt.NDArray[np.int_]) -> npt.NDArray[_VT_co]:
        """Returns a list of values at the positions"""
        # Method cache (prevent lookup in loop)
        __np_argwhere = np.argwhere
        __self_ensure_chunk_at_index = self.ensure_chunk_at_index
        __chunk_to_array = Chunk.to_array

        pos = np.asarray(pos, dtype=int)
        assert pos.ndim == 2 and pos.shape[1] == 3
        csize: int = self._chunk_size

        cind: npt.NDArray[np.int_]
        cinv: npt.NDArray[np.int_]
        cind, cinv = np.unique(pos // csize, axis=0, return_inverse=True)
        result = np.zeros(len(cinv), dtype=self._dtype)
        for n, i in enumerate(cind):
            pind = __np_argwhere(cinv == n).flatten()
            cpos = pos[pind] % csize
            chunk = __self_ensure_chunk_at_index(i, insert=False)
            result[pind] = __chunk_to_array(chunk)[tuple(cpos.T)]
        return result

    def get_value(self, pos: Vec3i) -> _VT_co:
        idx = self.chunk_index(pos)
        c: Chunk[_VT_co] | None = self.chunks.get(idx)
        if c is None:
            return self.fill
        else:
            return c.get(pos)

    def set_value(self, pos: Vec3i, value: Any) -> None:
        c = self.ensure_chunk_at_pos(pos)
        c.set(pos, value)

    def set_or_fill(self, pos: Vec3i, value: Any) -> None:
        c = self.ensure_chunk_at_pos(pos)
        c.set_or_fill(pos, value)

    def _set_slices(self, value: Any, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None) -> None:
        it = VoxelGridIterator.require_bounded(x, y, z)

        if isinstance(value, np.ndarray):
            assert value.shape == it.shape
            if self._dtype is not None:
                value = value.astype(self._dtype)
            for i, pos in it.iter_with_indices():
                self.set_value(pos, value[i])
        else:
            for pos in it:
                self.set_value(pos, value)

    def _set_positions(self, pos: npt.NDArray[np.int_] | Sequence[Arr3i], value: _VT_co | Sequence[_VT_co]) -> None:
        if isinstance(pos, LenType):
            if not len(pos):
                return  # No Op
        pos = np.asarray(pos, dtype=int)
        if len(pos) == 0:
            return  # No Op
        if pos.shape == (3,):
            self.set_value(pos, value)
        else:
            assert pos.ndim == 2 and pos.shape[1] == 3, f"shape={pos.shape}"
            if isinstance(value, (list, tuple, np.ndarray)):
                assert len(pos) == len(value)
                for p, v in zip(pos, value):
                    self.set_value(p, v)
            else:
                upos: npt.NDArray[np.int_] = np.unique(pos, axis=0)
                for p in upos:
                    self.set_value(p, value)

    def _set_chunks(
        self, mask: "ChunkGrid[np.bool8]", value: _VT_co | npt.NDArray[_VT_co] | Chunk[_VT_co] | "ChunkGrid[_VT_co]"
    ) -> None:
        assert self._chunk_size == mask._chunk_size
        # Method cache (prevent lookup in loop)
        __self_ensure_chunk_at_index = self.ensure_chunk_at_index

        if isinstance(value, ChunkGrid):
            indices = set(mask.chunks.keys())
            if mask.fill:
                indices.update(self.chunks.keys())
                indices.update(value.chunks.keys())

            __mask_ensure_chunk_at_index = mask.ensure_chunk_at_index
            __value_ensure_chunk_at_index = value.ensure_chunk_at_index
            for index in indices:
                m = __mask_ensure_chunk_at_index(index, insert=False)
                val = __value_ensure_chunk_at_index(index, insert=False)
                __self_ensure_chunk_at_index(index)[m] = val

            # Set fill value
            if mask.fill:
                self.fill = value.fill
        else:
            indices = set(mask.chunks.keys())
            if mask.fill:
                indices.update(self.chunks.keys())

            for index in indices:
                m = mask.ensure_chunk_at_index(index, insert=False)
                __self_ensure_chunk_at_index(index)[m] = value

            # Set fill value
            if mask.fill:
                if isinstance(value, Chunk):
                    self.fill = value.fill
                elif isinstance(value, np.ndarray):
                    if self._dtype.shape == value.shape:
                        self.fill = value  # type: ignore
                else:
                    self.fill = value

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(key, slice):
            self._set_slices(value, key)
        elif isinstance(key, tuple) and len(key) <= 3:
            self._set_slices(value, *key)
        elif isinstance(key, (np.ndarray, list)):
            self._set_positions(key, value)
        elif isinstance(key, ChunkGrid):
            self._set_chunks(key, value)
        else:
            raise IndexError("Invalid get")

    def cleanup(self, remove: bool = False) -> "ChunkGrid[_VT_co]":
        for chunk in self.chunks:
            chunk.cleanup()
        if remove:
            for chunk in list(self.chunks):
                if chunk.is_filled_with(self.fill):
                    del self.chunks[chunk.index]
        return self

    def pad_chunks(self, width: int = 1) -> "ChunkGrid[_VT_co]":
        visited: Set[ChunkIndex] = set()
        for s in range(0, width):
            extra: Set[ChunkIndex] = set(n for i in self.chunks.keys() for f, n in self.iter_neighbors_indices(i))
            extra = extra.difference(visited)
            for e in extra:
                self.ensure_chunk_at_index(e)
            visited.update(extra)
        return self

    def iter_hull(self) -> Iterator[Chunk[_VT_co]]:
        """Iter some of the outer chunks that represent the hull around all chunks"""
        if self.chunks:
            __self_chunks_get = self.chunks.get  # Cache method
            it = self.chunks.sliced_iterator()
            for x in it.x.range():
                for y in it.y.range():
                    for z in it.z.range():
                        c = __self_chunks_get((x, y, z), None)
                        if c is not None:
                            yield c
                            break
            for x in reversed(it.x.range()):
                for y in reversed(it.y.range()):
                    for z in reversed(it.z.range()):
                        c = __self_chunks_get((x, y, z), None)
                        if c is not None:
                            yield c
                            break

    def get_neigbors_at(
        self,
        index: ChunkIndex,
        neighbors: bool = True,
        edges: bool = True,
        corners: bool = True,
        insert: bool = False,
        ensure: bool = True,
    ) -> npt.NDArray[np.object_]:
        __face_direction = ChunkFace.direction

        __self_chunk_get: Callable[[Index3], Chunk[_VT_co]]
        if ensure:
            __self_chunk_get = functools.partial(self.ensure_chunk_at_index, insert=insert)
        else:
            __self_chunk_get = self.chunks.get  # type: ignore

        idx: npt.NDArray[np.int_] = np.array(index, dtype=int)

        chunks: npt.NDArray[np.object_] = np.full((3, 3, 3), None, dtype=np.object_)
        chunks[1, 1, 1] = __self_chunk_get(index)
        if neighbors:
            for face, index in self.iter_neighbors_indices(index):
                u, v, w = __face_direction(face) + 1
                chunks[u, v, w] = __self_chunk_get(index)
        if edges:
            # Add edges
            for a, b in ChunkFace.edges():
                d = __face_direction(a) + __face_direction(b)
                u, v, w = d + 1
                chunks[u, v, w] = __self_chunk_get(tuple(idx + d))  # type: ignore
        if corners:
            # Add corners
            for a, b, c in ChunkFace.corners():
                d = __face_direction(a) + __face_direction(b) + __face_direction(c)
                u, v, w = d + 1
                chunks[u, v, w] = __self_chunk_get(tuple(idx + d))  # type: ignore
        return chunks

    def get_block_at(
        self,
        index: ChunkIndex,
        shape: Tuple[int, int, int],
        *,
        offset: Optional[Arr3i] = None,
        edges: bool = True,
        corners: bool = True,
        ensure: bool = True,
        insert: bool = False,
    ) -> npt.NDArray[np.object_]:
        assert len(shape) == 3

        # Method cache
        __face_direction = ChunkFace.direction

        __self_chunks_get: Callable[[Vec3i], Chunk[_VT_co] | None]
        if ensure:
            __self_chunks_get = functools.partial(self.ensure_chunk_at_index, insert=insert)
        else:
            __self_chunks_get = self.chunks.get

        _shape: npt.NDArray[np.int_] = np.asarray(shape)
        chunks: npt.NDArray[np.object_] = np.full(shape, None, dtype=np.object_)
        offset = _shape // 2 if offset is None else np.asarray(offset, dtype=np.int_)
        assert offset.shape == (3,)

        # Corner/Edge case handling
        low: npt.NDArray[np.int_] = np.zeros(3, dtype=int)
        high: npt.NDArray[np.int_] = low + _shape - 1

        ignore_chunks = (not edges) or (not corners)

        for i in np.ndindex(shape):
            chunk_pos = np.add(index, i) - offset
            if ignore_chunks:
                s = np.sum(i == low) + np.sum(i == high)
                if not edges and s == 2:
                    continue
                if not corners and s == 3:
                    continue
            chunks[i] = __self_chunks_get(chunk_pos)

        return chunks

    # Operators

    def apply(
        self,
        func: Callable[[Chunk[_VT_co] | _VT_co], Chunk[_OT_co] | _OT_co],
        dtype: Type[_OT_co] | npt.DTypeLike = None,
        into: "ChunkGrid[_OT_co]" | None = None,
    ) -> "ChunkGrid[_OT_co]":
        # Apply fill value
        _fill: Any = func(self._fill)
        # Create new grid
        if into is None:
            if dtype is None:
                _dtype: np.dtype[_OT_co] = np.asarray(_fill).dtype
            else:
                _dtype = np.dtype(dtype)  # type: ignore
            grid: "ChunkGrid[_OT_co]" = ChunkGrid(self._chunk_size, _dtype, fill=_fill)
        else:
            grid = into
            grid.fill = _fill

        # Apply on chunks
        __grid_chunks_insert = grid.chunks.insert  # Cache method
        for i, a in self.chunks.items():
            new_chunk = func(a)
            assert isinstance(new_chunk, Chunk)
            __grid_chunks_insert(i, new_chunk)

        return grid

    def outer_join(
        self,
        rhs: Any,
        func: Callable[[Chunk[_VT_co] | _VT_co, Chunk[_VT_co] | _VT_co], Chunk[_OT_co] | _OT_co],
        dtype: Type[_OT_co] | np.dtype[_OT_co] | None = None,
        into: "ChunkGrid[_OT_co]" | None = None,
    ) -> "ChunkGrid[_OT_co]":

        # Join fill value
        if isinstance(rhs, ChunkGrid):
            _fill = func(self._fill, rhs._fill)
        else:
            _fill = func(self._fill, rhs)

        # Create new grid
        if into is None:
            if dtype is None:
                _dtype: np.dtype[_OT_co] = np.asarray(_fill).dtype
            else:
                _dtype = np.dtype(dtype)  # type: ignore
            grid: "ChunkGrid[_OT_co]" = ChunkGrid(self._chunk_size, _dtype, fill=_fill)
        else:
            grid = into
            grid.fill = _fill  # type: ignore

        # Join chunks
        __grid_chunks_insert = grid.chunks.insert  # Cache method
        if isinstance(rhs, ChunkGrid):
            assert grid._chunk_size == rhs._chunk_size
            indices = set(self.chunks.keys())
            indices.update(rhs.chunks.keys())

            _self_ensure_chunk_at_index = self.ensure_chunk_at_index  # Cache method
            _rhs_ensure_chunk_at_index = rhs.ensure_chunk_at_index  # Cache method

            for i in indices:
                a = _self_ensure_chunk_at_index(i, insert=False)
                b = _rhs_ensure_chunk_at_index(i, insert=False)
                new_chunk = func(a, b)
                assert isinstance(new_chunk, Chunk)
                __grid_chunks_insert(i, new_chunk)
        else:
            for i, a in self.chunks.items():
                new_chunk = func(a, rhs)
                assert isinstance(new_chunk, Chunk)
                __grid_chunks_insert(i, new_chunk)

        return grid

    # Comparison Operator

    def __eq__(self, rhs: Any) -> "ChunkGrid[np.bool8]":  # type: ignore[override]
        return self.outer_join(rhs, func=operator.eq, dtype=np.bool8)

    def __ne__(self, rhs: Any) -> "ChunkGrid[np.bool8]":  # type: ignore[override]
        return self.outer_join(rhs, func=operator.ne, dtype=np.bool8)

    def __lt__(self, rhs: Any) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.lt, dtype=np.bool8)

    def __le__(self, rhs: Any) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.le, dtype=np.bool8)

    def __gt__(self, rhs: Any) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.gt, dtype=np.bool8)

    def __ge__(self, rhs: Any) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.ge, dtype=np.bool8)

    eq = __eq__
    ne = __ne__
    lt = __lt__
    le = __le__
    gt = __gt__
    ge = __ge__

    equals = __eq__

    # Single Operator

    def __abs__(self: "ChunkGrid[SupportsAbs[_VT_co]]") -> "ChunkGrid[_VT_co]":  # type: ignore[type-var]
        return self.apply(operator.abs)  # type: ignore

    def __invert__(self) -> "ChunkGrid[_VT_co]":
        return self.apply(operator.inv)

    def __neg__(self) -> "ChunkGrid[_VT_co]":
        return self.apply(operator.neg)

    abs = __abs__
    invert = __invert__
    neg = __neg__

    # Logic Operator

    def __and__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.and_, dtype=self._dtype)

    def __or__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.or_, dtype=self._dtype)

    def __xor__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.xor, dtype=self._dtype)

    def __iand__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.iand, dtype=self._dtype, into=self)

    def __ior__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.ior, dtype=self._dtype, into=self)

    def __ixor__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.ixor, dtype=self._dtype, into=self)

    and_ = __and__
    or_ = __or__
    xor = __xor__

    # Math Operator

    def __add__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.add)

    def __sub__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.sub)

    def __mul__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.mul)

    def __matmul__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.matmul)

    def __mod__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.mod)

    def __pow__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.pow)

    def __floordiv__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.floordiv)

    def __iadd__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.iadd, into=self)

    def __isub__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.isub, into=self)

    def __imul__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.imul, into=self)

    def __imatmul__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.imatmul, into=self)

    def __imod__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.imod, into=self)

    def __ifloordiv__(self, rhs: Any) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.ifloordiv, into=self)

    # TrueDiv Operator

    def __truediv__(self, rhs: "ChunkGrid[_VT_co]" | npt.NDArray[_VT_co] | _VT_co) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.truediv, dtype=self._dtype)

    def __itruediv__(self, rhs: "ChunkGrid[_VT_co]" | npt.NDArray[_VT_co] | _VT_co) -> "ChunkGrid[_VT_co]":
        return self.outer_join(rhs, func=operator.itruediv, dtype=self._dtype, into=self)

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
    def sum(self, dtype: Type[_OT_co] | np.dtype[_OT_co]) -> _OT_co:
        ...

    def sum(self, dtype: Type[_OT_co] | npt.DTypeLike = None) -> Any:
        return sum(c.sum(dtype) for c in self.chunks)  # type: ignore

    @classmethod
    def stack(cls, grids: Sequence["ChunkGrid[_VT_co]"]) -> "ChunkGrid[_VT_co]":
        # assert len(grids) > 0
        # if len(grids) == 1:
        #     return grids[0]

        # Check that grid size matches!
        chunk_size = grids[0].chunk_size
        assert all(chunk_size == g.chunk_size for g in grids)

        # Get datatype
        dtypes: List[np.dtype[Any]] = [g.dtype for g in grids]
        if any(t != dtypes[0] for t in dtypes):
            raise ValueError("Mixed data type grids not supported!")
        assert dtypes[0] != np.void
        _dtype = np.dtype((dtypes[0].base, len(grids)))

        _fill: npt.NDArray[_VT_co] = np.array([g.fill for g in grids], _dtype)
        new_grid = ChunkGrid(chunk_size, dtype=_dtype, fill=_fill)

        indices = set(k for g in grids for k in g.chunks.keys())
        for ind in indices:
            new_grid.chunks.insert(
                ind, stack_chunks([g.ensure_chunk_at_index(ind, insert=False) for g in grids], dtype=_dtype)
            )
        return new_grid

    def set_block_at(
        self,
        index: ChunkIndex,
        data: npt.NDArray[_VT_co],
        op: Callable[[Chunk[_VT_co], npt.NDArray[_VT_co]], Any] | None = None,
        replace: bool = True,
    ) -> None:
        index_arr: npt.NDArray[np.int_] = np.asarray(index, dtype=int)
        size: int = self.chunk_size
        data_shape = np.array(data.shape, dtype=int)[:3]
        assert np.all(data_shape % size == 0)
        block_shape = tuple(data_shape // size)
        block_offset = np.array(block_shape, dtype=int) // 2

        op = op or Chunk.set_array

        if replace:
            for u in range(block_shape[0]):
                for v in range(block_shape[1]):
                    for w in range(block_shape[2]):
                        s0 = np.array((u, v, w), int) * size
                        s1 = s0 + size
                        chunk_index = index_arr - block_offset + (u, v, w)
                        chunk = self.ensure_chunk_at_index(tuple(chunk_index))  # type: ignore
                        op(chunk, data[s0[0] : s1[0], s0[1] : s1[1], s0[2] : s1[2]])

        else:
            for u in range(block_shape[0]):
                for v in range(block_shape[1]):
                    for w in range(block_shape[2]):
                        chunk_index = index_arr - block_offset + (u, v, w)
                        chunk = self.chunks.get(chunk_index)  # type: ignore
                        if chunk is None:
                            s0 = np.array((u, v, w), int) * size
                            s1 = s0 + size
                            chunk = self.ensure_chunk_at_index(tuple(chunk_index))
                            op(chunk, data[s0[0] : s1[0], s0[1] : s1[1], s0[2] : s1[2]])

    def padding_at(
        self,
        index: ChunkIndex,
        padding: int = 1,
        corners: bool = True,
        edges: bool = True,
    ) -> npt.NDArray[_VT_co]:
        assert padding >= 0
        if padding == 0:
            return self.ensure_chunk_at_index(index, insert=False).to_array()

        chunk_size = self._chunk_size
        pad_chunks = int(np.ceil(padding / chunk_size))
        pad_size = 1 + pad_chunks * 2
        pad_outer = (chunk_size - padding) % chunk_size
        chunks = self.get_block_at(index, (pad_size, pad_size, pad_size), corners=corners, edges=edges)
        block = self.block_to_array(chunks)
        if pad_outer:
            return block[pad_outer:-pad_outer, pad_outer:-pad_outer, pad_outer:-pad_outer]  # type: ignore
        return block

    def block_to_array(self, chunks: npt.NDArray[np.object_]) -> npt.NDArray[_VT_co]:
        chunks = np.atleast_3d(np.asarray(chunks, dtype=np.object_))
        cs = self._chunk_size
        dtype = self._dtype
        chunk_shape = (cs, cs, cs)

        def _to_array(chunk: Chunk[_VT_co] | None) -> npt.NDArray[_VT_co]:
            if chunk is None:
                return np.full(chunk_shape, self._fill, dtype=self._dtype)
            return chunk.to_array()

        data = [
            [[_to_array(chunks[u, v, w]) for w in range(chunks.shape[2])] for v in range(chunks.shape[1])]
            for u in range(chunks.shape[0])
        ]

        if dtype.subdtype:
            return np.concatenate(
                [np.concatenate([np.concatenate(v, axis=2) for v in u], axis=1) for u in data], axis=0
            )
        else:
            return np.block(data)

    def unique(self) -> npt.NDArray[_VT_co]:
        if not self.chunks:
            return np.empty((0,), dtype=self._dtype)
        return np.unique(np.concatenate([c.unique() for c in self.chunks]))
