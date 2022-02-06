import operator
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, Sequence, Set, Tuple, Type, TypeVar

import numpy as np
import numpy.typing as npt

from voxly.iterators import SliceOpt, VoxelGridIterator

from .faces import ChunkFace
from .index_dict import IndexDict
from .typing import Index3, LenType, Vec3i
from .chunk import Chunk


ChunkIndex = Index3

_VT = TypeVar("_VT", bound=np.generic)
_OT = TypeVar("_OT", bound=np.generic)


class ChunkGrid(Generic[_VT]):
    __slots__ = ("_chunk_size", "_dtype", "_fill_value", "chunks")

    def __init__(self, chunk_size: int, dtype: np.dtype[_VT] = None, fill_value: _VT = None):
        assert chunk_size > 0
        self._chunk_size = chunk_size
        self._dtype = np.dtype(dtype)
        self._fill_value = self._dtype.base.type(fill_value)  # If this fails, dtype is an unsupported complex type
        self.chunks: IndexDict[Chunk[_VT]] = IndexDict()

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def chunk_shape(self) -> Tuple[int, int, int]:
        s = self._chunk_size
        return s, s, s

    @property
    def fill_value(self) -> _VT:
        return self._fill_value

    @fill_value.setter
    def fill_value(self, value: _VT):
        self._fill_value = value
        for c in self.chunks:
            c._fill_value = value

    def size(self) -> Vec3i:
        return self.chunks.size() * self._chunk_size

    def astype(self, dtype: Type[_OT]) -> "ChunkGrid[_OT]":
        if self._dtype == dtype:
            return self
        new_grid: ChunkGrid[_OT] = ChunkGrid(self._chunk_size, dtype, fill_value=dtype(self._fill_value))
        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert
        for src in self.chunks.values():
            __new_grid_chunks_insert(src.index, src.astype(dtype))
        return new_grid

    def convert(self, func: Callable[[_VT], _OT]) -> "ChunkGrid[_OT]":
        func_vec = np.vectorize(func)
        new_grid: ChunkGrid[_OT] = ChunkGrid(self._chunk_size, fill_value=func(self._fill_value))
        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert
        for src in self.chunks:
            __new_grid_chunks_insert(src.index, src.convert(func, func_vec))
        return new_grid

    def copy(self, empty=False, dtype: np.dtype | Type[_OT] = None, fill_value: _OT = None) -> "ChunkGrid[_OT]":
        # Get type
        dtype = self.__dtype(dtype)
        # Get fill value
        fill_value = self._fill_value if fill_value is None else fill_value
        try:
            fill_value = dtype.base.type(fill_value)
        except Exception:
            pass
        # Create new grid
        new_grid = ChunkGrid(self._chunk_size, dtype, fill_value)
        # Fill if wanted
        if not empty:
            # Method cache (prevent lookup in loop)
            __new_grid_chunks_insert = new_grid.chunks.insert
            for src in self.chunks.values():
                __new_grid_chunks_insert(src.index, src.copy(dtype=dtype, fill_value=fill_value))
        return new_grid

    def split(self, splits: int, chunk_size: int = None) -> "ChunkGrid[_VT]":
        assert splits > 0 and self._chunk_size % splits == 0
        chunk_size = chunk_size or self._chunk_size
        new_grid: ChunkGrid[_VT] = ChunkGrid(chunk_size, self._dtype, self._fill_value)

        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert

        for c in self.chunks.values():
            for c_new in c.split(splits, chunk_size):
                __new_grid_chunks_insert(c_new.index, c_new)
        return new_grid

    def _new_chunk_factory(self, index: Index3) -> Chunk[_VT]:
        return Chunk(index, self._chunk_size, self._dtype, self._fill_value)

    def chunk_index(self, pos: Vec3i) -> Vec3i:
        res = np.asarray(pos, dtype=np.int) // self._chunk_size
        assert res.shape == (3,)
        return res

    def chunk_at_pos(self, pos: Vec3i) -> Optional[Chunk[_VT]]:
        return self.chunks.get(self.chunk_index(pos))

    def ensure_chunk_at_index(self, index: ChunkIndex, *, insert=True) -> Chunk[_VT]:
        return self.chunks.create_if_absent(index, self._new_chunk_factory, insert=insert)

    def ensure_chunk_at_pos(self, pos: Vec3i, insert=True) -> Chunk[_VT]:
        return self.ensure_chunk_at_index(self.chunk_index(pos), insert=insert)

    def empty_mask(self, default=False) -> np.ndarray:
        return np.full(self.chunk_shape, default, dtype=np.bool)

    @classmethod
    def iter_neighbors_indices(cls, index: ChunkIndex) -> Iterator[Tuple[ChunkFace, Vec3i]]:
        yield from ((f, np.add(index, f.direction())) for f in ChunkFace)

    def iter_neighbors(self, index: ChunkIndex, flatten=False) -> Iterator[Tuple[ChunkFace, Optional["Chunk"]]]:
        if flatten:
            yield from ((f, c) for f, c in self.iter_neighbors(index, False) if c is not None)
        else:
            yield from ((f, self.chunks.get(i, None)) for f, i in self.iter_neighbors_indices(index))

    def __bool__(self):
        raise ValueError(
            f"The truth value of {__class__} is ambiguous. "
            "Use a.any(), or a.all(), or wrap the comparison (0 < a) & (a < 0)"
        )

    def all(self):
        """True if all chunks contain only True values"""
        return all(c.all() for c in self.chunks.values())

    def any(self):
        """True if any chunk contains any True value"""
        return any(c.any() for c in self.chunks.values())

    def to_dense(self, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None) -> npt.NDArray[_VT]:
        return self.to_dense_with_offset[0]

    def to_dense_with_offset(
        self, x: SliceOpt = None, y: SliceOpt = None, z: SliceOpt = None
    ) -> Tuple[npt.NDArray[_VT], Vec3i]:
        """Convert the grid to a dense numpy array"""
        if len(self.chunks) == 0:
            return np.empty((0, 0, 0)), np.zeros(3)

        # Variable cache
        cs = self._chunk_size

        index_min, index_max = self.chunks.minmax(True)
        pos_min = index_min * cs
        pos_max = (index_max + 1) * cs
        voxel_it = VoxelGridIterator(pos_min, pos_max, x, y, z, clip=False)

        chunk_it = voxel_it // cs
        chunk_min = np.asarray(chunk_it.start)
        chunk_max = np.asarray(chunk_it.stop)
        chunk_len = chunk_max - chunk_min

        res = np.full(tuple(chunk_len * cs), self.fill_value, dtype=self.dtype)

        # Method cache (prevent lookup in loop)
        __self_chunks_get = self.chunks.get
        __chunk_to_array = Chunk.to_array

        for index in chunk_it:
            c = __self_chunks_get(index, None)
            if c is not None:
                u, v, w = (index - chunk_min) * cs
                res[u : u + cs, v : v + cs, w : w + cs] = __chunk_to_array(c)

        start = voxel_it.start - chunk_min * cs
        stop = voxel_it.stop - chunk_min * cs
        step = voxel_it.step

        return (
            res[start[0] : stop[0] : step[0], start[1] : stop[1] : step[1], start[2] : stop[2] : step[2]],
            chunk_min * cs,
        )

    def items(self, mask: "ChunkGrid[bool]" = None) -> Iterator[Tuple[Vec3i, _VT]]:
        # Method cache (prevent lookup in loop)
        __chunk_items = Chunk.items
        if mask is None:
            for i, c in self.chunks.items():
                yield from __chunk_items(c)
        else:
            # Method cache (prevent lookup in loop)
            __chunk_any_fast = Chunk.any_if_filled
            __mask_ensure_chunk_at_index = mask.ensure_chunk_at_index
            for i, c in self.chunks.items():
                m = __mask_ensure_chunk_at_index(i, insert=False)
                if __chunk_any_fast(m):
                    yield from __chunk_items(c, mask=m)

    def _argwhere_iter_arrays(self, mask: Optional["ChunkGrid[bool]"] = None) -> Iterator[npt.NDArray[np.int_]]:
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

    def argwhere(self, mask: Optional["ChunkGrid[bool]"] = None) -> Iterator[Vec3i]:
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

    def filter(self, other: "ChunkGrid[bool]", fill_value: Optional[_VT] = None) -> "ChunkGrid[_VT]":
        """Apply a filter mask to this grid and return the masked values"""
        result = self.copy(empty=True, fill_value=fill_value)

        # Method cache (prevent lookup in loop)
        __self_chunks_get = self.chunks.get
        __chunk_any = Chunk.any
        __result_chunks_insert = result.chunks.insert

        for i, o in other.chunks.items():
            c = __self_chunks_get(i, None)
            if c is not None and __chunk_any(c):
                __result_chunks_insert(i, c.filter(o, fill_value=fill_value))

    def __getitem__(self, item):
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

    def get_values(self, pos: Iterable[Vec3i] | npt.NDArray[np.int_]) -> npt.NDArray[_VT]:
        """Returns a list of values at the positions"""
        # Method cache (prevent lookup in loop)
        __np_argwhere = np.argwhere
        __self_ensure_chunk_at_index = self.ensure_chunk_at_index
        __chunk_to_array = Chunk.to_array

        pos = np.asarray(pos, dtype=int)
        assert pos.ndim == 2 and pos.shape[1] == 3
        csize = self._chunk_size
        cind, cinv = np.unique(pos // csize, axis=0, return_inverse=True)
        result = np.zeros(len(cinv), dtype=self._dtype)
        for n, i in enumerate(cind):
            pind = __np_argwhere(cinv == n).flatten()
            cpos = pos[pind] % csize
            chunk = __self_ensure_chunk_at_index(i, insert=False)
            result[pind] = __chunk_to_array(chunk)[tuple(cpos.T)]
        return result

    def get_value(self, pos: Vec3i) -> _VT:
        index = self.chunk_index(pos)
        c: Chunk[_VT] = self.chunks.get(index, None)
        if c is None:
            return self._fill_value
        else:
            return c.get_pos(pos)

    def set_value(self, pos: Vec3i, value: _VT) -> Chunk[_VT]:
        c = self.ensure_chunk_at_pos(pos)
        c.set_pos(pos, value)
        return c

    def set_or_fill(self, pos: Vec3i, value: _VT) -> Chunk[_VT]:
        c = self.ensure_chunk_at_pos(pos)
        c.set_or_fill(pos, value)
        return c

    def _set_slices(self, value: _VT | np.ndarray, x: int | slice = None, y: int | slice = None, z: int | slice = None):
        # Variable cache
        cs = self._chunk_size

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

    def _set_positions(self, pos: np.ndarray, value: _VT | Sequence[_VT]) -> None:
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
                upos = np.unique(pos, axis=0)
                for p in upos:
                    self.set_value(p, value)

    def _set_chunks(self, mask: "ChunkGrid", value: _VT | np.ndarray | Chunk[_VT] | "ChunkGrid[_VT]") -> None:
        assert self._chunk_size == mask._chunk_size
        # Method cache (prevent lookup in loop)
        __grid_ensure_chunk_at_index = ChunkGrid.ensure_chunk_at_index
        if isinstance(value, ChunkGrid):

            indices = set(mask.chunks.keys())
            if mask._fill_value:
                indices.update(self.chunks.keys())
                indices.update(value.chunks.keys())

            for index in indices:
                m = mask.ensure_chunk_at_index(index, insert=False)
                val = __grid_ensure_chunk_at_index(value, index, insert=False)
                __grid_ensure_chunk_at_index(self, index)[m] = val

            # Set fill value
            if mask._fill_value:
                self._fill_value = self._dtype.base.type(value._fill_value)
        else:
            indices = set(mask.chunks.keys())
            if mask._fill_value:
                indices.update(self.chunks.keys())

            for index in indices:
                m = mask.ensure_chunk_at_index(index, insert=False)
                __grid_ensure_chunk_at_index(self, index)[m] = value

            # Set fill value
            if mask._fill_value:
                if isinstance(value, Chunk):
                    self._fill_value = value._fill_value
                else:
                    if isinstance(value, np.ndarray):
                        assert value.dtype == self._dtype
                        self._fill_value = value
                    else:
                        self._fill_value = self._dtype.base.type(value)

    def __setitem__(self, key, value):
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

    def cleanup(self, remove=False):
        for chunk in self.chunks:
            chunk.cleanup()
        if remove:
            for chunk in list(self.chunks):
                if chunk.is_filled_with(self._fill_value):
                    del self.chunks[chunk.index]
        return self

    def pad_chunks(self, width: int = 1):
        visited: Set[ChunkIndex] = set()
        for s in range(0, width):
            extra: Set[ChunkIndex] = set(
                tuple(n) for i in self.chunks.keys() for f, n in self.iter_neighbors_indices(i)
            )
            extra = extra.difference(visited)
            for e in extra:
                self.ensure_chunk_at_index(e)
            visited.update(extra)
        return self

    def iter_hull(self) -> Iterator[Chunk[_VT]]:
        """Iter some of the outer chunks that represent the hull around all chunks"""
        if self.chunks:
            it = self.chunks.sliced_iterator()
            for x in it.x.range():
                for y in it.y.range():
                    for z in it.z.range():
                        c = self.chunks.get((x, y, z), None)
                        if c is not None:
                            yield c
                            break
            for x in reversed(it.x.range()):
                for y in reversed(it.y.range()):
                    for z in reversed(it.z.range()):
                        c = self.chunks.get((x, y, z), None)
                        if c is not None:
                            yield c
                            break

    def get_neigbors_at(
        self, index: ChunkIndex, neighbors=True, edges=True, corners=True, insert=False, ensure=True
    ):  # Method cache
        __face_direction = ChunkFace.direction

        if ensure:

            def getter(index):
                return self.ensure_chunk_at_index(index, insert=insert)

        else:

            def getter(index):
                return self.chunks.get(index, default=None)

        chunks = np.full((3, 3, 3), None, dtype=np.ndarray)
        chunks[1, 1, 1] = self
        if neighbors:
            for face, index in self.iter_neighbors_indices(index):
                u, v, w = __face_direction(face) + 1
                chunks[u, v, w] = getter(index)
        if edges:
            # Add edges
            for a, b in ChunkFace.edges():
                d = __face_direction(a) + __face_direction(b)
                u, v, w = d + 1
                chunks[u, v, w] = getter(index + d)
        if corners:
            # Add corners
            for a, b, c in ChunkFace.corners():
                d = __face_direction(a) + __face_direction(b) + __face_direction(c)
                u, v, w = d + 1
                chunks[u, v, w] = getter(index + d)
        return chunks

    def get_block_at(
        self,
        index: ChunkIndex,
        shape: Tuple[int, int, int],
        *,
        offset: Optional[Vec3i] = None,
        edges=True,
        corners=True,
        ensure=True,
        insert=False,
    ):
        assert len(shape) == 3

        # Method cache
        __face_direction = ChunkFace.direction

        if ensure:

            def getter(index):
                return self.ensure_chunk_at_index(index, insert=insert)

        else:

            def getter(index):
                return self.chunks.get(index, default=None)

        _shape = np.asarray(shape)
        chunks = np.full(shape, None, dtype=np.object)
        offset = _shape // 2 if offset is None else np.asarray(offset, dtype=np.int)
        assert offset.shape == (3,)

        # Corner/Edge case handling
        low = np.zeros(3, dtype=int)
        high = low + _shape - 1

        ignore_chunks = (not edges) or (not corners)

        for i in np.ndindex(shape):
            chunk_pos = np.add(index, i) - offset
            if ignore_chunks:
                s = np.sum(i == low) + np.sum(i == high)
                if not edges and s == 2:
                    continue
                if not corners and s == 3:
                    continue
            chunks[i] = getter(chunk_pos)

        return chunks

    # Operators

    def apply(
        self,
        func: Callable[[Chunk[_VT] | _VT], Chunk[_OT] | _OT],
        dtype: Type[_OT] = None,
        inplace=False,
    ) -> "ChunkGrid[_OT]":
        dtype = self.__dtype(dtype)
        # Inplace selection
        new_grid = self if inplace else self.copy(empty=True, dtype=dtype)
        new_grid._fill_value = func(self._fill_value)

        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert

        for i, a in self.chunks.items():
            new_chunk = func(a)
            assert isinstance(new_chunk, Chunk)
            __new_grid_chunks_insert(i, new_chunk)

        if dtype is not None:
            new_grid._dtype = dtype
            new_grid._fill_value = dtype.type(new_grid._fill_value)
        return new_grid

    def outer_join(
        self,
        rhs,
        func: Callable[[Chunk[_VT] | _VT, Chunk[_VT] | _VT], Chunk[_OT] | _OT],
        dtype: Type[_OT] = None,
        inplace=False,
    ) -> "ChunkGrid[_OT]":
        dtype = self.__dtype(dtype)

        rhs_is_grid = isinstance(rhs, ChunkGrid)

        fill_value = self._fill_value
        rhs_fill_value = rhs._fill_value if rhs_is_grid else rhs
        try:
            fill_value = dtype.base.type(func(fill_value, rhs_fill_value))
        except Exception:
            pass

        # Inplace selection
        if inplace:
            assert dtype == self._dtype  # Inplace must retain type
            new_grid = self
            new_grid._fill_value = fill_value
        else:
            new_grid = self.copy(empty=True, dtype=dtype, fill_value=fill_value)

        # Method cache (prevent lookup in loop)
        __new_grid_chunks_insert = new_grid.chunks.insert

        if rhs_is_grid:
            assert new_grid._chunk_size == rhs._chunk_size
            indices = set(self.chunks.keys())
            indices.update(rhs.chunks.keys())

            # Method cache (prevent lookup in loop)
            _ensure_chunk_at_index = ChunkGrid.ensure_chunk_at_index

            for i in indices:
                a = _ensure_chunk_at_index(self, i, insert=False)
                b = _ensure_chunk_at_index(rhs, i, insert=False)
                new_chunk = func(a, b)
                assert isinstance(new_chunk, Chunk)
                __new_grid_chunks_insert(i, new_chunk)
        else:
            for i, a in self.chunks.items():
                new_chunk = func(a, rhs)
                assert isinstance(new_chunk, Chunk)
                __new_grid_chunks_insert(i, new_chunk)

        return new_grid

    # Comparison Operator

    def __eq__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.eq, dtype=np.bool8)

    def __ne__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.ne, dtype=np.bool8)

    def __lt__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.lt, dtype=np.bool8)

    def __le__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.le, dtype=np.bool8)

    def __gt__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.gt, dtype=np.bool8)

    def __ge__(self, rhs) -> "ChunkGrid[np.bool8]":
        return self.outer_join(rhs, func=operator.ge, dtype=np.bool8)

    eq = __eq__
    ne = __ne__
    lt = __lt__
    le = __le__
    gt = __gt__
    ge = __ge__

    equals = __eq__

    # Single Operator

    def __abs__(self) -> "ChunkGrid[_VT]":
        return self.apply(operator.abs, dtype=self._dtype)

    def __invert__(self) -> "ChunkGrid[_VT]":
        return self.apply(operator.inv, dtype=self._dtype)

    def __neg__(self) -> "ChunkGrid[_VT]":
        return self.apply(operator.neg, dtype=self._dtype)

    abs = __abs__
    invert = __invert__
    neg = __neg__

    # Logic Operator

    def __and__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.and_, dtype=self._dtype)

    def __or__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.or_, dtype=self._dtype)

    def __xor__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.xor, dtype=self._dtype)

    def __iand__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.iand, dtype=self._dtype, inplace=True)

    def __ior__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.ior, dtype=self._dtype, inplace=True)

    def __ixor__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.ixor, dtype=self._dtype, inplace=True)

    and_ = __and__
    or_ = __or__
    xor = __xor__

    # Math Operator

    def __add__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.add)

    def __sub__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.sub)

    def __mul__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.mul)

    def __matmul__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.matmul)

    def __mod__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.mod)

    def __pow__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.pow)

    def __floordiv__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.floordiv)

    def __iadd__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.iadd, inplace=True)

    def __isub__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.isub, inplace=True)

    def __imul__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.imul, inplace=True)

    def __imatmul__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.imatmul, inplace=True)

    def __imod__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.imod, inplace=True)

    def __ifloordiv__(self, rhs) -> "ChunkGrid[_VT]":
        return self.outer_join(rhs, func=operator.ifloordiv, inplace=True)

    # TrueDiv Operator

    def __truediv__(self, rhs: "ChunkGrid[_VT]" | np.ndarray | _VT) -> "ChunkGrid[float]":
        return self.outer_join(rhs, func=operator.truediv, dtype=np.float)

    def __itruediv__(self, rhs: "ChunkGrid[_VT]" | np.ndarray | _VT) -> "ChunkGrid[float]":
        return self.outer_join(rhs, func=operator.itruediv, dtype=np.float, inplace=True)

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

    def sum(self, dtype: Type[_OT] = None) -> _OT:
        return sum(c.sum() for c in self.chunks)

    @classmethod
    def stack(cls, grids: Sequence["ChunkGrid[_VT]"], fill_value=None) -> "ChunkGrid[np.ndarray]":
        assert len(grids) > 0
        if len(grids) == 1:
            return grids[0]
        # Check that grid size matches!
        chunk_size = grids[0].chunk_size
        assert all(chunk_size == g.chunk_size for g in grids)
        indices = set(k for g in grids for k in g.chunks.keys())
        dtypes = [g.dtype for g in grids]
        if all(t == dtypes[0] for t in dtypes):
            assert dtypes[0].base is not np.void
            dtype = np.dtype((dtypes[0].base, (len(grids),)))
            fill_value = fill_value if fill_value is not None else np.zeros(1, dtype)
        else:
            raise ValueError("Mixed data type grids not supported!")

        new_grid = ChunkGrid(chunk_size, dtype=dtype, fill_value=fill_value)
        for ind in indices:
            new_grid.chunks.insert(
                ind,
                Chunk._stack(
                    [g.ensure_chunk_at_index(ind, insert=False) for g in grids], dtype=dtype, fill_value=fill_value
                ),
            )
        return new_grid

    def set_block_at(
        self,
        index: ChunkIndex,
        data: npt.NDArray[_VT],
        op: Callable[[Chunk[_VT], npt.NDArray[_VT]], Any] = None,
        replace: bool = True,
    ):
        index_arr = np.asarray(index, dtype=int)
        size = self.chunk_size
        data_shape = np.array(data.shape, dtype=int)[:3]
        assert np.all(data_shape % size == 0)
        block_shape = tuple(data_shape // size)
        block_offset = np.array(block_shape, dtype=int) // 2

        op = op or Chunk.set_array

        if replace:
            # dst = self.get_block_at(index, block_shape, ensure=True, insert=True)
            for u in range(block_shape[0]):
                for v in range(block_shape[1]):
                    for w in range(block_shape[2]):
                        s0 = np.array((u, v, w)) * size
                        s1 = s0 + size
                        chunk_index = index_arr - block_offset + (u, v, w)
                        chunk = self.ensure_chunk_at_index(chunk_index)
                        op(chunk, data[s0[0] : s1[0], s0[1] : s1[1], s0[2] : s1[2]])

        else:
            for u in range(block_shape[0]):
                for v in range(block_shape[1]):
                    for w in range(block_shape[2]):
                        chunk_index = index_arr - block_offset + (u, v, w)
                        chunk = self.chunks.get(chunk_index, None)
                        if chunk is None:
                            s0 = np.array((u, v, w)) * size
                            s1 = s0 + size
                            chunk = self.ensure_chunk_at_index(chunk_index)
                            op(chunk, data[s0[0] : s1[0], s0[1] : s1[1], s0[2] : s1[2]])

    def padding_at(
        self, index: ChunkIndex, padding: int = 1, fill_value: _VT | None = None, corners: bool = True, edges: bool = True
    ):
        assert padding >= 0
        if padding == 0:
            return self.ensure_chunk_at_index(index, insert=False).to_array()

        chunk_size = self._chunk_size
        pad_chunks = int(np.ceil(padding / chunk_size))
        pad_size = 1 + pad_chunks * 2
        pad_outer = (chunk_size - padding) % chunk_size
        chunks = self.get_block_at(index, (pad_size, pad_size, pad_size), corners=corners, edges=edges)
        block = self.block_to_array(chunks, fill_value=fill_value)
        if pad_outer:
            return block[pad_outer:-pad_outer, pad_outer:-pad_outer, pad_outer:-pad_outer]
        return block

    def block_to_array(self, chunks, fill_value: _VT | None = None) -> npt.NDArray[_VT]:
        chunks = np.atleast_3d(np.asarray(chunks, dtype=np.object_))
        cs = self._chunk_size
        fill_value = self._fill_value if fill_value is None else fill_value
        dtype = self._dtype
        chunk_shape = (cs, cs, cs)

        def _to_array(chunk: Chunk[_VT] | None) -> npt.NDArray[_VT]:
            if chunk is None:
                return np.full(chunk_shape, fill_value, dtype=self._dtype)
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

    def unique(self) -> npt.NDArray[_VT]:
        if not self.chunks:
            return np.empty((0,), dtype=self._dtype)
        return np.unique(np.concatenate([c.unique() for c in self.chunks]))
