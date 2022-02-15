from abc import abstractmethod
from typing import Any, Protocol, Tuple, TypeVar, runtime_checkable
import numpy as np
import numpy.typing as npt

Arr3i = npt.NDArray[np.int_]
Index3 = Tuple[int, int, int]
Vec3i = Arr3i | Index3


class BoolType(Protocol):
    def __bool__(self) -> bool:
        ...


_T_contra = TypeVar("_T_contra", contravariant=True)


@runtime_checkable
class LenType(Protocol):
    @abstractmethod
    def __len__(self) -> int:
        ...


class SupportsDunderLT(Protocol):
    @abstractmethod
    def __lt__(self, __rhs: Any) -> bool:
        ...


class SupportsDunderGT(Protocol):
    @abstractmethod
    def __gt__(self, __rhs: Any) -> bool:
        ...


@runtime_checkable
class SupportsDunderMul(Protocol[_T_contra]):
    @abstractmethod
    def __mul__(self, __rhs: _T_contra) -> Any:
        ...


@runtime_checkable
class NumpyDTypeProperty(Protocol):
    @property
    @abstractmethod
    def dtype(self) -> np.dtype[Any]:
        ...
