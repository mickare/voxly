

from typing import Any, Protocol, Tuple, TypeVar, runtime_checkable
import numpy as np
import numpy.typing as npt

Vec3i = npt.NDArray[np.int_]

Index3 = Tuple[int, int, int]


class BoolType(Protocol):
    def __bool__(self) -> bool: ...

_T_co = TypeVar('_T_co', contravariant=True)

@runtime_checkable
class LenType(Protocol):
    def __len__(self) -> int: ...

class SupportsDunderLT(Protocol[_T_co]):
    def __lt__(self, __rhs: _T_co) -> Any: ...

class SupportsDunderGT(Protocol[_T_co]):
    def __gt__(self, __rhs: _T_co) -> Any: ...
    
SupportsRichComparison = SupportsDunderLT | SupportsDunderGT

@runtime_checkable
class SupportsDunderMul(Protocol[_T_co]):
    def __mul__(self, __rhs: _T_co) -> Any: ...

