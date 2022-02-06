

from typing import Any, Protocol, Tuple, runtime_checkable
import numpy as np
import numpy.typing as npt

Vec3i = npt.NDArray[np.int_]

Index3 = Tuple[int, int, int]


class BoolType(Protocol):
    def __bool__(self) -> bool: ...


@runtime_checkable
class LenType(Protocol):
    def __len__(self) -> int: ...

class SupportsDunderLT(Protocol):
    def __lt__(self, __other: Any) -> Any: ...

class SupportsDunderGT(Protocol):
    def __gt__(self, __other: Any) -> Any: ...
    
SupportsRichComparison = SupportsDunderLT | SupportsDunderGT
