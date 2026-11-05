"""Backend adapters and discovery.

>>> from neutraltest.backends import discover, resolve_spec
>>> [b.name for b in resolve_spec("auto")]  # doctest: +SKIP
['cpu']
"""

from __future__ import annotations

from .base import Backend, Mechanism, as_backend, iter_devices
from .builtin import (
    CpuBackend,
    CudaBackend,
    MpsBackend,
    PrivateUse1Backend,
    XpuBackend,
    builtin_backends,
)
from .registry import ENTRY_POINT_GROUP, discover, find_backend, plugin_backends, resolve_spec

__all__ = [
    "Backend",
    "Mechanism",
    "CpuBackend",
    "CudaBackend",
    "MpsBackend",
    "PrivateUse1Backend",
    "XpuBackend",
    "ENTRY_POINT_GROUP",
    "as_backend",
    "builtin_backends",
    "discover",
    "find_backend",
    "iter_devices",
    "plugin_backends",
    "resolve_spec",
]
