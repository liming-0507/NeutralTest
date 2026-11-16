"""The backend abstraction.

A *backend* here is not "a torch device type we like".  It is "a target that can
execute aten operators", described by how it is registered with PyTorch rather
than by whose silicon it is.  That distinction is the whole point of the
package: vendor PyTorch distributions reach us through one of a small number of
registration mechanisms, and each mechanism needs slightly different probing
code, while the *tests* stay identical.

Mechanisms covered out of the box:

``cpu``
    Always present.

``cuda``
    Anything exposing the CUDA runtime through ``torch.cuda``.  Distributions
    that ship a CUDA-compatible runtime land here even if the silicon is not
    NVIDIA; the harness never needs to know which.

``xpu``
    Anything exposing ``torch.xpu`` (``torch.xpu.is_available()``).

``mps``
    Apple Metal.

``privateuse1``
    The generic out-of-tree extension point: a vendor renames
    ``PrivateUse1`` (``torch.utils.backend_registration.rename_privateuse1_backend``)
    and registers kernels for it.  Most non-CUDA accelerators use this.

A vendor that fits none of those can subclass :class:`Backend` and publish the
subclass as an entry point; see ``docs/backends.md``.
"""

from __future__ import annotations

import contextlib
import fnmatch
from abc import ABC, abstractmethod
from collections.abc import Iterator
from enum import Enum
from functools import cache
from typing import Any, ClassVar

import torch

from ..errors import BackendUnavailable

__all__ = ["Backend", "Mechanism", "DeviceLike"]


class Mechanism(str, Enum):
    """How a backend is registered with PyTorch."""

    CPU = "cpu"
    CUDA = "cuda"
    XPU = "xpu"
    MPS = "mps"
    PRIVATEUSE1 = "privateuse1"
    CUSTOM = "custom"

    def __str__(self) -> str:  # nicer pytest ids
        return self.value


DeviceLike = torch.device | str


class Backend(ABC):
    """A target that can execute aten operators.

    Subclasses set :attr:`name`, :attr:`mechanism` and :attr:`priority` at class
    level and implement :attr:`device_type` and :attr:`is_available`.  Every
    other method has a conservative default so a minimal adapter is ~15 lines.
    """

    #: Stable handle used by ``--backend``.  Must be unique among discovered
    #: backends and should not be a vendor brand unless the adapter is
    #: vendor-specific (the built-ins deliberately are not).
    name: ClassVar[str] = ""
    #: Registration mechanism, see :class:`Mechanism`.
    mechanism: ClassVar[Mechanism] = Mechanism.CUSTOM
    #: Auto-detection ordering, higher first.  CPU is always last regardless.
    priority: ClassVar[int] = 100
    #: Whether the device string needs an ordinal (``cuda:0``) or not (``cpu``).
    requires_index: ClassVar[bool] = True

    # ------------------------------------------------------------------ identity

    @property
    @abstractmethod
    def device_type(self) -> str:
        """The string ``torch.device`` expects, e.g. ``"cuda"`` or ``"xpu"``."""

    @abstractmethod
    def is_available(self) -> bool:
        """True when this backend can actually run now (driver, device, init)."""

    def aliases(self) -> tuple[str, ...]:
        """Extra names accepted by ``--backend`` (device type, legacy names)."""
        return (self.device_type,)

    def __repr__(self) -> str:
        return f"<Backend {self.name!r} device_type={self.device_type!r}>"

    def __str__(self) -> str:
        return self.name

    # ------------------------------------------------------------------ devices

    def device_count(self) -> int:
        """Number of usable devices; 0 when unavailable.

        Availability wins over whatever the module reports: some vendor modules
        answer ``device_count()`` even when their runtime cannot be used, and a
        count above zero would make the harness try to place tensors there.
        """
        if not self.is_available():
            return 0
        module = self._module()
        getter = getattr(module, "device_count", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                return int(getter())
        return 1

    def make_device(self, index: int = 0) -> torch.device:
        """Build the ``torch.device`` for ordinal *index*."""
        if not self.requires_index:
            return torch.device(self.device_type)
        count = self.device_count()
        if index >= count:
            raise IndexError(f"{self.name}: device index {index} out of range (count={count})")
        try:
            return torch.device(f"{self.device_type}:{index}")
        except RuntimeError as exc:
            raise BackendUnavailable(
                f"{self.name}: {self.device_type!r} is not a device type this torch build "
                f"knows; register it with rename_privateuse1_backend() first ({exc})"
            ) from exc

    def shuffle_device(self, device: DeviceLike | None, index: int = 0) -> torch.device:
        """Normalise an explicit device or fall back to the configured ordinal."""
        if device is None:
            return self.make_device(index)
        return device if isinstance(device, torch.device) else torch.device(device)

    def current_device_index(self) -> int:
        module = self._module()
        getter = getattr(module, "current_device", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                return int(getter())
        return 0

    def set_device_index(self, index: int) -> None:
        module = self._module()
        setter = getattr(module, "set_device", None)
        if callable(setter):
            setter(index)

    # ------------------------------------------------------------------ execution

    def synchronize(self, index: int | None = None) -> None:
        """Block until queued work on the current device completes."""
        if self.mechanism is Mechanism.CPU:
            return
        module = self._module()
        sync = getattr(module, "synchronize", None)
        if callable(sync):
            sync()
            return
        accelerator = getattr(torch, "accelerator", None)
        sync = getattr(accelerator, "synchronize", None)
        if callable(sync):
            with contextlib.suppress(Exception):
                sync(self.make_device(index or 0))
                return
        raise NotImplementedError(f"{self.name}: no synchronize entry point found")

    def empty_cache(self) -> None:
        """Release cached device memory, if the backend has a cache."""
        module = self._module()
        fn = getattr(module, "empty_cache", None)
        if callable(fn):
            fn()

    def memory_allocated(self) -> int | None:
        """Bytes currently held by tensors, or ``None`` when unreported."""
        module = self._module()
        fn = getattr(module, "memory_allocated", None)
        if callable(fn):
            with contextlib.suppress(Exception):
                return int(fn())
        return None

    def memory_reserved(self) -> int | None:
        module = self._module()
        fn = getattr(module, "memory_reserved", None)
        if callable(fn):
            with contextlib.suppress(Exception):
                return int(fn())
        return None

    def manual_seed(self, seed: int) -> None:
        """Seed device-side RNG if the backend has one."""
        module = self._module()
        fn = getattr(module, "manual_seed", None)
        if callable(fn):
            with contextlib.suppress(Exception):
                fn(seed)

    # ------------------------------------------------------------------ streams

    def has_streams(self) -> bool:
        module = self._module()
        return self.mechanism is not Mechanism.CPU and getattr(module, "Stream", None) is not None

    def new_stream(self, index: int = 0) -> Any:
        if not self.has_streams():
            raise NotImplementedError(f"{self.name}: this backend does not expose streams")
        return self._module().Stream(device=self.make_device(index))

    def stream(self, stream: Any = None) -> contextlib.AbstractContextManager[Any]:
        """Context manager running the block on *stream* (default: current)."""
        if not self.has_streams():
            raise NotImplementedError(f"{self.name}: this backend does not expose streams")
        return self._module().stream(stream)

    def stream_context(self, stream: Any) -> contextlib.AbstractContextManager[Any]:
        """Context manager that makes *stream* current for the block.

        ``torch.<device_type>.stream()`` is the usual spelling, but a mechanism
        reached through ``privateuse1`` keeps its streams on the module the
        vendor registered under its own name, not on the placeholder -- so the
        default asks for ``torch.<device_type>`` and a mechanism whose device
        type is an alias resolves the real one here.  A backend that has no
        streams (or no such entry point) gets an inert context manager, which
        makes the caller's fallback ordering -- explicit events plus
        synchronisation -- the behaviour it actually gets.
        """
        module = self._module()
        factory = getattr(module, "stream", None) if module is not None else None
        if not callable(factory):
            return contextlib.nullcontext()
        return factory(stream)

    # ------------------------------------------------------------------ probing

    @cache  # noqa: B019 - process-wide cache is intended
    def supports_dtype(self, dtype: torch.dtype) -> bool:
        """Cheap probe: can this backend allocate a tensor of *dtype*?"""
        try:
            probe = torch.empty(1, dtype=dtype, device=self.make_device(0))
        except Exception:  # noqa: BLE001 - any failure means "no"
            return False
        del probe
        return True

    def properties(self, index: int = 0) -> dict[str, Any]:
        """Best-effort description, used by ``neutraltest info``."""
        info: dict[str, Any] = {
            "name": self.name,
            "mechanism": str(self.mechanism),
            "device_type": self.device_type,
            "available": self.is_available(),
        }
        if not info["available"]:
            info["device_count"] = 0
            return info
        with contextlib.suppress(Exception):
            info["device_count"] = self.device_count()
        accelerator = getattr(torch, "accelerator", None)
        device = self.make_device(index)
        for key, getter in (
            ("capability", getattr(accelerator, "get_device_capability", None)),
            ("memory_info", getattr(accelerator, "get_memory_info", None)),
        ):
            if getter is None:
                continue
            with contextlib.suppress(Exception):
                info[key] = getter(device)
        return info

    # ------------------------------------------------------------------ internals

    def _module(self) -> Any:
        """The ``torch.<device_type>`` module, or ``None`` for a bare rename."""
        return getattr(torch, self.device_type, None)

    def matches(self, name: str) -> bool:
        """Whether *name* selects this backend (``--backend`` lookup)."""
        token = name.strip().lower()
        if not token:
            return False
        candidates = {self.name.lower(), self.device_type.lower(), *self.aliases()}
        if "*" in token or "?" in token:
            return any(fnmatch.fnmatchcase(candidate, token) for candidate in candidates)
        return token in candidates

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # ``__abstractmethods__`` is populated by ABCMeta *after* this hook runs,
        # so "leaves ``name`` unset" is the marker for an intermediate base.
        if not cls.name:
            return
        if cls.name != cls.name.lower():
            raise TypeError(f"{cls.__name__}.name must be lower-case, got {cls.name!r}")


def as_backend(obj: Any) -> Backend:
    """Coerce a plugin entry point result into a :class:`Backend`."""
    if isinstance(obj, Backend):
        return obj
    if isinstance(obj, type) and issubclass(obj, Backend):
        return obj()
    raise TypeError(f"expected a Backend subclass or instance, got {obj!r}")


def iter_devices(backend: Backend) -> Iterator[torch.device]:
    """Yield every device this backend exposes (empty when unavailable)."""
    if not backend.is_available():
        return
    for index in range(max(1, backend.device_count())):
        yield backend.make_device(index)
