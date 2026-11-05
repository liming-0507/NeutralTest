"""Backends for the registration mechanisms PyTorch ships with."""

from __future__ import annotations

import contextlib

import torch

from .base import Backend, Mechanism

__all__ = [
    "CpuBackend",
    "CudaBackend",
    "XpuBackend",
    "MpsBackend",
    "PrivateUse1Backend",
    "builtin_backends",
]


class CpuBackend(Backend):
    """The reference backend.  Always available, always included by ``auto``."""

    name = "cpu"
    mechanism = Mechanism.CPU
    priority = 0
    requires_index = False

    @property
    def device_type(self) -> str:
        return "cpu"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 1


class CudaBackend(Backend):
    """Anything served by the CUDA runtime interface (``torch.cuda``).

    This includes vendor distributions whose runtime is CUDA-compatible: from
    the harness' point of view the device behaves like CUDA, and that is exactly
    the contract the tests need.  No assumption about the silicon is made.
    """

    name = "cuda"
    mechanism = Mechanism.CUDA
    priority = 40

    @property
    def device_type(self) -> str:
        return "cuda"

    def is_available(self) -> bool:
        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return False
        with contextlib.suppress(Exception):
            return bool(cuda.is_available())
        return False


class XpuBackend(Backend):
    """Anything exposing the ``torch.xpu`` interface."""

    name = "xpu"
    mechanism = Mechanism.XPU
    priority = 50

    @property
    def device_type(self) -> str:
        return "xpu"

    def is_available(self) -> bool:
        xpu = getattr(torch, "xpu", None)
        if xpu is None:
            return False
        with contextlib.suppress(Exception):
            return bool(xpu.is_available())
        return False


class MpsBackend(Backend):
    """Apple Metal, exposed as device type ``mps``."""

    name = "mps"
    mechanism = Mechanism.MPS
    priority = 30
    requires_index = False

    @property
    def device_type(self) -> str:
        return "mps"

    def is_available(self) -> bool:
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is None:
            return False
        with contextlib.suppress(Exception):
            return bool(mps.is_available())
        return False

    def device_count(self) -> int:
        return 1 if self.is_available() else 0


class PrivateUse1Backend(Backend):
    """The out-of-tree extension point, whatever the vendor renamed it to.

    ``torch.utils.backend_registration.rename_privateuse1_backend("foo")`` makes
    ``torch.device("foo:0")`` valid and ``torch._C._get_privateuse1_backend_name()``
    return ``"foo"``.  The name is resolved on every call so an adapter created
    before the rename still behaves correctly, and it is also accepted as an
    alias, so ``--backend=foo`` works on a machine where the rename happened
    without the harness knowing anything about vendor ``foo``.
    """

    name = "privateuse1"
    mechanism = Mechanism.PRIVATEUSE1
    priority = 60

    @property
    def device_type(self) -> str:
        with contextlib.suppress(Exception):
            return str(torch._C._get_privateuse1_backend_name())
        return "privateuseone"

    def aliases(self) -> tuple[str, ...]:
        return (self.device_type, "privateuseone", "privateuse1")

    def is_available(self) -> bool:
        device_type = self.device_type
        # Nothing is registered: the stock name is still in place.
        if device_type in ("privateuseone", "privateuse1"):
            return False
        module = getattr(torch, device_type, None)
        probe = getattr(module, "is_available", None)
        if callable(probe):
            with contextlib.suppress(Exception):
                return bool(probe())
            return False
        # A rename without a convenience module: trust the accelerator API.
        accelerator = getattr(torch, "accelerator", None)
        current = getattr(accelerator, "current_accelerator", None)
        if callable(current):
            with contextlib.suppress(Exception):
                device = current()
                return device is not None and str(device.type) == device_type
        return False

    def device_count(self) -> int:
        module = getattr(torch, self.device_type, None)
        getter = getattr(module, "device_count", None)
        if callable(getter):
            with contextlib.suppress(Exception):
                return int(getter())
        return 1 if self.is_available() else 0

    def __repr__(self) -> str:  # show the rename, which is the useful part
        return f"<Backend 'privateuse1' device_type={self.device_type!r}>"


def builtin_backends() -> list[Backend]:
    """All mechanism adapters, most specific first."""
    return [
        PrivateUse1Backend(),
        XpuBackend(),
        CudaBackend(),
        MpsBackend(),
        CpuBackend(),
    ]
