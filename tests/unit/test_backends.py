"""Backend abstraction, device construction and adapter validation."""

from __future__ import annotations

import pytest
import torch

from torch_mbt.backends import (
    Backend,
    CpuBackend,
    CudaBackend,
    MpsBackend,
    PrivateUse1Backend,
    XpuBackend,
    as_backend,
    builtin_backends,
    iter_devices,
)
from torch_mbt.errors import BackendUnavailable


@pytest.fixture
def cpu() -> CpuBackend:
    return CpuBackend()


def test_cpu_backend_contract(cpu: CpuBackend) -> None:
    assert cpu.name == "cpu"
    assert cpu.device_type == "cpu"
    assert cpu.is_available()
    assert cpu.device_count() == 1
    assert cpu.requires_index is False
    assert cpu.make_device() == torch.device("cpu")
    assert cpu.make_device(3) == torch.device("cpu")


def test_cpu_backend_has_no_streams_and_a_noop_synchronize(cpu: CpuBackend) -> None:
    assert cpu.has_streams() is False
    with pytest.raises(NotImplementedError):
        cpu.new_stream()
    cpu.synchronize()  # must not raise
    assert cpu.memory_allocated() is None or cpu.memory_allocated() >= 0
    cpu.empty_cache()


def test_device_index_is_validated_against_device_count(cpu: CpuBackend) -> None:
    cuda = CudaBackend()
    if cuda.is_available():
        pytest.skip("this machine has CUDA; the out-of-range path needs an empty backend")
    with pytest.raises(IndexError, match="out of range"):
        cuda.make_device(0)


def test_shuffle_device_prefers_the_explicit_device(cpu: CpuBackend) -> None:
    explicit = torch.device("cpu")
    assert cpu.shuffle_device(explicit) == explicit
    assert cpu.shuffle_device("cpu") == explicit
    assert cpu.shuffle_device(None, index=0) == torch.device("cpu")


def test_matching_accepts_names_device_types_and_globs(cpu: CpuBackend) -> None:
    assert cpu.matches("cpu")
    assert cpu.matches(" CPU ")
    assert cpu.matches("cpu*")
    assert not cpu.matches("cuda")
    assert not cpu.matches("c?")


def test_supports_dtype_probes_are_cached(cpu: CpuBackend) -> None:
    assert cpu.supports_dtype(torch.float32) is True
    assert cpu.supports_dtype(torch.float32) is True  # cached second call


def test_properties_reports_identity(cpu: CpuBackend) -> None:
    info = cpu.properties()
    assert info["name"] == "cpu"
    assert info["device_type"] == "cpu"
    assert info["available"] is True
    assert info["device_count"] == 1


def test_privateuse1_tracks_the_registered_name() -> None:
    backend = PrivateUse1Backend()
    registered = torch._C._get_privateuse1_backend_name()
    assert backend.device_type == registered
    if registered in ("privateuseone", "privateuse1"):
        assert backend.is_available() is False
        assert backend.device_count() == 0
        # The generic handles still select it, which is what --backend=all needs.
        assert backend.matches("privateuse1")
        assert backend.matches("privateuseone")
    else:  # pragma: no cover - only on a machine with an out-of-tree backend
        assert backend.matches(registered)


def test_optional_backends_report_availability_without_raising() -> None:
    for backend in (CudaBackend(), XpuBackend(), MpsBackend()):
        assert isinstance(backend.is_available(), bool)
        if not backend.is_available():
            assert backend.device_count() == 0


def test_builtin_backends_cover_every_mechanism() -> None:
    names = [backend.name for backend in builtin_backends()]
    assert names == ["privateuse1", "xpu", "cuda", "mps", "cpu"]
    assert all(isinstance(backend, Backend) for backend in builtin_backends())


def test_as_backend_accepts_instances_and_classes() -> None:
    assert isinstance(as_backend(CpuBackend()), CpuBackend)
    assert isinstance(as_backend(CpuBackend), CpuBackend)
    with pytest.raises(TypeError, match="expected a Backend"):
        as_backend(object())


def test_iter_devices_yields_one_device_per_ordinal() -> None:
    assert list(iter_devices(CpuBackend())) == [torch.device("cpu")]
    assert list(iter_devices(CudaBackend())) == []


def test_adapter_names_must_be_lower_case() -> None:
    with pytest.raises(TypeError, match="lower-case"):

        class BadName(Backend):
            name = "MyAccel"
            mechanism = CpuBackend.mechanism

            @property
            def device_type(self) -> str:
                return "myaccel"

            def is_available(self) -> bool:
                return False


def test_intermediate_base_classes_may_omit_the_name() -> None:
    class Abstractish(Backend):
        @property
        def device_type(self) -> str:
            return "abstract"

        def is_available(self) -> bool:
            return False

    assert Abstractish.name == ""
    assert isinstance(Abstractish(), Backend)


def test_custom_adapter_works_through_the_public_api() -> None:
    class FakeAccelerator(Backend):
        name = "fake"
        priority = 70

        @property
        def device_type(self) -> str:
            return "meta"

        def is_available(self) -> bool:
            return True

        def device_count(self) -> int:
            return 2

    backend = FakeAccelerator()
    assert backend.make_device(1) == torch.device("meta:1")
    assert backend.device_count() == 2
    assert backend.matches("fake")
    assert backend.matches("meta")
    with pytest.raises(IndexError, match="out of range"):
        backend.make_device(2)
    assert [str(device) for device in iter_devices(backend)] == ["meta:0", "meta:1"]


def test_an_unregistered_device_type_is_reported_clearly() -> None:
    class NotRegistered(Backend):
        name = "not-registered"

        @property
        def device_type(self) -> str:
            return "definitely-not-a-device-type"

        def is_available(self) -> bool:
            return True

    with pytest.raises(BackendUnavailable, match="rename_privateuse1_backend"):
        NotRegistered().make_device(0)
