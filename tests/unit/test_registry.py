"""Discovery, de-duplication and ``--backend`` spec resolution."""

from __future__ import annotations

import pytest

from torch_mbt.backends import Backend, Mechanism
from torch_mbt.backends.registry import (
    ENTRY_POINT_GROUP,
    discover,
    find_backend,
    plugin_backends,
    resolve_spec,
)
from torch_mbt.errors import BackendNotFound


class StubBackend(Backend):
    """A discoverable adapter that is always available."""

    name = "stub"
    mechanism = Mechanism.CUSTOM
    priority = 55

    @property
    def device_type(self) -> str:
        return "stub"

    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2


def _names(backends) -> list[str]:
    return [backend.name for backend in backends]


def test_discovery_always_contains_cpu_once_and_puts_it_last() -> None:
    names = _names(discover(with_entry_points=False))
    assert names.count("cpu") == 1
    assert names[-1] == "cpu"
    assert "privateuse1" in names


def test_discovery_is_stable_across_calls() -> None:
    first = _names(discover(with_entry_points=False))
    second = _names(discover(with_entry_points=False))
    assert first == second


def test_extra_adapters_are_discovered() -> None:
    names = _names(discover(with_entry_points=False, extra=[StubBackend()]))
    assert "stub" in names
    assert names.index("stub") < names.index("cpu")


def test_adapters_for_the_same_device_type_are_deduplicated() -> None:
    class Duplicate(StubBackend):
        name = "duplicate"
        priority = 10

    class Native(StubBackend):
        name = "stub"

    discovered = discover(with_entry_points=False, extra=[StubBackend(), Duplicate(), Native()])
    assert _names(discovered).count("stub") == 1
    assert "duplicate" not in _names(discovered)


def test_entry_point_group_name_is_public_api() -> None:
    assert ENTRY_POINT_GROUP == "torch_mbt.backends"


def test_plugin_discovery_returns_a_list_even_when_nothing_is_installed() -> None:
    assert isinstance(plugin_backends(), list)


def test_auto_selects_only_available_backends() -> None:
    backends = resolve_spec("auto")
    assert backends
    assert all(backend.is_available() for backend in backends)
    assert "cpu" in _names(backends)


def test_default_spec_is_auto() -> None:
    assert _names(resolve_spec(None)) == _names(resolve_spec("auto"))


def test_all_keeps_unavailable_backends_so_they_are_reported_as_skips() -> None:
    backends = resolve_spec("all")
    names = _names(backends)
    assert "cpu" in names
    assert any(not backend.is_available() for backend in backends)


def test_explicit_list_preserves_the_requested_order() -> None:
    names = _names(resolve_spec("cuda,cpu"))
    assert names == ["cuda", "cpu"]


def test_repeated_names_are_collapsed() -> None:
    assert _names(resolve_spec("cpu,cpu")) == ["cpu"]


def test_aliases_resolve_to_the_generic_backend() -> None:
    names = _names(resolve_spec("privateuse1"))
    assert names == ["privateuse1"]
    assert _names(resolve_spec("privateuseone")) == ["privateuse1"]


def test_globs_expand_to_every_match_in_discovery_order() -> None:
    assert _names(resolve_spec("*pu")) == ["xpu", "cpu"]
    assert _names(resolve_spec("cp*")) == ["cpu"]


def test_unknown_backend_lists_what_is_known() -> None:
    with pytest.raises(BackendNotFound, match="known backends"):
        resolve_spec("definitely-not-a-backend")


def test_auto_all_and_empty_are_rejected_when_combined() -> None:
    for spec in ("auto,cpu", "all,cpu", ""):
        with pytest.raises(BackendNotFound):
            resolve_spec(spec)


def test_find_backend_returns_the_adapter_or_raises() -> None:
    discovered = discover(with_entry_points=False)
    assert find_backend(discovered, "cpu").name == "cpu"
    with pytest.raises(BackendNotFound):
        find_backend(discovered, "nope")
