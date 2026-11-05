"""The process-wide run policy."""

from __future__ import annotations

import pytest

from neutraltest.policy import (
    RunPolicy,
    get_policy,
    patched_policy,
    reset_policy,
    set_policy,
)


@pytest.fixture(autouse=True)
def _restore_policy():
    yield
    reset_policy()


def test_defaults_are_the_strict_ones() -> None:
    policy = reset_policy()
    assert policy.on_unsupported == "fail"
    assert policy.tolerance_multiplier == 1.0
    assert policy.strict_backends is False
    assert policy.device_index == 0
    assert policy.seed == 20240101


def test_get_policy_returns_the_live_singleton() -> None:
    assert get_policy() is get_policy()


def test_set_policy_mutates_in_place() -> None:
    policy = get_policy()
    returned = set_policy(on_unsupported="skip", tolerance_multiplier=4.0)
    assert returned is policy
    assert get_policy().on_unsupported == "skip"
    assert get_policy().tolerance_multiplier == 4.0


def test_set_policy_rejects_invalid_values_without_partial_mutation() -> None:
    set_policy(on_unsupported="skip")
    with pytest.raises(ValueError, match="on_unsupported"):
        set_policy(on_unsupported="ignore")
    assert get_policy().on_unsupported == "skip"


@pytest.mark.parametrize(
    "changes",
    [
        {"tolerance_multiplier": 0},
        {"tolerance_multiplier": -1.0},
        {"device_index": -1},
        {"on_unsupported": "maybe"},
    ],
)
def test_invalid_policies_are_rejected(changes: dict) -> None:
    with pytest.raises(ValueError):
        RunPolicy(**changes)


def test_patched_policy_restores_the_previous_object() -> None:
    before = get_policy()
    with patched_policy(on_unsupported="xfail", seed=7) as patched:
        assert patched is get_policy()
        assert patched is not before
        assert get_policy().on_unsupported == "xfail"
        assert get_policy().seed == 7
    assert get_policy() is before
    assert get_policy().on_unsupported == "fail"


def test_reset_policy_restores_defaults() -> None:
    set_policy(on_unsupported="skip", seed=123)
    reset_policy()
    assert get_policy().on_unsupported == "fail"
    assert get_policy().seed == 20240101


def test_as_dict_covers_every_field() -> None:
    payload = RunPolicy().as_dict()
    assert set(payload) == {
        "on_unsupported",
        "tolerance_multiplier",
        "strict_backends",
        "device_index",
        "seed",
    }
