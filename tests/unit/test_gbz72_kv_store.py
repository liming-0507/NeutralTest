"""The cache store's contract on slot lists it cannot honour."""

from __future__ import annotations

import pytest
import torch

from neutraltest.gbz73.inference.ops import kv_write_forward


def _operands(rows: int, tokens: int, slots: list[int]):
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(tokens, 4, generator=generator),
        torch.randn(tokens, 4, generator=generator),
        torch.randn(rows, 4, generator=generator),
        torch.randn(rows, 4, generator=generator),
        torch.tensor(slots, dtype=torch.int64),
    )


def test_a_repeated_slot_is_refused() -> None:
    with pytest.raises(ValueError, match="names a row twice"):
        kv_write_forward(*_operands(rows=1, tokens=3, slots=[0, 0, 0]))


def test_a_slot_past_the_last_row_is_refused() -> None:
    with pytest.raises(ValueError, match="leaves the cache"):
        kv_write_forward(*_operands(rows=2, tokens=2, slots=[0, 5]))


def test_distinct_slots_store_their_own_token() -> None:
    keys, values, cache_k, cache_v, slots = _operands(rows=4, tokens=2, slots=[1, 3])
    stored_k, stored_v = kv_write_forward(keys, values, cache_k, cache_v, slots)
    assert torch.equal(stored_k[1], keys[0])
    assert torch.equal(stored_k[3], keys[1])
    assert torch.equal(stored_v[1], values[0])
    assert torch.equal(stored_k[0], cache_k[0])
