"""Unit tests for the part 7.2 sampler: filters, oracle, and the matrix itself.

The interesting tests here are the ones that make something *fail*. A matrix
that passes on a correct sampler proves little on its own; these tests pin down
that it goes red on a generator that is wrong in a specific way.
"""

from __future__ import annotations

import pytest
import torch

from neutraltest.gbz72.sampling import items, ops, oracle, runner

# ------------------------------------------------------------------ the filters


def test_top_k_keeps_the_k_most_likely() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    mass = ops.reference_mass(logits, top_k=2)
    assert mass.tolist()[0][:3] == [0.0, 0.0, 0.0]
    assert mass.sum().item() == pytest.approx(1.0, abs=1e-6)


def test_top_p_drops_the_tail() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    full = ops.reference_mass(logits)
    cut = ops.reference_mass(logits, top_p=0.7)
    # The two most likely tokens carry 0.8707 of the mass; the third would push
    # the running total past 0.7, so it is dropped along with everything below.
    assert cut[0][:3].tolist() == [0.0, 0.0, 0.0]
    assert cut.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert float(full[0][4]) < 0.7


def test_top_p_of_one_changes_nothing() -> None:
    logits = torch.tensor([[0.5, -1.0, 2.0]])
    assert torch.allclose(ops.reference_mass(logits, top_p=1.0), ops.reference_mass(logits))


def test_temperature_above_one_flattens() -> None:
    logits = torch.tensor([[0.0, 1.0, 2.0]])
    hot = ops.reference_mass(logits, temperature=2.0)
    cold = ops.reference_mass(logits)
    assert float(hot.max()) < float(cold.max())


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -1.0}, "temperature"),
        ({"top_k": 0}, "top_k"),
        ({"top_k": -3}, "top_k"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
    ],
)
def test_out_of_range_filters_are_rejected(kwargs: dict, expected: str) -> None:
    with pytest.raises(ValueError, match=expected):
        ops.filter_logits(torch.zeros(2, 3), **kwargs)


def test_draws_stay_inside_the_allowed_set() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    picks = ops.draw(logits, seed=3, num_samples=256, top_k=2)
    assert set(picks.reshape(-1).tolist()) <= {2, 3}


def test_a_seed_makes_a_draw_repeatable() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    first = ops.draw(logits, seed=7, num_samples=64)
    second = ops.draw(logits, seed=7, num_samples=64)
    assert torch.equal(first, second)


# -------------------------------------------------------------------- the oracle


def _batch() -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(4, 16, generator=generator)


def test_frequency_normalises_per_row() -> None:
    measured = oracle.frequency(ops.draw, _batch(), draws=512, top_p=0.8)
    assert measured.shape == (4, 16)
    assert torch.allclose(measured.sum(-1), torch.ones(4, dtype=torch.float64), atol=1e-9)


def test_a_faithful_generator_matches_the_reference() -> None:
    # The return value is the worst excess: at or below zero means no token was
    # drawn more often than its reference mass allows for this many draws.
    assert oracle.assert_matches(ops.draw, _batch(), draws=2048, top_p=0.9) <= 0.0


def test_an_argmax_generator_is_caught() -> None:
    def argmax_draw(logits, *, seed=0, num_samples=1, **kwargs):
        best = logits.argmax(-1, keepdim=True)
        return best.expand(*logits.shape[:-1], num_samples)

    with pytest.raises(AssertionError, match="against a reference"):
        oracle.assert_matches(argmax_draw, _batch(), draws=1024)


def test_a_generator_that_ignores_the_cutoff_is_caught() -> None:
    def blind(logits, *, seed=0, num_samples=1, **kwargs):
        return ops.draw(logits, seed=seed, num_samples=num_samples)

    with pytest.raises(AssertionError, match="against a reference"):
        oracle.assert_matches(blind, _batch(), draws=1024, top_p=0.5)


def test_a_constant_generator_is_caught() -> None:
    def constant(logits, *, seed=0, num_samples=1, **kwargs):
        return torch.zeros(*logits.shape[:-1], num_samples, dtype=torch.int64)

    with pytest.raises(AssertionError):
        oracle.assert_matches(constant, _batch(), draws=1024)


def test_allowance_has_a_floor_for_erased_tokens() -> None:
    mass = torch.tensor([0.0, 1.0])
    allowance = oracle.allowance(mass, draws=4096)
    assert float(allowance.min()) == pytest.approx(oracle.FLOOR)
    assert float(allowance.max()) <= 1.0


# ------------------------------------------------------------------- the matrix


def test_every_row_is_checked_against_every_setting() -> None:
    report = runner.run_all(None)
    assert len(report.results) == len(items.ROWS) * len(runner.SETTINGS)
    assert not report.failed


def test_the_matrix_goes_red_on_a_blind_generator() -> None:
    def blind(logits, *, seed=0, num_samples=1, **kwargs):
        return ops.draw(logits, seed=seed, num_samples=num_samples)

    # A quarter of the production draw count: enough to separate a filter-blind
    # generator from a faithful one, without paying for full precision twice.
    report = runner.run_all(blind, draws=oracle.DRAWS // 4)
    assert report.failed
    # Plain sampling has nothing to ignore, so those cells still pass.
    assert report.cell("A1", "plain").outcome == "passed"


def test_failures_are_recorded_with_their_reason() -> None:
    def broken(logits, **kwargs):
        raise RuntimeError("no sampler registered")

    result = runner.run_cell(items.ROWS[0], "plain", {}, broken)
    assert result.outcome == "failed"
    assert "no sampler registered" in result.detail


def test_the_row_catalogue_has_the_documented_ids() -> None:
    assert [row.id for row in items.ROWS] == [
        "A1",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "B3",
        "B4",
        "C5",
        "D2",
        "F1",
    ]
