"""E1 and E2 have to be able to fail: each stub below makes one of them go red.

A row whose check cannot be made to fail is a check nobody has seen work. Every
test here replaces the column's kernel with one that is wrong in exactly the way
the row exists to find, runs the row on the real sample, and requires the row to
refuse it -- then puts the real kernel back and requires the row to pass on the
same column, so a stub that failed for a reason of its own cannot be mistaken for
discrimination.
"""

# ===== E1 / E2 (clause 7.2) =====

from typing import Any

import pytest
import torch

from neutraltest.backends import CpuBackend
from neutraltest.gbz72 import items, marks, reasons
from neutraltest.gbz73.inference.columns import BY_NAME

#: Columns the stubs are exercised on: the three families whose kernels return a
#: single tensor, forward and backward, so the reduction and the two elementwise
#: shapes are all covered.
COLUMNS = ("add_fwd", "softmax_fwd", "softmax_bwd", "rms_norm_fwd", "rms_norm_bwd")

Answers = tuple[torch.Tensor, ...]


def _context(feature: str, column: str, kernel=None) -> Any:
    """The row's context on ``column``, with ``kernel`` standing in for the real one.

    ``InferenceContext`` builds its kernel per call, so the only honest way to
    inject one is to answer for the instance: every other verb -- the sample, the
    device, the projection -- stays the real one.
    """
    from neutraltest.gbz73.inference.context import InferenceContext

    context = InferenceContext(
        column=BY_NAME[column], backend=CpuBackend(), feature_id=feature, seed=7
    )
    if kernel is not None:
        context.kernel = lambda: kernel  # type: ignore[method-assign]
    return context


def _real(column: str) -> Any:
    return _context("E1", column).kernel()


def _run(feature: str, column: str, kernel=None) -> Any:
    """Run the row on ``column``; raises what the row refused with."""
    return items.module_for(feature).run(_context(feature, column, kernel))


def _first(answer: Answers) -> torch.Tensor:
    return answer[0]


# -- E1 ------------------------------------------------------------------


def _drifting(real: Any) -> Any:
    """A kernel that answers differently on each call.

    A counter rather than a random draw: the movement has to be reproducible, so
    that the only thing the row can be reacting to is the movement itself.
    """
    calls = {"n": 0}

    def kernel(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        return real(*args, **kwargs) + calls["n"] * 1e-6

    return kernel


@pytest.mark.parametrize("column", COLUMNS)
def test_e1_refuses_a_kernel_that_drifts_between_calls(column: str) -> None:
    """The defect E1 exists for: the same call does not give the same answer."""
    with pytest.raises(AssertionError, match="differs from the first call"):
        _run("E1", column, _drifting(_real(column)))
    # The control: the real kernel passes the same row on the same column.
    _run("E1", column)


@pytest.mark.parametrize("column", COLUMNS)
def test_e1_refuses_a_kernel_that_reads_the_global_generator(column: str) -> None:
    """A result carrying a number nobody passed in is not a function of its inputs."""
    real = _real(column)

    def kernel(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **kwargs) + (float(torch.rand(())) + 1.0) * 1e-3

    with pytest.raises(AssertionError, match="global random generator"):
        _run("E1", column, kernel)


def test_e1_refuses_a_kernel_that_writes_through_its_input() -> None:
    """A call that edits what it was handed answers a different question.

    The stub scales the first floating operand in place before calling the real
    kernel, so what it returns is the answer for operands the caller never
    passed. What it leaves behind is not the evidence the row reports on -- the
    row restores nothing and does not claim to -- so the test matches the
    operand message, which a stub that merely produced new numbers would miss.
    """
    real = _real("add_fwd")

    def kernel(*args: Any, **kwargs: Any) -> Any:
        for value in args:
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                value.mul_(1.5)
                break
        return real(*args, **kwargs)

    with pytest.raises(AssertionError, match="modified its input"):
        _run("E1", "add_fwd", kernel)


def test_e1_refuses_a_kernel_whose_answer_follows_the_thread_pool() -> None:
    """A reduction ordered by the thread cut is the CPU form of an unordered one.

    The stub adds a value computed in chunks the size of the pool, so one worker
    and the process's own count add the operands in different orders.
    """
    real = _real("rms_norm_fwd")

    def kernel(*args: Any, **kwargs: Any) -> Any:
        answer = real(*args, **kwargs)
        flat = answer.reshape(-1)
        chunk = max(1, min(torch.get_num_threads(), flat.numel()))
        # float32 on purpose: the chunk sums then carry rounding that depends on
        # where the chunks start, which a float64 sum would absorb.
        pieces = [
            flat[i : i + chunk].sum(dtype=torch.float32) for i in range(0, flat.numel(), chunk)
        ]
        return answer + float(sum(float(piece) for piece in pieces)) * 1e-3

    with pytest.raises(AssertionError, match="thread"):
        _run("E1", "rms_norm_fwd", kernel)


# -- E2 ------------------------------------------------------------------


def test_e2_reports_a_column_with_no_random_source_as_having_none() -> None:
    """The statement the clause asks for, made positively and with its evidence."""
    context = _context("E2", "softmax_fwd")
    items.module_for("E2").run(context)
    assert context.notes, "E2 passed without saying anything about the random source"
    assert context.notes[0].startswith("declares no random source")
    assert "seed" in context.notes[0]


@pytest.mark.parametrize("column", COLUMNS)
def test_e2_refuses_a_kernel_that_draws_from_the_cpu_generator(column: str) -> None:
    """A draw from the global stream moves when somebody reseeds the stream."""
    real = _real(column)

    def kernel(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **kwargs) + float(torch.rand(())) * 1e-3

    with pytest.raises(AssertionError, match="CPU generator"):
        _run("E2", column, kernel)


def test_e2_refuses_a_kernel_that_draws_on_a_stream_nothing_reseeds() -> None:
    """A second stream the row does not reseed is still a source, and still red.

    The stub draws from a generator of its own, so the host probe stays quiet;
    the row must not pass it on the grounds that the host stream did not move.
    """
    real = _real("add_fwd")
    own = torch.Generator().manual_seed(11)

    def kernel(*args: Any, **kwargs: Any) -> Any:
        return real(*args, **kwargs) + torch.rand((1,), generator=own, dtype=torch.float32) * 1e-3

    with pytest.raises(AssertionError, match="differs from the first call"):
        _run("E1", "add_fwd", kernel)


# -- the table -----------------------------------------------------------


def test_e1_and_e2_cover_every_column_and_keep_no_row_reason() -> None:
    """Both rows claim all twelve columns, and neither keeps a "not applicable" text."""
    for feature in ("E1", "E2"):
        assert marks.columns_of(feature) == marks.TABLE_ORDER
        assert feature not in reasons.ROW_REASONS
        assert feature not in reasons.PARTIAL_REASONS


def test_e1_and_e2_are_not_the_same_check_twice() -> None:
    """Two green cells must not be the same evidence stated twice."""
    e1 = items.module_for("E1").__doc__ or ""
    e2 = items.module_for("E2").__doc__ or ""
    assert "Why this is not E7" in e1
    assert "E1" in e2
    assert "no random source" in e2
    assert "no random source" not in e1
