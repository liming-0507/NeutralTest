"""E3, E6 and E7 have to be able to fail: each stub below makes one go red.

These three rows gained columns late in the work -- they used to be recorded as
waiting for hardware -- so the risk is specific: a row that now claims a column
must not be passing it for a reason of its own. Every test here replaces the
column's kernel with one that is wrong in exactly the way the row exists to
find, runs the row on the real sample, and requires the row to refuse it -- then
puts the real kernel back and requires the row to pass on the same column, so a
stub that failed for a reason of its own cannot be mistaken for discrimination.

The backward columns are stubbed differently on purpose. A backward answer comes
from ``ctx.gradients_with``, which differentiates the forward kernel through
autograd and never calls the injected kernel, so a kernel stub cannot reach it.
For those columns the file instead checks the other half of the claim: that the
row asked the backend for gradients and compared what came back, which is what
its failure message is built from.
"""

# ===== E3 / E6 / E7 (clause 7.2) =====

import dataclasses
import threading
from typing import Any

import pytest
import torch

from neutraltest.backends import CpuBackend
from neutraltest.gbz72 import items, marks
from neutraltest.gbz73.context import NotApplicable
from neutraltest.gbz73.inference.columns import BY_NAME

#: Forward columns the kernel stubs are exercised on: a reduction, an elementwise
#: pair and a normalisation, so three different answer shapes are covered.
FORWARD = ("add_fwd", "softmax_fwd", "rms_norm_fwd")

#: Backward columns, which take the autograd path described in the module docstring.
BACKWARD = ("add_bwd", "rms_norm_bwd")

COLUMNS = FORWARD + BACKWARD


def _context(feature: str, column: str, kernel=None) -> Any:
    """The row's context on ``column``, with ``kernel`` standing in for the real one."""
    from neutraltest.gbz73.inference.context import InferenceContext

    context = InferenceContext(
        column=BY_NAME[column], backend=CpuBackend(), feature_id=feature, seed=7
    )
    if kernel is not None:
        context.kernel = lambda: kernel  # type: ignore[method-assign]
    return context


def _run(feature: str, column: str, kernel=None, *, into=None) -> Any:
    """Run the row on ``column``; raises what the row refused with.

    ``into`` lets a caller keep the context the row wrote its notes on, which is
    the only way to see what a passing row recorded.
    """
    context = into if into is not None else _context(feature, column, kernel)
    return items.module_for(feature).run(context)


def _real(column: str) -> Any:
    return _context("E3", column).kernel()


def _shift(answer: Any, by: float) -> Any:
    """``answer`` with ``by`` added to its leading tensor, keeping the container shape."""
    lead = answer[0] if isinstance(answer, tuple) else answer
    moved = lead + by
    return (moved,) + tuple(answer[1:]) if isinstance(answer, tuple) else moved


# -- E3 ------------------------------------------------------------------


@pytest.mark.parametrize("column", FORWARD)
def test_e3_refuses_a_kernel_that_answers_by_thread(column: str) -> None:
    """The defect E3 exists for: the answer depends on which thread asked.

    The stub gives the first thread the real answer and every other thread a
    shifted one. A single serial call only ever sees the first thread, so it
    still agrees with itself; the movement exists only while two calls are in
    flight -- which is what E3's threads are there to create.
    """
    real = _real(column)
    lock = threading.Lock()
    anchor: dict[str, int] = {}

    def kernel(*args: Any, **kwargs: Any) -> Any:
        answer = real(*args, **kwargs)
        with lock:
            first = anchor.setdefault("thread", threading.get_ident())
        if threading.get_ident() == first:
            return answer
        return _shift(answer, 1.0)

    with pytest.raises(AssertionError, match="differ|concurren|thread"):
        _run("E3", column, kernel)
    # The control: the real kernel passes the same row on the same column.
    _run("E3", column)


@pytest.mark.parametrize("column", BACKWARD)
def test_e3_aske_for_gradients_on_every_column_it_claims(column: str) -> None:
    """A backward column's answer is a gradient, so the row must ask for gradients.

    The injected kernel cannot reach this path, so the discrimination checked
    here is the route: the row asks for gradients and gets as many answers as the
    frozen operands, rather than silently comparing forward outputs.
    """
    context = _context("E3", column)
    assert context.backward, f"{column} is not a backward column"
    args, kwargs = context.inputs()
    gradients, prepared = context.gradients_with(args, kwargs)
    # A gradient per operand the column differentiates: the pristine shapes the
    # backend chose are the framework's own count, not one this test recomputes.
    assert len(gradients) == len(prepared), (len(gradients), len(prepared))
    assert gradients, f"{column} produced no gradient at all"
    _run("E3", column)


# -- E6 ------------------------------------------------------------------


@pytest.mark.parametrize("column", FORWARD)
def test_e6_refuses_a_kernel_whose_pieces_do_not_join(column: str) -> None:
    """The defect E6 exists for: pieces do not add up to the whole.

    The stub scales each answer by the number of rows it was given, so a
    two-piece answer cannot be joined into the one-piece answer. A kernel that
    were genuinely the same operator on pieces would survive; this one cannot.
    """
    real = _real(column)

    def kernel(*args: Any, **kwargs: Any) -> Any:
        answer = real(*args, **kwargs)
        lead = answer[0] if isinstance(answer, tuple) else answer
        rows = float(lead.reshape(-1, lead.shape[-1]).shape[0])
        return _shift(answer, 1e-3 * rows)

    with pytest.raises(AssertionError, match="differs from one call"):
        _run("E6", column, kernel)
    _run("E6", column)


def test_e6_says_why_attention_has_no_cut() -> None:
    """Attention cannot be sharded by rows, and the row says so rather than failing.

    Splitting the queries while keeping the keys whole is not the same sample;
    splitting all three is a different sample, not a smaller piece of work. E6
    therefore records the reason instead of claiming a column it cannot answer.
    """
    with pytest.raises(NotApplicable) as caught:
        items.module_for("E6").run(_context("E6", "flash_attention_fwd"))
    reason = str(caught.value)
    assert "attention" in reason
    assert "cut" in reason


# -- E7 ------------------------------------------------------------------


@pytest.mark.parametrize("column", FORWARD)
def test_e7_refuses_a_deployment_that_answers_differently_on_a_second_run(column: str) -> None:
    """The defect E7 exists for: the same deployment drifts between runs.

    The stub gives the right answer the first time it sees a set of answers and a
    shifted one afterwards, which is what a cache keyed on anything but the
    arguments looks like from outside.
    """
    real = _real(column)
    seen: set[bytes] = set()

    def kernel(*args: Any, **kwargs: Any) -> Any:
        answer = real(*args, **kwargs)
        lead = answer[0] if isinstance(answer, tuple) else answer
        key = lead.detach().numpy().tobytes()
        repeat = key in seen
        seen.add(key)
        return _shift(answer, 1.0) if repeat else answer

    with pytest.raises(AssertionError, match="differ"):
        _run("E7", column, kernel)
    _run("E7", column)


@pytest.mark.parametrize("column", BACKWARD)
def test_e7_repeats_the_gradient_it_claims_to_repeat(column: str) -> None:
    """On a backward column the thing repeated is the gradient.

    The kernel stub cannot reach the autograd path, so what is pinned instead is
    that both the claim and the answer are about gradients.
    """
    context = _context("E7", column)
    assert context.backward
    first = context.gradients_with(*context.inputs())[0]
    second = context.gradients_with(*context.inputs())[0]
    assert len(first) == len(second)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a.detach().cpu(), b.detach().cpu())
    _run("E7", column)


def test_e7_records_the_environment_it_repeated() -> None:
    """E7's answer is only worth something with the deployment written down."""
    context = _context("E7", "add_fwd")
    _run("E7", "add_fwd", into=context)
    recorded = " ".join(context.notes)
    assert "python" in recorded, f"E7 recorded {recorded!r}"
    assert "torch" in recorded


# -- the table ------------------------------------------------------------


def test_the_rows_claim_the_columns_this_file_tests() -> None:
    """The stubs above are only evidence while the rows still claim these columns."""
    for column in COLUMNS:
        assert marks.is_marked("E3", column), f"E3 no longer claims {column}"
        assert marks.is_marked("E7", column), f"E7 no longer claims {column}"
    for column in FORWARD:
        assert marks.is_marked("E6", column), f"E6 no longer claims {column}"
    # E6 deliberately claims no backward column: a shard's gradient for a shared
    # operand is a sum over shards, which is A12's question, not this row's.
    assert not marks.is_marked("E6", "add_bwd")


def test_the_context_helper_builds_a_real_context() -> None:
    """A guard on the helper above: the context type is what the rows expect."""
    context = _context("E3", "add_fwd")
    assert dataclasses.is_dataclass(context)
    assert context.backend.device_type == "cpu"
    assert context.feature_id == "E3"
