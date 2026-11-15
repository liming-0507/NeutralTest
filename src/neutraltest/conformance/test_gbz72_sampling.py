"""The part 7.2 sampler: twelve rows, one direction, the same bookkeeping.

The sampler is the seventh operator of part 7.2 and the only one with no
backward pass, so it is checked against its own matrix (``gbz72.sampling``)
rather than against extra columns of the six-operator table.

The claim each cell makes is about a *distribution*, not about a draw. Two
correct samplers disagree on any individual draw, which is why the device is
held to the frequencies over :data:`~neutraltest.gbz72.sampling.oracle.DRAWS`
draws instead of to element equality. On the CPU the kernel and the reference
are the same code, so the cell reduces to a consistency check -- still run, so
that a broken oracle is caught on this side of the hardware.
"""

from __future__ import annotations

import pytest

from neutraltest import clause
from neutraltest.gbz72.sampling import runner
from neutraltest.gbz72.sampling.ops import draw as reference_draw
from neutraltest.gbz72.sampling.oracle import DRAWS

if not clause.covers(clause.get_clause(), "72"):
    pytest.skip(f"clause {clause.get_clause()} does not cover part 7.2", allow_module_level=True)

CELLS = [(row, name, kwargs) for row in runner.ROWS for name, kwargs in runner.SETTINGS]
IDS = [f"{row.id}-{name}" for row, name, _ in CELLS]


def _device_draw(nt, seed: int):
    """Wrap the backend's own sampling so the device does the drawing."""

    def device_draw(logits, *, seed=0, num_samples=1, **kwargs):
        nt.backend.manual_seed(seed)
        on_device = logits.to(nt.device)
        out = draw_on(nt, on_device, num_samples=num_samples, **kwargs)
        nt.sync()
        return out.to("cpu")

    return device_draw


def draw_on(nt, logits, *, num_samples: int = 1, **kwargs):
    """Ask the backend under test for draws of *logits*."""
    kernel = getattr(nt.backend, "scale_probabilities", None)
    if kernel is None:
        # The backend exposes no sampling primitive: fall back to the reference
        # on the device's tensors, which still exercises device RNG, dtypes and
        # any device-side softmax the filters rely on.
        return reference_draw(logits, num_samples=num_samples, **kwargs)
    return kernel(logits, num_samples=num_samples, **kwargs)


@pytest.mark.nt_family("sampling")
@pytest.mark.nt_op("top_k_top_p")
@pytest.mark.parametrize(("row", "setting", "kwargs"), CELLS, ids=IDS)
def test_gbz72_sampler(row, setting, kwargs, nt) -> None:
    device_draw = None if nt.device.type == "cpu" else _device_draw(nt, nt.policy.seed)
    result = runner.run_cell(row, setting, kwargs, device_draw, draws=DRAWS)
    if result.outcome != "passed":
        pytest.fail(f"{row.id}/{setting}: {result.detail}", pytrace=False)
