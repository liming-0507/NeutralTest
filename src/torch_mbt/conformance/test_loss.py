"""Loss functions."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from torch_mbt.conformance.support import params, run_case
from torch_mbt.seeding import randn
from torch_mbt.testing import OpCase

F32 = (torch.float32,)
F32_BF16 = (torch.float32, torch.bfloat16)


def _pair(dtype: torch.dtype, generator: torch.Generator):
    return (randn((8, 16), dtype, generator), randn((8, 16), dtype, generator)), {}


def _logits_target(dtype: torch.dtype, generator: torch.Generator):
    logits = randn((4, 6), dtype, generator)
    target = torch.randint(0, 6, (4,), generator=generator)
    return (logits, target), {}


def _log_probs_target(dtype: torch.dtype, generator: torch.Generator):
    logits = randn((4, 6), dtype, generator)
    target = torch.randint(0, 6, (4,), generator=generator)
    return (F.log_softmax(logits, dim=-1), target), {}


def _bce(dtype: torch.dtype, generator: torch.Generator):
    logits = randn((8, 16), dtype, generator)
    target = torch.sigmoid(randn((8, 16), dtype, generator))
    return (logits, target), {}


def _kl(dtype: torch.dtype, generator: torch.Generator):
    logits = randn((4, 6), dtype, generator)
    q = F.softmax(randn((4, 6), dtype, generator), dim=-1)
    return (F.log_softmax(logits, dim=-1), q), {"reduction": "batchmean"}


CASES = [
    OpCase(
        id="mse-loss",
        family="loss",
        op="mse_loss",
        fn=F.mse_loss,
        build=_pair,
        dtypes=F32_BF16,
        kind="loss",
        grad=True,
        grad_inputs=(0, 1),
    ),
    OpCase(
        id="l1-loss",
        family="loss",
        op="l1_loss",
        fn=F.l1_loss,
        build=_pair,
        dtypes=F32_BF16,
        kind="loss",
    ),
    OpCase(
        id="smooth-l1-loss",
        family="loss",
        op="smooth_l1_loss",
        fn=F.smooth_l1_loss,
        build=_pair,
        dtypes=F32_BF16,
        kind="loss",
    ),
    OpCase(
        id="cross-entropy",
        family="loss",
        op="cross_entropy",
        fn=F.cross_entropy,
        build=_logits_target,
        dtypes=F32,
        kind="loss",
    ),
    OpCase(
        id="nll-loss",
        family="loss",
        op="nll_loss",
        fn=F.nll_loss,
        build=_log_probs_target,
        dtypes=F32,
        kind="loss",
    ),
    OpCase(
        id="bce-with-logits",
        family="loss",
        op="binary_cross_entropy_with_logits",
        fn=F.binary_cross_entropy_with_logits,
        build=_bce,
        dtypes=F32_BF16,
        kind="loss",
    ),
    OpCase(
        id="kl-div", family="loss", op="kl_div", fn=F.kl_div, build=_kl, dtypes=F32, kind="loss"
    ),
]


@pytest.mark.parametrize("case,dtype", params(CASES))
def test_loss(case, dtype, backend):
    run_case(case, dtype, backend)
