"""Vendor-neutral multi-backend conformance testing for PyTorch.

PyTorch's dispatch has a handful of registration mechanisms, not one per vendor.
This package tests *mechanisms* (``cpu``, ``cuda``, ``xpu``, ``mps``,
``privateuse1``) and maps them onto whichever devices a machine happens to have,
so the same conformance suite runs against a CPU box, a CUDA box, and an
out-of-tree accelerator without a line of vendor-specific code.

Quick start::

    pip install torch-mbt
    torch-mbt info
    torch-mbt doctor --backend=cpu
    pytest --backend=cpu tests/

Writing a test::

    from torch_mbt import OpCase, check_case

    def test_add(mbt):
        case = OpCase(
            id="add/elementwise",
            family="elementwise",
            op="add",
            fn=torch.add,
            build=lambda dtype, gen: ((randn((8, 8), dtype, gen), randn((8, 8), dtype, gen)), {}),
        )
        check_case(case, mbt.backend, torch.float32)
"""

from __future__ import annotations

from .backends import Backend, Mechanism, discover, resolve_spec
from .compare import assert_all_finite, assert_close, assert_grad_close
from .errors import (
    BackendNotFound,
    BackendUnavailable,
    TorchMbtError,
    UnsupportedByBackend,
    is_unsupported_error,
)
from .policy import (
    RunPolicy,
    get_policy,
    patched_policy,
    reset_policy,
    set_policy,
)
from .report import Recorder, Result, render_table
from .seeding import make_generator, randn, seed_all, seeded
from .testing import (
    OpCase,
    check_backward,
    check_case,
    evaluate,
    move_to,
    unsupported_policy,
)
from .tolerance import Tolerance, tolerance_for

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # backends
    "Backend",
    "Mechanism",
    "discover",
    "resolve_spec",
    # errors
    "TorchMbtError",
    "BackendNotFound",
    "BackendUnavailable",
    "UnsupportedByBackend",
    "is_unsupported_error",
    # policy
    "RunPolicy",
    "get_policy",
    "set_policy",
    "reset_policy",
    "patched_policy",
    # tolerance and comparison
    "Tolerance",
    "tolerance_for",
    "assert_close",
    "assert_grad_close",
    "assert_all_finite",
    # seeding
    "make_generator",
    "randn",
    "seed_all",
    "seeded",
    # cases
    "OpCase",
    "check_case",
    "check_backward",
    "evaluate",
    "move_to",
    "unsupported_policy",
    # reporting
    "Recorder",
    "Result",
    "render_table",
]
