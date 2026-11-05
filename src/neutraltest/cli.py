"""``neutraltest`` command line: inspect backends, smoke test them, run the suite.

Three verbs, deliberately few:

``info``
    What PyTorch sees: every discovered backend, its device type, whether it is
    usable, and which dtypes it can allocate.

``doctor``
    Run a fixed set of tiny operators on every selected backend and print a
    pass/fail row per operation.  This is the "did my adapter get wired up at
    all?" check, before blaming the conformance suite.

``test``
    ``pytest`` with the bundled conformance suite, forwarding every argument.
"""

from __future__ import annotations

import argparse
import platform
import sys
from collections.abc import Sequence

import torch

from . import __version__
from .backends.base import Backend
from .backends.registry import BackendNotFound, discover, resolve_spec
from .report import render_table
from .tolerance import tolerance_for

__all__ = ["main"]

_PROBE_DTYPES = (
    torch.float32,
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int32,
    torch.int64,
    torch.bool,
)


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:1] == ["test"]:
        # Everything after "test" belongs to pytest; argparse cannot both own a
        # subcommand and forward an arbitrary tail such as "-q -k matmul".
        return _test(_forward_options(raw[1:]))

    parser = _build_parser()
    args = parser.parse_args(raw)
    command = args.command
    if command == "info":
        return _info(args)
    if command == "doctor":
        return _doctor(args)
    return _test(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neutraltest",
        description="Vendor-neutral multi-backend conformance testing for PyTorch.",
    )
    parser.add_argument("--version", action="version", version=f"neutraltest {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    info = subparsers.add_parser("info", help="list discovered backends")
    _add_backend_arguments(info)

    doctor = subparsers.add_parser("doctor", help="smoke test the selected backends")
    _add_backend_arguments(doctor)

    test = subparsers.add_parser(
        "test",
        help="run the conformance suite with pytest",
        description="Run the bundled conformance suite. Any argument this command does not "
        "recognise is forwarded to pytest verbatim, e.g. 'neutraltest test --backend=cpu -q'.",
    )
    _add_backend_arguments(test)
    test.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to pytest",
    )
    return parser


def _forward_options(tokens: Sequence[str]) -> argparse.Namespace:
    """Split ``test`` arguments into our two options and a pytest tail."""
    parser = argparse.ArgumentParser(prog="neutraltest test", add_help=False)
    parser.add_argument("--backend", default=None, metavar="SPEC")
    parser.add_argument("--device-index", type=int, default=None)
    known, rest = parser.parse_known_args(list(tokens))
    return argparse.Namespace(
        backend=known.backend or "auto",
        device_index=known.device_index if known.device_index is not None else 0,
        pytest_args=rest,
    )


def _add_backend_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--backend",
        default="auto",
        metavar="SPEC",
        help="'auto', 'all', or a comma separated list such as 'cpu,xpu'",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        dest="device_index",
        metavar="N",
        help="device ordinal to probe (default: 0)",
    )


# --------------------------------------------------------------------------- info


def _info(args: argparse.Namespace) -> int:
    discovered = discover()
    try:
        selected = resolve_spec(args.backend, discovered)
    except BackendNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    chosen = {backend.name for backend in selected}

    print(f"neutraltest {__version__}")
    print(f"python      {platform.python_version()} ({platform.machine()})")
    print(f"torch       {torch.__version__}")
    print(f"registered  {_privateuse1_name()}")
    print()

    rows = []
    for backend in discovered:
        rows.append(
            (
                ("* " if backend.name in chosen else "  ") + backend.name,
                str(backend.mechanism),
                backend.device_type,
                "yes" if backend.is_available() else "no",
                str(backend.device_count() if backend.is_available() else 0),
            )
        )
    print(render_table(rows, ("", "mechanism", "device_type", "available", "devices")))
    print("\n* selected by --backend=" + str(args.backend))

    available = [backend for backend in selected if backend.is_available()]
    if available:
        print()
        dtype_rows = []
        for backend in available:
            cells = ("yes" if backend.supports_dtype(dtype) else "-" for dtype in _PROBE_DTYPES)
            dtype_rows.append((backend.name, *cells))
        print(
            render_table(
                dtype_rows, ("backend", *(str(d).removeprefix("torch.") for d in _PROBE_DTYPES))
            )
        )
        print()
        print("tolerance budgets (rtol, atol):")
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            print(
                f"  {str(dtype).removeprefix('torch.'):9s} "
                f"elementwise {tolerance_for(dtype, 'elementwise')!s:26s} "
                f"matmul {tolerance_for(dtype, 'matmul')!s:26s} "
                f"conv {tolerance_for(dtype, 'convolution')}"
            )
    return 0


def _privateuse1_name() -> str:
    try:
        return str(torch._C._get_privateuse1_backend_name())
    except Exception:  # noqa: BLE001 - ancient/embedded builds
        return "<unknown>"


# --------------------------------------------------------------------------- doctor


def _doctor(args: argparse.Namespace) -> int:
    discovered = discover()
    try:
        selected = resolve_spec(args.backend, discovered)
    except BackendNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not selected:
        print("error: no backend selected", file=sys.stderr)
        return 2

    failures = 0
    for backend in selected:
        if not backend.is_available():
            print(f"{backend.name}: unavailable — skipped")
            continue
        print(f"{backend.name}: device_type={backend.device_type} index={args.device_index}")
        rows = []
        for check, run in _smoke_checks(backend, args.device_index):
            try:
                detail = run() or "ok"
                rows.append((check, "PASS", str(detail)))
            except Exception as exc:  # noqa: BLE001 - the point is to report it
                failures += 1
                rows.append((check, "FAIL", f"{type(exc).__name__}: {_first_line(exc)}"))
        print(render_table(rows, ("check", "result", "detail")))
        print()
    return 1 if failures else 0


def _smoke_checks(backend: Backend, index: int):
    """Yield ``(name, callable)`` pairs; each callable raises on failure."""

    target = backend.make_device(index)

    def make(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(0)
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(
            dtype=dtype, device=target
        )

    def allocate() -> str:
        device = backend.make_device(index)
        tensor = torch.zeros(4, device=device)
        return f"{tuple(tensor.shape)} on {tensor.device}"

    def elementwise() -> str:
        left, right = make(64, 64), make(64, 64)
        result = torch.relu(left + right)
        backend.synchronize()
        return f"sum={float(result.double().sum()):.4f}"

    def matmul() -> str:
        left, right = make(32, 64), make(64, 32)
        result = left @ right
        backend.synchronize()
        return f"{tuple(result.shape)}"

    def reduction() -> str:
        tensor = make(128, 64)
        return f"mean={float(tensor.mean()):.6f}"

    def softmax() -> str:
        tensor = make(16, 32)
        return f"rowsum={float(tensor.softmax(-1).sum(-1).mean()):.6f}"

    def backward() -> str:
        tensor = make(16, 16).requires_grad_(True)
        (tensor * tensor).sum().backward()
        backend.synchronize()
        grad = tensor.grad
        assert grad is not None, "backward() produced no gradient for a differentiable op"
        return f"grad_sum={float(grad.double().sum()):.4f}"

    def transfer() -> str:
        host = torch.arange(8, dtype=torch.float32)
        device_tensor = host.to(backend.make_device(index))
        return f"roundtrip={bool(torch.equal(device_tensor.cpu(), host))}"

    return [
        ("allocate", allocate),
        ("elementwise", elementwise),
        ("matmul", matmul),
        ("reduction", reduction),
        ("softmax", softmax),
        ("backward", backward),
        ("transfer", transfer),
    ]


# --------------------------------------------------------------------------- test


def _test(args: argparse.Namespace) -> int:
    import pytest

    forwarded = list(args.pytest_args or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    suite = _suite_directory()
    argv = []
    if args.backend != "auto":
        argv += [f"--backend={args.backend}"]
    if args.device_index:
        argv += [f"--device-index={args.device_index}"]
    argv += forwarded
    if suite is not None and not _mentions_path(forwarded):
        argv.append(str(suite))
    if suite is None:
        print(
            "warning: the packaged conformance suite was not found; run pytest in your "
            "own test directory instead",
            file=sys.stderr,
        )
    print("running: pytest " + " ".join(argv))
    return int(pytest.main(argv))


def _suite_directory():
    from pathlib import Path

    candidate = Path(__file__).resolve().parent / "conformance"
    return candidate if candidate.is_dir() else None


def _mentions_path(arguments: Sequence[str]) -> bool:
    return any(not argument.startswith("-") for argument in arguments)


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return text[0][:200] if text else repr(exc)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
