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

``gbz73``
    The FP8 clause matrix: every row of the standard against every operator in
    the catalogue, printed as a table.  ``--json`` for the machine-readable form.

``sdk``
    Where this environment sits against the SDK release it claims to run, and
    against PyTorch's own published support window.
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
    if raw[:1] == ["gbz73"]:
        args = _build_parser().parse_args(raw)
        return _gbz73(args)
    if raw[:1] == ["sdk"]:
        args = _build_parser().parse_args(raw)
        return _sdk(args)
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
    if command == "known":
        from .known_cli import run_known

        return run_known(args)
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

    sdk = subparsers.add_parser(
        "sdk",
        help="show where this environment sits against its SDK release and PyTorch",
        description="Measure the interpreter against the SDK release it claims to run "
        "(torch window, python window, per component) and against PyTorch's own published "
        "support window. Deviations are reported, not enforced.",
    )
    _add_sdk_arguments(sdk)
    sdk.add_argument("--json", action="store_true", help="machine readable output")

    doctor = subparsers.add_parser("doctor", help="smoke test the selected backends")
    _add_backend_arguments(doctor)

    test = subparsers.add_parser(
        "test",
        help="run the conformance suite with pytest",
        description="Run the bundled conformance suite. Any argument this command does not "
        "recognise is forwarded to pytest verbatim, e.g. 'neutraltest test --backend=cpu -q'.",
    )
    _add_backend_arguments(test)
    _add_sdk_arguments(test)
    test.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to pytest",
    )

    known = subparsers.add_parser(
        "known",
        help="list the registered backend deviations and re-check them",
        description="Known issues are measured, explained deviations that are reported as "
        "KNOWN instead of failed. Without --verify this lists them; with --verify it re-runs "
        "exactly those cells and says whether each deviation still reproduces.",
    )
    _add_backend_arguments(known)
    _add_sdk_arguments(known)
    known.add_argument(
        "--verify",
        action="store_true",
        help="re-run every registered cell and report what still reproduces",
    )
    known.add_argument(
        "--id", default=None, metavar="ID", help="print the full evidence for one entry"
    )
    known.add_argument(
        "--registry",
        action="append",
        default=None,
        metavar="PATH",
        help="extra registry file or directory (repeatable)",
    )
    known.add_argument("--json", action="store_true", help="machine readable output")
    known.add_argument(
        "--fail-on-fixed",
        action="store_true",
        dest="fail_on_fixed",
        help="exit 1 when a registered deviation no longer reproduces",
    )

    matrix = subparsers.add_parser(
        "gbz73",
        help="run the FP8 clause matrix and print the coverage table",
        description="Run every row of the FP8 clause matrix against every operator in the "
        "catalogue and print the result beside the coverage the text declares.",
    )
    _add_backend_arguments(matrix)
    _add_sdk_arguments(matrix)
    matrix.add_argument(
        "--ops", default=None, metavar="LIST", help="comma separated operator names"
    )
    matrix.add_argument(
        "--clause",
        default="73",
        choices=("73", "72", "all"),
        help="which clause to cover: 73 (default), 72, or all",
    )
    matrix.add_argument(
        "--features", default=None, metavar="LIST", help="comma separated row ids, e.g. A1,A2,B3"
    )
    matrix.add_argument(
        "--json", action="store_true", help="print the report as JSON instead of a table"
    )
    matrix.add_argument(
        "--fp32-precision",
        default="highest",
        choices=("highest", "high", "medium"),
        help="pin torch.set_float32_matmul_precision before measuring (default: highest)",
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


def _add_sdk_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sdk",
        default=None,
        metavar="FILE",
        help="SDK profile (JSON): the release's torch/python windows and its components",
    )
    parser.add_argument(
        "--on-sdk-out-of-band",
        default="warn",
        choices=("warn", "skip", "fail"),
        dest="on_sdk_out_of_band",
        help="what a deviation from the declared window means: warn (default), skip, or fail",
    )


def _load_sdk(args: argparse.Namespace):
    from . import sdk

    path = getattr(args, "sdk", None)
    profile = sdk.load_catalogue(path) if path else None
    return sdk, profile


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


def _gbz73(args: argparse.Namespace) -> int:
    from . import matrix as matrices
    from .gbz72.marks import STANDARD_COLUMNS
    from .gbz73.features import FEATURES
    from .gbz73.inference.columns import COLUMNS as COLUMNS72
    from .gbz73.ops import OPS

    discovered = discover()
    try:
        selected = resolve_spec(args.backend, discovered)
    except BackendNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    backends = [b for b in selected if b.is_available()]
    if not backends:
        print(
            f"error: no backend selected is available ({', '.join(b.name for b in selected)})",
            file=sys.stderr,
        )
        return 2

    wanted_ops: tuple[str, ...] | None = None
    if args.ops:
        wanted_ops = tuple(name.strip() for name in args.ops.split(",") if name.strip())
        operators = {op.name for op in OPS}
        columns = set(COLUMNS72) | set(STANDARD_COLUMNS)
        if args.clause == "73":
            unknown = set(wanted_ops) - operators
            if unknown:
                print(f"error: unknown operator(s): {', '.join(sorted(unknown))}", file=sys.stderr)
                return 2
        elif args.clause == "72":
            unknown = set(wanted_ops) - columns
            if unknown:
                print(f"error: unknown column(s): {', '.join(sorted(unknown))}", file=sys.stderr)
                return 2
        else:
            unknown = set(wanted_ops) - operators - columns
            if unknown:
                print(f"error: unknown operator(s): {', '.join(sorted(unknown))}", file=sys.stderr)
                return 2

    features = FEATURES
    if args.features:
        wanted = {name.strip().upper() for name in args.features.split(",") if name.strip()}
        unknown = wanted - {feature.id for feature in FEATURES}
        if unknown:
            print(f"error: unknown row(s): {', '.join(sorted(unknown))}", file=sys.stderr)
            return 2
        features = tuple(feature for feature in FEATURES if feature.id in wanted)

    failures = 0
    for backend in backends:
        runs = matrices.run(
            backend,
            clause=args.clause,
            ops=wanted_ops,
            features=features,
            fp32_precision=args.fp32_precision,
        )
        if args.json:
            import json

            if len(runs) == 1:
                payload = runs[0].as_dict()
            else:
                payload = {"clauses": {run.clause: run.as_dict() for run in runs}}
            print(json.dumps(payload, indent=2))
        else:
            print(
                f"backend: {backend.name}  device_type={backend.device_type}  index={args.device_index}"
            )
            for run in runs:
                if len(runs) > 1:
                    print(f"clause: {run.clause}")
                print(f"fp32 matmul precision: {run.report.fp32_precision}")
                print(run.render())
                print()
                print(run.summary())
                print()
                print(run.disclosure())
                declared, exercised = run.coverage()
                print()
                print(f"declared by the text: {declared} cells; exercised by this run: {exercised}")
        failures += sum(len(run.report.failed) for run in runs)
    return 1 if failures else 0


# --------------------------------------------------------------------------- sdk


def _sdk(args: argparse.Namespace) -> int:
    sdk, profile = _load_sdk(args)
    status = sdk.measure(profile)
    if args.json:
        import json

        print(json.dumps(status.snapshot.as_dict(), indent=2))
    else:
        print(sdk.render(status.snapshot))
        print()
        print(sdk.describe(status))
    if status.band == "deviation" and args.on_sdk_out_of_band == "fail":
        # An inspection command still reports with exit 1, so a pipeline can
        # gate on "this environment is the one we certified" without parsing.
        return 1
    return 0


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
