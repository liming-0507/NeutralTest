"""pytest plugin: options, fixtures, markers, per-backend reporting.

The plugin is registered automatically through the ``pytest11`` entry point, so
``pip install`` is all a test suite needs::

    pytest --backend=cpu
    pytest --backend=all --list-backends
    pytest --backend=xpu --on-unsupported=skip --tolerance-multiplier=2

Design notes that matter in practice:

* **Unavailable backends are parametrised, not filtered.**  A backend that is
  requested but missing produces one *skipped* test per case with the reason,
  rather than quietly vanishing from the run.  ``--backend-strict`` turns that
  into a hard error instead.
* **Seeding is per test.**  An autouse fixture reseeds torch/random so a failure
  reproduces in isolation.
* **Reporting is attributed per backend**, because "the suite failed" is useless
  when four backends ran.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from .backends.base import Backend, as_backend
from .backends.registry import BackendNotFound, discover, resolve_spec
from .errors import UnsupportedByBackend
from .policy import RunPolicy, get_policy, set_policy
from .report import SUMMARY_OUTCOMES, Recorder, Result, render_table
from .seeding import seed_all
from .tolerance import Tolerance, tolerance_for

__all__ = [
    "pytest_addoption",
    "pytest_configure",
    "pytest_cmdline_main",
    "pytest_report_header",
    "pytest_generate_tests",
    "pytest_collection_modifyitems",
    "pytest_terminal_summary",
    "MbtContext",
]

STATE = pytest.StashKey[dict[str, Any]]()
DEFAULT_SEED = 20240101

#: Declared by the plugin itself, so a suite needs no ``markers =`` ini block.
MARKERS = (
    ("mbt_backend(name, ...)", "run this test only on the named backend(s)"),
    ("mbt_exclude_backend(name, ...)", "never run this test on the named backend(s)"),
    ("mbt_family(name)", "operator family, used for report grouping"),
    ("mbt_op(name)", "operator name, used for report grouping"),
    ("mbt_grad", "the test exercises the backward pass"),
    ("mbt_slow", "the test is slow; deselect with -m 'not mbt_slow'"),
)


@dataclass(frozen=True)
class MbtContext:
    """Everything a conformance test needs, bundled by the ``mbt`` fixture."""

    backend: Backend
    device: torch.device
    policy: RunPolicy

    def tensor(
        self,
        *shape: int,
        dtype: torch.dtype = torch.float32,
        seed: int = 0,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Deterministic sample already living on the target device."""
        generator = torch.Generator(device="cpu").manual_seed(seed)
        base = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
        return base.to(dtype=dtype, device=self.device)

    @staticmethod
    def reference(tensor: torch.Tensor) -> torch.Tensor:
        """The CPU ``float64`` golden value for *tensor*."""
        return tensor.detach().to("cpu", torch.float64)

    def sync(self) -> None:
        self.backend.synchronize()

    def tolerance(self, dtype: torch.dtype, kind: str = "elementwise") -> Tolerance:
        return tolerance_for(dtype, kind, self.policy.tolerance_multiplier)


# --------------------------------------------------------------------------- options


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("torch-mbt", "multi-backend conformance testing")
    group.addoption(
        "--backend",
        action="store",
        default=os.environ.get("TORCH_MBT_BACKEND", "auto"),
        metavar="SPEC",
        help=(
            "backend selection: 'auto' (available, default), 'all' (every discovered, "
            "unavailable ones are skipped), or a comma separated list such as 'cpu,xpu'; "
            "globs are accepted"
        ),
    )
    group.addoption(
        "--device-index",
        action="store",
        type=int,
        default=int(os.environ.get("TORCH_MBT_DEVICE_INDEX", "0")),
        dest="device_index",
        metavar="N",
        help="device ordinal used by the `device` fixture (default: 0)",
    )
    group.addoption(
        "--list-backends",
        action="store_true",
        dest="list_backends",
        help="print the discovered backends and exit",
    )
    group.addoption(
        "--backend-strict",
        action="store_true",
        dest="backend_strict",
        help="fail instead of skipping when a requested backend is unavailable",
    )
    group.addoption(
        "--on-unsupported",
        action="store",
        choices=("fail", "skip", "xfail"),
        default="fail",
        dest="on_unsupported",
        help="what to do when a backend has no kernel for an operator (default: fail)",
    )
    group.addoption(
        "--tolerance-multiplier",
        action="store",
        type=float,
        default=1.0,
        dest="tolerance_multiplier",
        metavar="FLOAT",
        help="scale every tolerance, for noisy accelerators (default: 1.0)",
    )
    group.addoption(
        "--mbt-seed",
        action="store",
        type=int,
        default=DEFAULT_SEED,
        dest="mbt_seed",
        metavar="INT",
        help=f"base seed for per-test reseeding (default: {DEFAULT_SEED})",
    )
    group.addoption(
        "--mbt-report",
        action="store",
        default=None,
        dest="mbt_report",
        metavar="PATH",
        help="write a JSON report of per-backend results",
    )
    group.addoption(
        "--mbt-no-summary",
        action="store_true",
        dest="mbt_no_summary",
        help="suppress the per-backend terminal summary",
    )


# --------------------------------------------------------------------------- hooks


def pytest_configure(config: pytest.Config) -> None:
    global _ACTIVE_CONFIG
    for marker, description in MARKERS:
        config.addinivalue_line("markers", f"{marker}: {description}")
    spec = config.getoption("backend")
    discovered = discover()
    try:
        selected = resolve_spec(spec, discovered)
    except BackendNotFound as exc:
        raise pytest.UsageError(str(exc)) from None
    if not selected:
        raise pytest.UsageError(f"--backend={spec!r} selected no backend")

    set_policy(
        on_unsupported=config.getoption("on_unsupported"),
        tolerance_multiplier=config.getoption("tolerance_multiplier"),
        strict_backends=config.getoption("backend_strict"),
        device_index=config.getoption("device_index"),
        seed=config.getoption("mbt_seed"),
    )

    unavailable = [backend.name for backend in selected if not backend.is_available()]
    strict = config.getoption("backend_strict")
    asks_for_what_exists = spec.strip().lower() == "auto"
    if strict and unavailable and not asks_for_what_exists:
        raise pytest.UsageError(
            "requested backend(s) unavailable on this machine: " + ", ".join(unavailable)
        )

    config.stash[STATE] = {
        "spec": spec,
        "discovered": discovered,
        "selected": selected,
        "unavailable": unavailable,
        "recorder": Recorder(),
        "seed": config.getoption("mbt_seed"),
        "summary": not config.getoption("mbt_no_summary"),
        # snapshotted: a test is allowed to mutate the global policy, the report is not
        "on_unsupported": config.getoption("on_unsupported"),
    }

    seed_all(config.getoption("mbt_seed"))
    _ACTIVE_CONFIG = config


def pytest_cmdline_main(config: pytest.Config) -> int | None:
    if not config.getoption("list_backends", default=False):
        return None
    discovered = discover()
    try:
        selected = resolve_spec(config.getoption("backend"), discovered)
    except BackendNotFound:
        selected = discovered
    chosen = {backend.name for backend in selected}
    rows = []
    for backend in discovered:
        rows.append(
            (
                ("* " if backend.name in chosen else "  ") + backend.name,
                str(backend.mechanism),
                backend.device_type,
                "yes" if backend.is_available() else "no",
                str(backend.device_count() if backend.is_available() else 0),
                str(backend.priority),
                ", ".join(backend.aliases()),
            )
        )
    print(f"torch {torch.__version__}   python {sys.version.split()[0]}")
    print(
        render_table(
            rows, ("", "mechanism", "device_type", "available", "devices", "priority", "aliases")
        )
    )
    print("\n* = selected by --backend=" + str(config.getoption("backend")))
    return 0


def pytest_report_header(config: pytest.Config) -> list[str]:
    state = config.stash.get(STATE, None)
    if state is None:
        return []
    selected = state["selected"]
    policy = get_policy()
    names = ", ".join(
        f"{backend.name}{'' if backend.is_available() else ' (unavailable)'}"
        for backend in selected
    )
    lines = [
        f"torch-mbt: backends = {names}",
        f"torch-mbt: policy = on_unsupported:{policy.on_unsupported}, "
        f"tolerance_multiplier:{policy.tolerance_multiplier:g}, "
        f"device_index:{policy.device_index}, seed:{policy.seed}",
    ]
    if state["unavailable"]:
        lines.append(
            "torch-mbt: WARNING unavailable backends will be reported as skipped: "
            + ", ".join(state["unavailable"])
        )
    return lines


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "backend" not in metafunc.fixturenames:
        return
    state = metafunc.config.stash.get(STATE, None)
    if state is None:
        return
    selected = state["selected"]
    if not selected:
        return
    metafunc.parametrize(
        "backend",
        selected,
        ids=[backend.name for backend in selected],
        indirect=True,
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        backend = _backend_of(item)
        if backend is None:
            continue
        only = _marker_names(item, "mbt_backend")
        if only and backend.name not in only:
            item.add_marker(
                pytest.mark.skip(reason=f"backend {backend.name!r} not in {sorted(only)}")
            )
        excluded = _marker_names(item, "mbt_exclude_backend")
        if backend.name in excluded:
            item.add_marker(
                pytest.mark.skip(reason=f"backend {backend.name!r} is excluded by marker")
            )


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Any:
    """Let ``--on-unsupported`` govern errors a test raises itself.

    ``testing.unsupported_policy`` covers the conformance engine, but a test may
    call a backend directly; without this the option would only work for some
    tests, which is worse than not having it.
    """
    try:
        return (yield)
    except UnsupportedByBackend as exc:
        policy = get_policy()
        if policy.on_unsupported == "fail":
            raise
        if policy.on_unsupported == "skip":
            pytest.skip(str(exc))
        pytest.xfail(str(exc))


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[Any]) -> Any:
    report = yield
    backend = _backend_of(item)
    # Custom attributes: pytest reports are open to extension and the hooks that
    # read them below live in this module.
    report.mbt_backend = backend.name if backend else "-"
    report.mbt_family = _marker_str(item, "mbt_family")
    report.mbt_op = _marker_str(item, "mbt_op")
    report.mbt_dtype = _dtype_of(item)
    excinfo = getattr(call, "excinfo", None)
    report.mbt_unsupported = bool(excinfo is not None and _is_unsupported(excinfo.value))
    return report


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    outcome = _outcome_of(report)
    if outcome is None:
        return
    recorder = _recorder()
    if recorder is None:
        return
    recorder.add(
        Result(
            nodeid=report.nodeid,
            backend=getattr(report, "mbt_backend", "-"),
            outcome=outcome,
            duration=float(getattr(report, "duration", 0.0)),
            family=getattr(report, "mbt_family", ""),
            op=getattr(report, "mbt_op", ""),
            dtype=getattr(report, "mbt_dtype", ""),
            unsupported=bool(getattr(report, "mbt_unsupported", False)),
            message=_first_line(report),
        )
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    config = session.config
    state = config.stash.get(STATE, None)
    if state is None:
        return
    path = config.getoption("mbt_report", default=None)
    if path:
        state["recorder"].write_json(path)


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    state = config.stash.get(STATE, None)
    if state is None or not state.get("summary", True):
        return
    recorder: Recorder = state["recorder"]
    if not recorder.results:
        return

    rows = []
    for name, counts in sorted(recorder.by_backend().items()):
        rows.append(
            (
                name,
                *(str(counts.get(outcome, 0)) for outcome in SUMMARY_OUTCOMES),
                str(counts.get("unsupported", 0)),
                f"{recorder.duration_by_backend().get(name, 0.0):.2f}s",
            )
        )
    headers = ("backend", *SUMMARY_OUTCOMES, "unsupported", "time")

    terminalreporter.section("torch-mbt summary")
    terminalreporter.write_line(render_table(rows, headers))

    unsupported = sum(1 for result in recorder.results if result.unsupported)
    if unsupported:
        policy = state.get("on_unsupported") or get_policy().on_unsupported
        hint = "" if policy != "fail" else "; use --on-unsupported=skip to treat them as skips"
        terminalreporter.write_line(
            f"{unsupported} result(s) raised UnsupportedByBackend (policy: {policy}){hint}"
        )
    failures = [result for result in recorder.results if result.outcome == "failed"]
    if failures:
        families = sorted({result.family or "<ungrouped>" for result in failures})
        terminalreporter.write_line("failing families: " + ", ".join(families))


# --------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="session")
def backend(request: pytest.FixtureRequest) -> Backend:
    """The backend under test, parametrised over ``--backend``."""
    value = getattr(request, "param", None)
    if value is None:
        raise pytest.UsageError("the `backend` fixture requires --backend selection")
    backend = as_backend(value)
    if not backend.is_available():
        pytest.skip(f"backend {backend.name!r} is unavailable on this machine")
    return backend


@pytest.fixture(scope="session")
def device_index(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption("device_index"))


@pytest.fixture(scope="session")
def device(backend: Backend, device_index: int) -> torch.device:
    """The device created from :func:`Backend.make_device`."""
    return backend.make_device(device_index)


@pytest.fixture(scope="session")
def mbt(backend: Backend, device: torch.device) -> MbtContext:
    """Backend + device + policy, plus small deterministic-tensor helpers."""
    return MbtContext(backend=backend, device=device, policy=get_policy())


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """The reference device, for tests that always want a CPU golden."""
    return torch.device("cpu")


@pytest.fixture(autouse=True)
def _mbt_reseed(request: pytest.FixtureRequest) -> None:
    """Reseed before every test so a failure reproduces in isolation."""
    state = request.config.stash.get(STATE, None)
    seed = DEFAULT_SEED if state is None else state["seed"]
    backend = _backend_of(request.node)
    seed_all(seed, backend.device_type if backend else None)


# --------------------------------------------------------------------------- helpers


def _recorder() -> Recorder | None:
    for config in _live_configs():
        state = config.stash.get(STATE, None)
        if state is not None:
            return state["recorder"]
    return None


def _live_configs() -> list[pytest.Config]:
    # ``pytest_runtest_logreport`` receives no config object; the plugin keeps
    # the active config on the module while the session runs.
    return [_ACTIVE_CONFIG] if _ACTIVE_CONFIG is not None else []


_ACTIVE_CONFIG: pytest.Config | None = None


def pytest_unconfigure(config: pytest.Config) -> None:
    global _ACTIVE_CONFIG
    if _ACTIVE_CONFIG is config:
        _ACTIVE_CONFIG = None


def _backend_of(node: pytest.Item) -> Backend | None:
    callspec = getattr(node, "callspec", None)
    if callspec is None:
        return None
    value = callspec.params.get("backend")
    if isinstance(value, Backend):
        return value
    if isinstance(value, type) and issubclass(value, Backend):
        return value()
    return None


def _marker_names(item: pytest.Item, name: str) -> set[str]:
    marker = item.get_closest_marker(name)
    if marker is None:
        return set()
    names: set[str] = set()
    for arg in marker.args:
        if isinstance(arg, str):
            names.add(arg.lower())
        elif isinstance(arg, (list, tuple, set)):
            names.update(str(entry).lower() for entry in arg)
    return names


def _marker_str(item: pytest.Item, name: str) -> str:
    marker = item.get_closest_marker(name)
    if marker is None or not marker.args:
        return ""
    return str(marker.args[0])


def _dtype_of(item: pytest.Item) -> str:
    callspec = getattr(item, "callspec", None)
    if callspec is None:
        return ""
    dtype = callspec.params.get("dtype")
    if dtype is None and len(callspec.params) == 1:
        dtype = next(iter(callspec.params.values()))
    if isinstance(dtype, torch.dtype):
        return str(dtype).removeprefix("torch.")
    return "" if dtype is None else str(dtype)


def _is_unsupported(exc: BaseException) -> bool:
    from .errors import UnsupportedByBackend, is_unsupported_error

    return isinstance(exc, UnsupportedByBackend) or is_unsupported_error(exc)


def _outcome_of(report: pytest.TestReport) -> str | None:
    if report.when == "call":
        if hasattr(report, "wasxfail"):
            return "xpassed" if report.outcome == "passed" else "xfailed"
        return report.outcome
    if report.when == "setup" and report.outcome == "skipped":
        return "skipped"
    if report.when in ("setup", "teardown") and report.outcome == "failed":
        return "error"
    return None


def _first_line(report: pytest.TestReport) -> str:
    if report.outcome == "passed":
        return ""
    longrepr = getattr(report, "longrepr", None)
    if longrepr is None:
        return ""
    text = str(longrepr).strip().splitlines()
    return text[-1][:400] if text else ""
