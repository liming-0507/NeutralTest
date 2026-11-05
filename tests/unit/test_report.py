"""Result recording and the JSON report."""

from __future__ import annotations

import json

from neutraltest.report import Recorder, Result, render_table


def _recorder() -> Recorder:
    recorder = Recorder()
    recorder.add(
        Result(
            nodeid="tests/test_ops.py::test_add[cpu-fp32]",
            backend="cpu",
            outcome="passed",
            duration=0.5,
            family="elementwise",
            op="add",
            dtype="float32",
        )
    )
    recorder.add(
        Result(
            nodeid="tests/test_ops.py::test_add[xpu-fp32]",
            backend="xpu",
            outcome="failed",
            duration=0.2,
            family="elementwise",
            op="add",
            dtype="float32",
            message="values differ",
        )
    )
    recorder.add(
        Result(
            nodeid="tests/test_ops.py::test_add[xpu-fp16]",
            backend="xpu",
            outcome="skipped",
            duration=0.0,
            family="elementwise",
            op="add",
            dtype="float16",
        )
    )
    recorder.add(
        Result(
            nodeid="tests/test_conv.py::test_conv[cpu-fp32]",
            backend="cpu",
            outcome="failed",
            duration=1.5,
            family="convolution",
            op="conv2d",
            dtype="float32",
            unsupported=True,
        )
    )
    return recorder


def test_totals_and_per_backend_counts() -> None:
    recorder = _recorder()
    assert recorder.totals() == {"passed": 1, "failed": 2, "skipped": 1}
    by_backend = recorder.by_backend()
    assert by_backend["cpu"]["passed"] == 1
    assert by_backend["cpu"]["failed"] == 1
    assert by_backend["xpu"]["failed"] == 1
    assert by_backend["xpu"]["skipped"] == 1


def test_unsupported_results_are_counted_separately() -> None:
    counts = _recorder().by_backend()
    assert counts["cpu"]["unsupported"] == 1
    assert "unsupported" not in counts["xpu"]


def test_family_grouping_falls_back_for_unlabelled_results() -> None:
    recorder = Recorder()
    recorder.add(Result(nodeid="x::y", backend="cpu", outcome="passed"))
    assert "<ungrouped>" in recorder.by_family()


def test_durations_are_summed_per_backend() -> None:
    durations = _recorder().duration_by_backend()
    assert durations["cpu"] == 2.0
    assert durations["xpu"] == 0.2


def test_json_report_round_trips(tmp_path) -> None:
    target = tmp_path / "reports" / "nt.json"
    written = _recorder().write_json(target)
    assert written == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["totals"] == {"passed": 1, "failed": 2, "skipped": 1}
    assert payload["backends"]["cpu"]["unsupported"] == 1
    assert payload["duration_seconds"] == 2.2
    assert len(payload["results"]) == 4
    assert payload["results"][1]["message"] == "values differ"
    assert payload["families"]["elementwise"]["failed"] == 1


def test_rendered_table_is_fixed_width_and_aligned() -> None:
    text = render_table(
        [("cpu", "3", "0"), ("verylongname", "12", "1")], ("backend", "passed", "failed")
    )
    lines = text.splitlines()
    assert lines[0].startswith("backend")
    assert set(lines[1]) == {"-", " "}
    assert lines[2].startswith("cpu")
    assert lines[3].startswith("verylongname")
    assert all(len(line.rstrip()) <= len(lines[0].rstrip()) for line in lines[2:])


def test_empty_recorder_reports_nothing() -> None:
    recorder = Recorder()
    assert recorder.totals() == {}
    assert recorder.as_dict()["duration_seconds"] == 0
