"""Regenerate docs/operator-matrix.md from the conformance case lists."""

from __future__ import annotations

import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFORMANCE = ROOT / "src" / "torch_mbt" / "conformance"
OUT = ROOT / "docs" / "operator-matrix.md"

SHORT = {"float32": "f32", "float64": "f64", "float16": "f16", "bfloat16": "bf16"}


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))

    rows: list[tuple[str, str, str, str, str, str, str]] = []
    params = 0
    for path in sorted(CONFORMANCE.glob("test_*.py")):
        module = importlib.import_module(f"torch_mbt.conformance.{path.stem}")
        for case in module.CASES:
            params += len(case.dtypes)
            dtypes = "/".join(SHORT.get(str(d).removeprefix("torch."), str(d)) for d in case.dtypes)
            rows.append(
                (
                    case.family,
                    case.id,
                    case.op,
                    dtypes,
                    case.kind,
                    "yes" if case.grad else "",
                    path.name,
                )
            )

    rows.sort()
    lines = [
        "# Operator matrix",
        "",
        f"Every case below is parametrised over its dtypes, so {len(rows)} cases become "
        f"{params} tests.",
        "",
        "`kind` selects the tolerance budget (see [writing-tests.md](writing-tests.md#tolerances));",
        "`bwd` marks cases that also compare input gradients against CPU `float64`.",
        "",
    ]
    family = None
    for fam, cid, op, dtypes, kind, grad, module in rows:
        if fam != family:
            lines += [
                "",
                f"## {fam}",
                "",
                "| case | op | dtypes | kind | bwd | module |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
            family = fam
        lines.append(
            f"| `{cid}` | `{op}` | {dtypes} | {kind} | {grad} | `torch_mbt/conformance/{module}` |"
        )

    gradients = sum(1 for row in rows if row[5])
    lines += ["", f"**Total: {len(rows)} cases**, {gradients} of which check gradients.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(rows)} cases, {params} tests)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
