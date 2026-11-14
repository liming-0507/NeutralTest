#!/usr/bin/env python3
"""Fail if anything names an accelerator vendor, or a vendor's stack, in the tree.

This project ships without vendor code and without vendor names: a backend is
whatever registers a device against PyTorch's own extension points, and an
SDK release is described by a JSON profile that is never edited by the project.
The rule is easier to keep when a machine checks it.

Usage::

    tools/no_vendor_names.py            # scan the tracked-ish tree
    tools/no_vendor_names.py --list     # also print the pattern table

Exit status: 0 clean, 1 matches found, 2 usage error.

The scan is deliberately about *names*, not behaviour: it must stay fast enough
to run in CI on every commit and easy to explain to a contributor.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

#: ``(label, regex)``.  Case-insensitive; word boundaries so "input" and
#: "manual" do not trip the ``npu``/``mlu`` entries.
PATTERNS: tuple[tuple[str, str], ...] = (
    ("vendor", r"\bcambricon\b"),
    ("stack", r"\bmlu\b|\bneuware\b|\bbangc\b|\bcnnl\b|\bcnrt\b|\bcncl\b"),
    # ``xpu`` is deliberately absent: it is PyTorch's own generic device type
    # (``torch.xpu``), not any one vendor's name, and the registry names it.
    ("vendor", r"\bkunlun\b|\bkunlunxin\b|\bsuiren\b"),
    ("vendor", r"\bmetax\b|\bmuxi\b|\bmthreads\b|\bmoore\s*threads\b"),
    ("vendor", r"\bascend\b|\bnpu\b|\btorch_npu\b"),
    ("adapter", r"\btorch_mlu\b"),
)

#: Files whose whole job is to spell the patterns out.  They are exempt, and
#: nothing else is: an exemption list that grows is a rule that has stopped
#: being enforced.
EXEMPT = ("tools/no_vendor_names.py",)

#: Never scanned: dependencies, build artefacts, caches.
SKIP_DIRS = {
    ".git",
    ".venv",
    ".venv-matrix",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "dist",
    "src/neutraltest.egg-info",
    "node_modules",
}

#: Text suffixes that are worth scanning.
SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".toml",
    ".cfg",
    ".ini",
    ".yml",
    ".yaml",
    ".sh",
    ".txt",
    ".rst",
}


def _tracked(root: Path) -> list[Path]:
    """Files git would see, so a scan matches what a reviewer reads."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        files = [root / name for name in out.split("\0") if name]
    except (OSError, subprocess.CalledProcessError):
        files = [path for path in root.rglob("*") if path.is_file()]
    return files


def scan(root: Path) -> list[tuple[Path, int, str, str]]:
    compiled = [(label, re.compile(regex, re.IGNORECASE)) for label, regex in PATTERNS]
    hits: list[tuple[Path, int, str, str]] = []
    for path in _tracked(root):
        if not path.is_file() or path.suffix not in SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if relative.as_posix() in EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            for label, regex in compiled:
                match = regex.search(line)
                if match:
                    hits.append((relative, number, label, match.group(0)))
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".", help="tree to scan (default: .)")
    parser.add_argument("--list", action="store_true", help="print the pattern table too")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2

    if args.list:
        for label, regex in PATTERNS:
            print(f"{label:10} {regex}")

    hits = scan(root)
    for path, number, label, text in hits:
        print(f"{path}:{number}: [{label}] {text}")
    if hits:
        print(f"\n{len(hits)} vendor reference(s) found", file=sys.stderr)
        return 1
    print("no vendor names found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
