"""E5 -- resource ceilings (registers, shared memory, allocation peaks).

These are properties of a device kernel and of a profiler; a CPU run can only
report the dispatch it saw. Disclosed rather than faked.
"""

from __future__ import annotations

from ..context import RunContext

ID = "E5"
OPS = None


def run(ctx: RunContext) -> None:
    ctx.na("no register/shared-memory accounting on this backend; needs a device profiler")
