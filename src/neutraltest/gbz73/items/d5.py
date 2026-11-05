"""D5 -- interface versions.

There is one API surface here and no released predecessor to diff against, so
the row is disclosed as not applicable rather than filled with a vacuous check.
"""

from __future__ import annotations

from ..context import RunContext

ID = "D5"
OPS = None


def run(ctx: RunContext) -> None:
    ctx.na("single API surface: there is no previous interface version to compare against")
