"""``python -m torch_mbt`` — same entry point as the ``torch-mbt`` script."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
