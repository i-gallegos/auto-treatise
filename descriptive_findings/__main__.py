"""Unified CLI: ``python -m descriptive_findings --doctrine <d>``."""

from __future__ import annotations

import sys

from descriptive_findings.analyze import main

if __name__ == "__main__":
    sys.exit(main())
