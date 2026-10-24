#!/usr/bin/env python3
"""Target-free diagnostic wrapper with a non-graded 240 s wall allowance.

This exists only to measure how far a frozen closure that timed out at the
official 120 s cap is from completion.  A receipt produced through this
wrapper is diagnostic evidence and can never qualify a Phase 2 submission.
The FLOP budget, residual-time cap, process isolation, and RSS sampling remain
identical to the official meter driver.
"""

from __future__ import annotations

import run_synthetic_meter as meter


DIAGNOSTIC_WALL_TIME_LIMIT_S = 240.0


def main() -> int:
    meter.WALL_TIME_LIMIT_S = DIAGNOSTIC_WALL_TIME_LIMIT_S
    return meter.main()


if __name__ == "__main__":
    raise SystemExit(main())
