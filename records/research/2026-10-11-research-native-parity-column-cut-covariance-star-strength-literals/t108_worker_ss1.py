#!/usr/bin/env python3
"""TAG opus-physicist T108: the trainer's generated-seed worker v4 (trainer/worker_generated_v4.py, unchanged, run through runpy; the worker of every
physicist run today) on the PINNED otr_m.py copy (physicist/otr_frozen_bba0dea0, sha asserted), with the engineer's star drop-in v2
(engineer/a33_starsum/star_ss_torch_v2.py) under v1's module name, so a request's 'ss' -1 is the native ONE-DIRECTION star (a = A 1, P1 = W' a,
T2 = (W o W)' (A' a) / a'a, s2 = T2 P1'); every other 'ss' value behaves as v1 (v2 = v1 plus rank -1). For the sealed 7100200-231 look of
S48T = the MS point + the one-direction star vs S47e (lead 10-11 17:3xZ). Without 'ss' -1 the arithmetic is worker_generated_v4 exactly.
usage: as worker_generated_v4.py, run from O/trainer: python -B ../physicist/t108_worker_ss1.py <run dir> <worker index>"""
import hashlib
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER, FROZEN_M = HERE.parent / 'trainer', HERE / 'otr_frozen_bba0dea0'
sys.path.insert(0, str(TRAINER))
sys.path.insert(0, str(FROZEN_M))
import otr_m as _M_FROZEN
assert Path(_M_FROZEN.__file__).resolve().parent == FROZEN_M and hashlib.sha256(Path(_M_FROZEN.__file__).read_bytes()).hexdigest().startswith('bba0dea0'), _M_FROZEN.__file__
sys.path.append(str(HERE.parent / 'engineer' / 'a33_starsum'))
import star_ss_torch_v2
sys.modules['star_ss_torch'] = star_ss_torch_v2
sys.argv = [str(TRAINER / 'worker_generated_v4.py')] + sys.argv[1:]
runpy.run_path(str(TRAINER / 'worker_generated_v4.py'), run_name='__main__')
