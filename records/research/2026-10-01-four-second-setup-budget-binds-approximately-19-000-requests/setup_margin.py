"""TAG opus-release: setup margin of candidates on the platform, from one light setup-only probe each with the guard held open
(setup_probe.sh, RELEASE_SETUP_SLOWDOWN=0; requests per key are deterministic, so machine load does not matter).

The estimator binds its tree keys most-used first until its 4.0 s guard. A normal platform worker gets through about
setup_clock.PLATFORM_REQUESTS requests in those 4.0 s (calibration in setup_clock.py), worker w-2 of #333111 managed 17,954, the
re-set-up slow worker of #333099 14,589. Printed per candidate: requests of the binding phase, the share of a normal worker's
budget that is, keys bound by a normal worker, by a w-2-like worker (x1.06 slower) and by a #333099-like worker (x1.30 slower),
requests of the complete deferred fusion and of the unguarded tail, flops of a complete setup.
usage: setup_margin.py name=candidate.py [name=candidate.py ...]   -> runs/setup/margin_<name>.json (reused if present)"""
import json
from pathlib import Path
import subprocess
import sys

import setup_clock

R = Path(__file__).resolve().parent
N = setup_clock.PLATFORM_REQUESTS
print(f"{'candidate':34s} {'keys':>5s} {'binding req':>11s} {'of normal':>9s} {'bound x1.0':>10s} {'x1.06':>6s} {'x1.30':>6s} "
      f"{'fusion req':>10s} {'tail req':>8s} {'setup flops':>15s}  note")
for item in sys.argv[1:]:
    name, candidate = item.split('=', 1)
    out = R/'runs'/'setup'/f'margin_{name}.json'
    if not out.exists():
        subprocess.run(['bash', str(R/'setup_probe.sh'), candidate, f'margin_{name}', '', '0.0'], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    if not out.exists():
        print(f'{name:34s} PROBE FAILED (see runs/setup/margin_{name}_server.log)', flush=True)
        continue
    r = json.loads(out.read_text())
    trace, phases = r['binding_trace'], {p[0]: p[3] for p in r['phases']}
    bound = lambda slow: sum(1 for x in trace if x <= N/slow)
    total = r['requests'] - r['requests_before']
    print(f"{name:34s} {r['keys_total']:5d} {r['binding_requests']:11d} {r['binding_requests']/N:9.3f} {bound(1.0):10d} "
          f"{bound(1.06):6d} {bound(1.30):6d} {phases['lvfuse'] - r['binding_requests']:10d} {total - phases['lvfuse']:8d} "
          f"{r['flops']:15d}  {'SETUP FELL BACK' if r['failed'] else ''}", flush=True)
