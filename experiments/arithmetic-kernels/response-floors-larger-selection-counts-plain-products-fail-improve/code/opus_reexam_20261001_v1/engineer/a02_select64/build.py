"""TAG opus_engineer a02: selection counts rounded up to 64 (real units, not zero padding) for chosen roles; births stay at 32.
usage: build.py <roles>   roles = any of l (live), r (response), f (L15 final rows), e.g. lrf -> candidate_<roles>.py
RRPI + count-asserted call-site edits + one appended select() with a granularity argument (the appended def is the last one)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from buildlib import build, RRPI, RRPI_SHA
roles = sys.argv[1]
edits = []
if 'l' in roles:
    edits.append(("            live_indices, alive = select(response_gate, gate_floor)\n",
                  "            live_indices, alive = select(response_gate, gate_floor, 64)\n", 1))
if 'r' in roles:
    edits.append(("            response_indices, response_mask = select(response_score, gate_floor)\n",
                  "            response_indices, response_mask = select(response_score, gate_floor, 64)\n", 1))
if 'f' in roles:
    edits.append(("    rows, unused_mask = select(_L15_ROWCUT_Z - z, 0.0)\n",
                  "    rows, unused_mask = select(_L15_ROWCUT_Z - z, 0.0, 64)\n", 1))
APPEND = '''
# BEGIN OPUS-ENGINEER a02 select64 (2026-10-01): = the bank's select with the kept count rounded up to a multiple of `step`
# (default 32 = the bank's rule); the live, response and final-layer row selections pass 64, births keep 32.
def select(score, floor, step=32):
    keep = []
    ge = score >= floor
    nz = fnp.nonzero(ge)
    keep.append((ge, nz))
    count = min(len(score), (int(nz[0].shape[0]) + step - 1) // step * step)
    neg = -score
    order = fnp.argsort(neg, kind='stable')
    keep.append((neg, order))
    indices = order[:count]
    mask = fnp.zeros(len(score), dtype=fnp.bool_)
    fnp.put(mask, indices, True)
    return (indices, mask)
# END OPUS-ENGINEER a02 select64
'''
build(RRPI, Path(__file__).resolve().parent/('candidate_%s.py' % roles), edits=edits, append=APPEND, expect_parent_sha=RRPI_SHA)
