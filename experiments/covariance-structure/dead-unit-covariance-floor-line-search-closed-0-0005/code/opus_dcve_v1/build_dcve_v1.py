"""[opus-dcve] DCV-EARLY arms: S2tr 46e9315d + _DCV_FLOOR = 0.002 (DCV2) + a per-layer roster floor table _DCVE_FLOORS read by a
_dcv_roster wrapper (layer = max(_R320_RESP) + 1: _dc_cov_star / _dc_cov_diagonal run at the top of layer l, before the loop records
_R320_RESP[l]); layers absent from the table keep _DCV_FLOOR. Fragment appended LAST (value table + 12-line wrapper)."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_s2tr_v1/candidate_S2tr_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('46e9315d')
FRAG = '''

# BEGIN [opus-dcve] DCV-EARLY (value-only; append LAST): the dead-unit covariance roster floor by layer
_DCV_FLOOR = 0.002
_DCVE_FLOORS = {floors}
_DCVE_PARENT_ROSTER = _dcv_roster


def _dcv_roster(gate, floor):
    global _DCV_FLOOR
    f = _DCVE_FLOORS.get((max(_R320_RESP) + 1) if _R320_RESP else 0)
    if f is None:
        return _DCVE_PARENT_ROSTER(gate, floor)
    keep, _DCV_FLOOR = _DCV_FLOOR, f
    try:
        return _DCVE_PARENT_ROSTER(gate, floor)
    finally:
        _DCV_FLOOR = keep
# END [opus-dcve] DCV-EARLY
'''
for tag, v in (('dcve3', 0.003), ('dcve4', 0.004)):
    t = src + FRAG.replace('{floors}', '{' + ', '.join(f'{l}: {v}' for l in range(1, 10)) + '}')
    o = D/f'cand_s2tr_{tag}_v1.py'
    with o.open('x') as f: f.write(t)
    compile(t, str(o), 'exec'); print(o.name, hashlib.sha256(t.encode()).hexdigest()[:8])
