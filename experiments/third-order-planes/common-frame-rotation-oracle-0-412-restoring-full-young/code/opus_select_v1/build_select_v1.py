"""SEL_MODE=energy: deployable energy-ordered Nystrom rows at the joins (no index sort; free).
SEL_MODE=svd: ORACLE (unmetered numpy) frame rotation after each join so the leading r_u=256 coordinates hold the top singular
directions of the old-tier PAIR MEANS (the u/v split keeps F_u only in F[:256]); basis, factors (or lazy map/join) and core rotated."""
from pathlib import Path
import hashlib
D=Path(__file__).resolve().parent;R=D.parent
src=(R/'astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('2b10843a')
o1="            _jn_rows = fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank])\n"
n1="            _jn_rows = (_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank] if _SEL_MODE == 'energy' else fnp.sort(_e2_H(fnp.argsort(_e2_H(-_e2_H(_jn_d.astype(fnp.float64))), kind='stable'))[:rank]))\n"
i=src.rindex(o1);src=src[:i]+n1+src[i+len(o1):]
o2="""        state['rank'] = state.pop('rank_new')   # PT-60
        _slr_finish_factors(state, factors)
"""
n2="""        state['rank'] = state.pop('rank_new')   # PT-60
        if _SEL_MODE == 'svd':
            Qn, factors, core = _sel_rotate(layer, state, Qn, factors, core)
        _slr_finish_factors(state, factors)
"""
j=src.rindex(o2);assert j>i;src=src[:j]+n2+src[j+len(o2):]
frag='''

# BEGIN OPUS-SELECT switches (screen only: env SEL_MODE in {'', 'energy', 'svd'}); svd = UNMETERED ORACLE, not deployable
import os as _sel_os, json as _sel_json
_SEL_MODE = _sel_os.environ.get('SEL_MODE', '')


def _sel_rotate(layer, state, Qn, factors, core):
    import numpy as _np
    cols = int(state['columns']);blocks = _uvp_blocks(state, 0, cols)
    group = state['lazy']
    if factors is not None:
        F = _np.array(factors, copy=True).astype(_np.float64)
    else:
        mp = _np.array(group['map'], copy=True).astype(_np.float64); base = _np.array(group['base'], copy=True).astype(_np.float64)
        F = _np.concatenate([mp @ base, _np.array(group['join'], copy=True).astype(_np.float64)], axis=1)
    M = _np.concatenate([0.5 * (F[:, a:a + h] + F[:, a + h:a + 2 * h]) for a, h in blocks], axis=1)
    ev, V = _np.linalg.eigh(M @ M.T); Rot = V[:, _np.argsort(ev)[::-1]]
    ru = 256; tot = float((M * M).sum()); before = float((M[:ru] ** 2).sum()) / tot; after = float(((Rot.T @ M)[:ru] ** 2).sum()) / tot
    path = _sel_os.environ.get('SEL_STATS')
    if path:
        with open(path, 'a') as f:f.write(_sel_json.dumps(dict(layer=int(layer), rank=int(F.shape[0]), pairs=int(M.shape[1]), lazy=factors is None, Fu_energy_kept_before=before, after=after)) + '\\n')
    Qn2 = fnp.asarray(_np.ascontiguousarray((_np.array(Qn, copy=True).astype(_np.float64) @ Rot).astype(_np.float32)))
    if factors is not None:
        factors = fnp.asarray(_np.ascontiguousarray((Rot.T @ F).astype(_np.float32)))
    else:
        group['map'] = fnp.asarray(_np.ascontiguousarray((Rot.T @ mp).astype(_np.float32)))
        group['join'] = fnp.asarray(_np.ascontiguousarray((Rot.T @ _np.array(group['join'], copy=True).astype(_np.float64)).astype(_np.float32)))
    if core is not None:
        C = _np.array(core, copy=True).astype(_np.float64); core = fnp.asarray(_np.ascontiguousarray((Rot.T @ C @ Rot).astype(_np.float32)))
    return Qn2, factors, core
# END OPUS-SELECT switches
'''
o=D/'candidate_2b108_select_v1.py';o.open('x').write(src+frag);compile(o.read_text(),str(o),'exec');print(hashlib.sha256(o.read_bytes()).hexdigest())
