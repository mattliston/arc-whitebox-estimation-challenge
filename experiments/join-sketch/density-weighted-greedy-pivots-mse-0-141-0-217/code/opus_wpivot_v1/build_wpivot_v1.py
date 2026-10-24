"""Importance-weighted greedy pivots on 2b108+R256: rank by w*d_res, w from phi(a_{L-1}) of the join's own rows (previous layer's
standardized pre-activation means, captured from the _deadcut_push call). Pre-registered proxies P1 (pure pdf) / P2 (0.5 + 0.5 pdf)."""
from pathlib import Path
import hashlib
D=Path(__file__).resolve().parent;R=D.parent
base=(R/'opus_gbfix_v1/cand_2b108_frag_r256_v1.py').read_text();fr=(R/'opus_gbfix_v1/fragment_join_greedy_r256_2b108lineage_v1.py.txt').read_text()
assert hashlib.sha256(fr.encode()).hexdigest().startswith('0c795483') and fr in base
lines=fr.splitlines(keepends=True)
a=next(i for i,l in enumerate(lines) if l.startswith('def _ojx_greedy('));b=next(i for i in range(a+1,len(lines)) if lines[i][:1] not in (' ','\n'))
g=''.join(lines[a:b])
o1="        P = fnp.sort(hold(fnp.argsort(hold(-dres), kind='stable'))[:b])\n";assert g.count(o1)==1
g=g.replace(o1,"        P = fnp.sort(hold(fnp.argsort(hold(-(dres if w is None else hold(dres * w))), kind='stable'))[:b])\n")
o2="            P2 = fnp.sort(hold(fnp.argsort(hold(-dres), kind='stable'))[:rest])\n";assert g.count(o2)==1
g=g.replace(o2,"            P2 = fnp.sort(hold(fnp.argsort(hold(-(dres if w is None else hold(dres * w))), kind='stable'))[:rest])\n")
o3="    dres = hold(d.astype(fnp.float64))\n";assert g.count(o3)==1
g=g.replace(o3,"    dres = hold(d.astype(fnp.float64))\n    w = _wp_weight(hold, int(d.shape[0]))\n")
for name,form in (('P1','w = hold(hold(e / 1.0) + 1e-6)'),('P2','w = hold(hold(e * 0.5) + 0.5)')):
    frag=f'''

# BEGIN OPUS-WPIVOT FRAGMENT {name}_v1 (append LAST after the R256 fragment): pivot ranking by w * d_res, w from phi(a) of the join rows
_WP_A = {{}}
_WP_CUR = [None, None]
_WP_STATS = dict(weighted=0, plain=0)
_WP_PARENT_DEADCUT = _deadcut_push


def _deadcut_push(layer, w32, p, previous, newborn, lam, state, a):
    _WP_A[layer] = a
    return _WP_PARENT_DEADCUT(layer, w32, p, previous, newborn, lam, state, a)


_WP_PARENT_IMPL = _astra_bcc_push_impl


def _astra_bcc_push_impl(layer, w32, p, previous, *args, **kwargs):
    _WP_CUR[0], _WP_CUR[1] = layer, previous
    try:
        return _WP_PARENT_IMPL(layer, w32, p, previous, *args, **kwargs)
    finally:
        _WP_CUR[0], _WP_CUR[1] = None, None


def _wp_weight(hold, n):
    layer, previous = _WP_CUR
    a = _WP_A.get(layer - 1) if layer is not None else None
    if a is None or previous is None or int(previous.shape[0]) != n:
        _WP_STATS['plain'] += 1
        return None
    ap = hold(hold(a[previous]).astype(fnp.float64))
    e = hold(fnp.exp(hold(hold(ap * ap) * -0.5)))
    {form}
    _WP_STATS['weighted'] += 1
    return w


{g}
_WP_PARENT_PREDICT = Estimator.predict


def _wp_predict(self, *args, **kwargs):
    _WP_A.clear()
    try:
        return _WP_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _WP_A.clear()
Estimator.predict = _wp_predict
# END OPUS-WPIVOT FRAGMENT {name}_v1
'''
    o=D/f'cand_2b108_r256_wpivot{name}_v1.py';o.open('x').write(base+frag);compile(o.read_text(),str(o),'exec');print(name,hashlib.sha256(o.read_bytes()).hexdigest()[:12])
