"""[opus-kinksel] (a) OTT row order by DOWNSTREAM KINK IMPORTANCE at L12-14, on S2tr + JORD (c0458f0e). OTT keeps `kept` rows of the
old tier exact (count from its own score = _OTT_Z - a, dead rows a + _OTT_NEG_Z, >= 0, rounded up to 64: unchanged here, bill-neutral)
and approximates the rest by the first-chaos trace; here the kept rows are chosen by score + beta t_j / mean t instead of score,
t_j = sum_{i kink at layer l+1} W_{l+1}[j, i]^2. The next layer's kink set is not known yet, so it uses a cheap PROXY:
a_{l+1} ~ (W' m) / sqrt((W o W)' v) with m, v the Gaussian ReLU post mean / variance from this layer's mu, sigma (covariances
ignored), kink = |a_{l+1}| < c (c = 1.0). W_{l+1} comes from the network's weights (stashed at predict entry). ~5 M flops a layer.
Arms: ottk_b1 (beta 1), ottk_b2 (beta 2). One text edit in the LAST _astra_bcc_recon_impl + a block appended LAST."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_grow_v1/cand_s2tr_grow_jord_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('c0458f0e')
i0 = src.rfind('\ndef _astra_bcc_recon_impl(')
i1 = src.find('\ndef ', i0 + 1)
OLD = "    ordered = fnp.argsort(-score, kind='stable')\n"
NEW = """    _okt = _OK_T[0] if _OK_ON else None   # [opus-kinksel] (a): same kept count, kink-importance order
    if _okt is not None and _okt[0] == _UVP_CUR[0]:
        _okv = _okt[1] if ctx['rows'] is None else _okt[1][ctx['rows']]
        score = score + _OK_BETA * _okv / fnp.mean(_okv)
""" + OLD
body = src[i0:i1]; assert body.count(OLD) == 1
j = i0 + body.index(OLD)
src2 = src[:j] + NEW + src[j + len(OLD):]
BLOCK = '''

# BEGIN [opus-kinksel] (a) OTT KINK ORDER helpers (append LAST)
_OK_ON = True
_OK_LAYERS = (12, 13, 14)
_OK_C = 1.0
_OK_W = [None]
_OK_T = [None]
_OK_PARENT_PUSH = _shared_old384_push
_OK_PARENT_PREDICT = Estimator.predict


def _ok_predict(self, *args, **kwargs):
    mlp = args[0] if args else kwargs.get('mlp')
    _OK_W[0] = getattr(mlp, 'weights', None)
    try:
        return _OK_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _OK_W[0] = None
        _OK_T[0] = None


Estimator.predict = _ok_predict


def _shared_old384_push(layer, w32, p, previous, newborn, lam, state, rows=None):
    _OK_T[0] = None
    weights = _OK_W[0]
    if _OK_ON and layer in _OK_LAYERS and weights is not None and layer + 1 < len(weights):
        hold = _E2Hold()
        mu, s, a, g = _E2_LAYER['mu'], _E2_LAYER['s'], _E2_LAYER['a'], _E2_LAYER['g']
        pdf = hold(hold(fnp.exp(hold(hold(a * a) * -0.5))) * 0.3989422804014327)
        m = hold(hold(mu * g) + hold(s * pdf))
        q = hold(hold(hold(hold(mu * mu) + hold(s * s)) * g) + hold(hold(mu * s) * pdf))
        v = hold(fnp.maximum(hold(q - hold(m * m)), 0.0))
        Wn = hold(fnp.asarray(weights[layer + 1], dtype=fnp.float32))
        W2 = hold(Wn * Wn)
        mun = hold(hold(Wn.T) @ hold(m.astype(fnp.float32)))
        varn = hold(hold(W2.T) @ hold(v.astype(fnp.float32)))
        an = hold(mun / hold(fnp.sqrt(hold(fnp.maximum(varn, 1e-30)))))
        kink = hold(fnp.nonzero(hold(fnp.abs(an) < _OK_C))[0])
        if int(kink.shape[0]) > 0:
            _OK_T[0] = (layer, hold(fnp.sum(hold(W2[:, kink]), axis=1)).astype(fnp.float64))
    try:
        return _OK_PARENT_PUSH(layer, w32, p, previous, newborn, lam, state, rows=rows)
    finally:
        _OK_T[0] = None
# END [opus-kinksel] (a) OTT KINK ORDER helpers
'''
for tag, beta in (('ottk_b1', 1.0), ('ottk_b2', 2.0)):
    out = src2 + BLOCK + '\n_OK_BETA = %r   # [opus-kinksel] ARM %s\n' % (beta, tag)
    o = D/f'cand_s2trj_{tag}_v1.py'
    with o.open('x') as f: f.write(out)
    compile(out, str(o), 'exec'); print(o.name, hashlib.sha256(out.encode()).hexdigest()[:8])
