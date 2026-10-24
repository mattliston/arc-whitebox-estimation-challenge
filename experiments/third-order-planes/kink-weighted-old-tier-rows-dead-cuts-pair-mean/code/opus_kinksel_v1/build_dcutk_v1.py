"""[opus-kinksel] (b) DEAD-CUT row choice by DOWNSTREAM KINK IMPORTANCE at L12-14, on S2tr + JORD (c0458f0e). The dead cut keeps
`count` rows of layer l for the bank push (score = a + _DEADCUT_Z >= 0, rounded up to 64: unchanged here, bill-neutral); here the kept
rows are the top-count by score + beta t_j / mean t, so a just-dead row with a large downstream kink weight can displace the least
important kept row. t_j = sum_{i kink at layer l+1} W_{l+1}[j, i]^2 with the same l+1 kink PROXY as (a) (Gaussian ReLU post moments of
this layer, a_{l+1} ~ W'm / sqrt((W o W)'v), |a_{l+1}| < 1). Arms dcutk_b1 / dcutk_b2. Text edit in the LAST _astra_bcc_deadcut_impl
+ a _deadcut_push wrapper appended LAST."""
import hashlib
from pathlib import Path
D = Path(__file__).resolve().parent; R = D.parent
src = (R/'opus_grow_v1/cand_s2tr_grow_jord_v1.py').read_text(); assert hashlib.sha256(src.encode()).hexdigest().startswith('c0458f0e')
i0 = src.rfind('\ndef _astra_bcc_deadcut_impl(')
i1 = src.find('\ndef ', i0 + 1)
OLD = "    rows = fnp.argsort(-score, kind='stable')[:count]\n"
NEW = """    _dbt = _DB_T[0] if _DB_ON else None   # [opus-kinksel] (b): same count, kink-importance choice
    if _dbt is not None and _dbt[0] == layer:
        score = score + _DB_BETA * _dbt[1] / fnp.mean(_dbt[1])
""" + OLD
body = src[i0:i1]; assert body.count(OLD) == 1
j = i0 + body.index(OLD)
src2 = src[:j] + NEW + src[j + len(OLD):]
BLOCK = '''

# BEGIN [opus-kinksel] (b) DEAD-CUT KINK CHOICE helpers (append LAST)
_DB_ON = True
_DB_LAYERS = (12, 13, 14)
_DB_C = 1.0
_DB_W = [None]
_DB_T = [None]
_DB_PARENT_DEADCUT = _deadcut_push
_DB_PARENT_PREDICT = Estimator.predict


def _db_predict(self, *args, **kwargs):
    mlp = args[0] if args else kwargs.get('mlp')
    _DB_W[0] = getattr(mlp, 'weights', None)
    try:
        return _DB_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _DB_W[0] = None
        _DB_T[0] = None


Estimator.predict = _db_predict


def _deadcut_push(layer, w32, p, previous, newborn, lam, state, a):
    _DB_T[0] = None
    weights = _DB_W[0]
    if _DB_ON and layer in _DB_LAYERS and weights is not None and layer + 1 < len(weights):
        hold = _E2Hold()
        mu, s, g = _E2_LAYER['mu'], _E2_LAYER['s'], _E2_LAYER['g']
        pdf = hold(hold(fnp.exp(hold(hold(a * a) * -0.5))) * 0.3989422804014327)
        m = hold(hold(mu * g) + hold(s * pdf))
        q = hold(hold(hold(hold(mu * mu) + hold(s * s)) * g) + hold(hold(mu * s) * pdf))
        v = hold(fnp.maximum(hold(q - hold(m * m)), 0.0))
        Wn = hold(fnp.asarray(weights[layer + 1], dtype=fnp.float32))
        W2 = hold(Wn * Wn)
        mun = hold(hold(Wn.T) @ hold(m.astype(fnp.float32)))
        varn = hold(hold(W2.T) @ hold(v.astype(fnp.float32)))
        an = hold(mun / hold(fnp.sqrt(hold(fnp.maximum(varn, 1e-30)))))
        kink = hold(fnp.nonzero(hold(fnp.abs(an) < _DB_C))[0])
        if int(kink.shape[0]) > 0:
            _DB_T[0] = (layer, hold(fnp.sum(hold(W2[:, kink]), axis=1)).astype(fnp.float64))
    try:
        return _DB_PARENT_DEADCUT(layer, w32, p, previous, newborn, lam, state, a)
    finally:
        _DB_T[0] = None
# END [opus-kinksel] (b) DEAD-CUT KINK CHOICE helpers
'''
for tag, beta in (('dcutk_b1', 1.0), ('dcutk_b2', 2.0)):
    out = src2 + BLOCK + '\n_DB_BETA = %r   # [opus-kinksel] ARM %s\n' % (beta, tag)
    o = D/f'cand_s2trj_{tag}_v1.py'
    with o.open('x') as f: f.write(out)
    compile(out, str(o), 'exec'); print(o.name, hashlib.sha256(out.encode()).hexdigest()[:8])
