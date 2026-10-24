# Complete signed-pair read rotation. Every channel remains; no approximation.
# The power-of-two balance uses deterministic <=32-row operand samples only.
_ASTRA_SPR_PARENT_READ = _RTT_PARENT_READ
_ASTRA_SPR_STATS = dict(calls=0, shapes=[], scales=[], cohorts=[])

def _astra_spr_rotate(a, b, cohorts):
    m, k, n = int(a.shape[0]), int(a.shape[1]), int(b.shape[1])
    bounds, halves, expected = [], [], 0
    for offset, half in cohorts:
        if offset != expected or half <= 0:
            raise RuntimeError('pair rotation needs contiguous complete cohorts')
        expected += 2 * half
        bounds.extend((offset + half, expected))
        halves.append(half)
    if expected != k:
        raise RuntimeError('pair rotation incomplete young roster')
    aa = fnp.split(a, bounds[:-1], axis=1)
    bb = fnp.split(b, bounds[:-1], axis=0)
    ap, am = fnp.concatenate(aa[::2], axis=1), fnp.concatenate(aa[1::2], axis=1)
    bp, bm = fnp.concatenate(bb[::2], axis=0), fnp.concatenate(bb[1::2], axis=0)
    u0, u1 = ap + am, ap - am
    v0, v1 = (bp + bm) * 0.5, (bp - bm) * 0.5
    del aa, bb, ap, am, bp, bm
    su0, su1 = u0[::max(1, (m+31)//32)], u1[::max(1, (m+31)//32)]
    sv0, sv1 = v0[:, ::max(1, (n+31)//32)], v1[:, ::max(1, (n+31)//32)]
    norms = fnp.stack((fnp.sum(su0*su0), fnp.sum(su1*su1),
                       fnp.sum(sv0*sv0), fnp.sum(sv1*sv1))).tolist()
    denominator, numerator = float(norms[0])*float(norms[3]), float(norms[1])*float(norms[2])
    exponent = int(round(0.25 * _rc_math.log2(numerator/denominator))) if min(numerator,denominator)>0 else 0
    exponent = max(-16, min(16, exponent))
    scale = 2.0**exponent
    fnp.multiply(u0, scale, out=u0)
    fnp.multiply(v0, 1.0/scale, out=v0)
    half_bounds, at = [], 0
    for half in halves[:-1]:
        at += half
        half_bounds.append(at)
    up, um = fnp.split(u0, half_bounds, axis=1), fnp.split(u1, half_bounds, axis=1)
    vp, vm = fnp.split(v0, half_bounds, axis=0), fnp.split(v1, half_bounds, axis=0)
    aout = fnp.concatenate(tuple(x for pair in zip(up,um) for x in pair), axis=1)
    bout = fnp.concatenate(tuple(x for pair in zip(vp,vm) for x in pair), axis=0)
    return aout, bout, scale

def _astra_spr_read(left, young_right, lam, live_indices, state):
    old = int(state['columns'])
    m, k, n = int(left.shape[0]), int(young_right.shape[1]), int(young_right.shape[0])
    if not k:
        return _ASTRA_SPR_PARENT_READ(left, young_right, lam, live_indices, state)
    cohorts=[]
    for start, width in state['cohorts']:
        start, width = int(start), int(width)
        if start < old:
            if start+width>old:
                raise RuntimeError('pair rotation splits old cohort')
            continue
        if width % 2 or start+width>old+k:
            raise RuntimeError('pair rotation invalid young cohort')
        if width:
            cohorts.append((start-old,width//2))
    young = left[:,old:]
    if m<n:
        a, b = young*lam[old:], young_right.T
    else:
        a, b = young, (young_right*lam[old:]).T
    aa, bb, scale = _astra_spr_rotate(a,b,cohorts)
    replacement = fnp.concatenate((left[:,:old],aa),axis=1) if old else aa
    ones = fnp.ones((k,),dtype=fnp.float32)
    coefficients = fnp.concatenate((lam[:old],ones)) if old else ones
    _ASTRA_SPR_STATS['calls'] += 1
    _ASTRA_SPR_STATS['shapes'].append((m,k,n,old))
    _ASTRA_SPR_STATS['scales'].append(scale)
    _ASTRA_SPR_STATS['cohorts'].append(len(cohorts))
    return _ASTRA_SPR_PARENT_READ(replacement,bb.T,coefficients,live_indices,state)

_RTT_PARENT_READ = _astra_spr_read
_ASTRA_SPR_PARENT_PREDICT = Estimator.predict
def _astra_spr_predict(self,*args,**kwargs):
    _ASTRA_SPR_STATS.clear()
    _ASTRA_SPR_STATS.update(calls=0,shapes=[],scales=[],cohorts=[])
    return _ASTRA_SPR_PARENT_PREDICT(self,*args,**kwargs)
Estimator.predict = _astra_spr_predict
