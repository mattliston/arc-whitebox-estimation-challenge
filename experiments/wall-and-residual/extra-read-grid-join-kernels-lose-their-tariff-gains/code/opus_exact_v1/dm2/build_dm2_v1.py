"""[opus-exact] dm2 v1: two byte-identical data-movement diets as asserted single-occurrence edits of LIVE defs in a base file:
(1) Smirnov young read: the operand that the read scales by lam is written by that multiply straight into its padded-grid slot
    (out=), and the kernel skips its copy when it is handed that slot itself (one m*k or k*n copy and one request less per call);
(2) _nonlinear_pair_batch: the three (n, n) factors are applied as three multiplies on the series' unstacked slices instead of
    being stacked first (one 3 n^2 stack less, one request more per call).
usage: build_dm2_v1.py <base.py> <base sha prefix> <out.py>"""
import hashlib, sys
from pathlib import Path
base = Path(sys.argv[1]).read_text(); assert hashlib.sha256(base.encode()).hexdigest().startswith(sys.argv[2])


def sub(s, old, new):
    assert s.count(old) == 1, old[:60]
    return s.replace(old, new)


s = sub(base, """        fnp.copyto(pad_dest[0], a)
        fnp.copyto(pad_dest[1], b)
""", """        if a is not pad_dest[0]:   # [opus-exact] dm2: an operand the read already wrote into its slot is not copied again
            fnp.copyto(pad_dest[0], a)
        if b is not pad_dest[1]:
            fnp.copyto(pad_dest[1], b)
""")
s = sub(s, """    a=hold(left[:,old:]);young_lam=hold(lam[old:])
    if m<n:a=hold(a*young_lam);b=hold(young_right.T)
    else:b=hold(hold(young_right*young_lam).T)
    name,L,P=plan;kernel=_ASTRA_SM40_KERNELS[name]
""", """    a=hold(left[:,old:]);young_lam=hold(lam[old:])
    name,L,P=plan;kernel=_ASTRA_SM40_KERNELS[name]
    # [opus-exact] dm2: the lam-scaled operand is written by its multiply straight into the kernel's padded slot (same values)
    _dm2_ar,_dm2_ak,_dm2_an=kernel.record['shape'];_dm2_u=1<<L;_dm2_rank=0 if old_product is None else int(old_product[0].shape[1])
    _dm2_pad=_astra_sm40_padding_views(_ASTRA_SM40_ENV,m,k,n,_dm2_rank,_rc_math.ceil(m/_dm2_ar/_dm2_u)*_dm2_u,
        _rc_math.ceil((k+_dm2_rank)/_dm2_ak/_dm2_u)*_dm2_u,_rc_math.ceil(n/_dm2_an/_dm2_u)*_dm2_u,_dm2_ar,_dm2_ak,_dm2_an)[4]
    if m<n:hold(fnp.multiply(a,young_lam,out=_dm2_pad[0]));a=_dm2_pad[0];b=hold(young_right.T)
    else:hold(fnp.multiply(young_right,young_lam,out=hold(_dm2_pad[1].T)));b=_dm2_pad[1]
""")
s = sub(s, """        cov, pk21, pk22 = fnp.unstack(_e2_H(_psp_pair_series(st, n, _e2_H)) * fnp.stack((_ps_ss32, _ps_vs32, _ps_vv32)))
""", """        _dm2_x0, _dm2_x1, _dm2_x2 = fnp.unstack(_e2_H(_psp_pair_series(st, n, _e2_H)))   # [opus-exact] dm2: no stack of the factors
        cov, pk21, pk22 = _e2_H(_dm2_x0 * _ps_ss32), _e2_H(_dm2_x1 * _ps_vs32), _e2_H(_dm2_x2 * _ps_vv32)
""")
Path(sys.argv[3]).write_text(s); compile(s, 'x', 'exec')
print(hashlib.sha256(s.encode()).hexdigest())
