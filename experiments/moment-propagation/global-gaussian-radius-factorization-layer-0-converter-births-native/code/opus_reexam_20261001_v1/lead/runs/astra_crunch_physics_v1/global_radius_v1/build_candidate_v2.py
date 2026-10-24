"""Freeze a literal L0 trace-ownership repair; no execution of the estimator."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
os.nice(10)
import ast,hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
parent=H/'candidate_global_radius_v1.py'
src=parent.read_text()
assert hashlib.sha256(src.encode()).hexdigest()=='4e7749ea80410646c0a7ff7c83c231e7e6cc7267fe476f06a6df5cf4b4147122'
edits=[]
def change(old,new):
    global src
    if old.startswith('            if _BFIX_GAMMA:'):
        anchor=src.index('            b = _gr_l0_bsource(aux,host)')
        assert src[anchor:].count(old)==1
        src=src[:anchor]+src[anchor:].replace(old,new,1)
    else:
        assert src.count(old)==1,(src.count(old),old[:80])
        src=src.replace(old,new)
    edits.append(dict(old=old,new=new))
change('            if _BFIX_GAMMA:\n                b = b + _e2_H(_e2_H(_e2_H(_BFIX_GAMMA * _e2_H(radial_new - radial)) / 6) * m)\n            if _S19_BSCALE is not None:   # PT-33 S19: carried b x bscale after the b-fix, before trace_post (covhunter; torch b_post)\n                b = b * _S19_BSCALE\n            selected, births = select(_e2_gate_min, gate_floor)',
       '            if l != 0 and _BFIX_GAMMA:\n                b = b + _e2_H(_e2_H(_e2_H(_BFIX_GAMMA * _e2_H(radial_new - radial)) / 6) * m)\n            if l != 0 and _S19_BSCALE is not None:   # L0 owns exact physical trace separately\n                b = b * _S19_BSCALE\n            selected, births = select(_e2_gate_min, gate_floor)')
change('    m,cov,pk21,pk22,r2,r3,r4=_gr_convert_first(cov,pk21,pk22,mg,r2g,r3g,r4g)',
       '    gr_gaussian_radial=_gr_original_gaussian_radial(cov,pk21,pk22,mg,r2g,r3g,r4g,metric)\n    m,cov,pk21,pk22,r2,r3,r4=_gr_convert_first(cov,pk21,pk22,mg,r2g,r3g,r4g)')
change('               gr_gaussian_pdf=pdfg,gr_input_radial=_GR_INPUT_RADIAL,',
       '               gr_gaussian_pdf=pdfg,gr_gaussian_radial=gr_gaussian_radial,gr_input_radial=_GR_INPUT_RADIAL,')
change('def _gr_l0_bsource(aux, host):\n    # Exact trace coefficient, retaining the frozen source-strength dial.\n    return host*(-6.0/((_GR_N+1.0)*_S19_BSRC_DIV))',
       'def _gr_l0_bsource(aux, host):\n    # Exact physical spherical trace plus mapped frozen Gaussian b-fix.\n    # T_s=T_g/c3-sym(C_s,m_s)/(n+1)+k_m*m_s**3. The physical\n    # term is not subject to Gaussian-calibrated source or b-fix dials.\n    optional = (_S19_BSCALE * _BFIX_GAMMA * aux[\'gr_gaussian_radial\'] * _GR_N) / 6.0\n    return host*((optional-1.0)/(_GR_N+1.0))')
fragment='''

# Reconstruct the frozen Gaussian radial scalar from the existing L0 planes.
# No network is rerun and no wide product is introduced. This is exactly the
# original Gaussian post22 operation order, including marginal restoration.
def _gr_original_gaussian_radial(cov, pk21, pk22, mg, r2g, r3g, r4g, metric):
    mg32 = _ps32(mg)
    post22 = pk22 - 2 * mg32[:, None] * pk21.T - 2 * mg32[None, :] * pk21 + 4 * fnp.outer(mg32, mg32) * cov - 2 * cov * cov
    var = r2g - mg*mg
    k4 = r4g - 4*mg*r3g + 6*mg*mg*r2g - 3*mg**4 - 3*var*var
    post22 = post22.astype(fnp.float64)
    fnp.fill_diagonal(post22,k4)
    cov64 = cov.astype(fnp.float64)
    fnp.fill_diagonal(cov64,var)
    den = fnp.trace(cov64)**2 + 2*fnp.sum(cov64*cov64) if metric == 'covariance' else _GR_N*(_GR_N+2)
    return 3*fnp.sum(post22)/den
'''
src+=fragment
ast.parse(src)
out=H/'candidate_global_radius_v2.py';assert not out.exists();out.write_text(src)
(H/'fragment_trace_ownership_v2.py.txt').write_text(fragment)
receipt=dict(parent=str(parent),parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),candidate=str(out),candidate_sha256=hashlib.sha256(src.encode()).hexdigest(),edits=edits,fragment=fragment,
    scope='Only L0 trace ownership. All L1+ b propagation, L0 beta/birth/K4L/NP/F4, final restoration, and frozen learned parameters remain unchanged.',
    caveat='The exact raw-moment identity also has a tiny rank-one mean-cube remainder; this is proven and explicitly not implemented in this trace-only repair. The existing approximate third-tensor bank remains approximate.',
    diagnostic='TRACE_IDENTITY_v2.json; failed first tiny guard was a proof-bookkeeping sign error, corrected before candidate generation.')
(H/'BUILD_CANDIDATE_v2.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k not in ('edits','fragment')}))
