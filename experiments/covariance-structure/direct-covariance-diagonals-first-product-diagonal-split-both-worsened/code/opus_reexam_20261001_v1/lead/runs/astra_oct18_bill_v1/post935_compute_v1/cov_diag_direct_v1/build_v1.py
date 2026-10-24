import ast,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
R=D.parents[2]
P=R/'astra_oct18_accuracy_v1/setup_memo_diet_v1/candidate_compact10_capacity35_memodiet_HELD_v1.py'
s=P.read_text(); expected='95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf'
assert hashlib.sha256(P.read_bytes()).hexdigest()==expected
nodes={n.name:n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
lines=s.splitlines(keepends=True)
functions=[]
for name in ('_flms_cov_star','_dc_cov_star'):
    node=nodes[name];f=''.join(lines[node.lineno-1:node.end_lineno])
    body=f.splitlines(keepends=True);out=[];count=0
    for line in body:
        out.append(line)
        if line.lstrip().startswith('first = _mm('):
            indent=line[:len(line)-len(line.lstrip())]
            out.append(indent+'_o18_covdiag = _o18_direct_covdiag('+('w' if name=='_flms_cov_star' else 'wa')+', first, hold)\n')
            count+=1
        if line.strip()=='Cpre = cov.astype(fnp.float64)':
            out.append('    if _o18_covdiag is not None:\n        fnp.fill_diagonal(Cpre, _o18_covdiag)\n        fnp.fill_diagonal(cov, _o18_covdiag)\n')
    assert count==(1 if name=='_flms_cov_star' else 3)
    functions.append(''.join(out))
header='''# Direct hidden covariance diagonal, fixed layers1..14 only.
# _PHYS_LAYER holds the PREVIOUS layer while covariance is pushed.
_O18_CDIAG_STATS = dict(calls=0, layers=[], retired=True)
def _o18_direct_covdiag(w, first, hold):
    previous = int(_PHYS_LAYER[0])
    if not 0 <= previous <= 13:
        return None
    # Keep the parent's float32 factors; accumulate their elementwise product
    # in float64. Compute before any later kernel can reuse scratch storage.
    value = fnp.sum(hold(w * first), axis=0, dtype=fnp.float64)
    _O18_CDIAG_STATS['calls'] += 1
    _O18_CDIAG_STATS['layers'].append(previous + 1)
    return value

'''
footer='''
_ASTRA_DENSE_CONTEXT = tuple((name, globals()[name])
                            for name, _value in _ASTRA_DENSE_CONTEXT)
_O18_CDIAG_PARENT_PREDICT = Estimator.predict
def _o18_cdiag_predict(self, *args, **kwargs):
    _O18_CDIAG_STATS.update(calls=0, layers=[], retired=False)
    try:
        return _O18_CDIAG_PARENT_PREDICT(self, *args, **kwargs)
    finally:
        _O18_CDIAG_STATS['retired'] = True
Estimator.predict = _o18_cdiag_predict
'''
fragment=header+'\n\n'.join(functions)+'\n'+footer
candidate=D/'candidate_95c_direct_covdiag_HELD_v1.py'
combined=s+'\n'+fragment;ast.parse(combined)
with(D/'fragment_v1.py').open('x')as f:f.write(fragment)
with candidate.open('x')as f:f.write(combined)
bill_upper=14*(1024*1024+2*1024*1023+3*1024)
r=dict(status='HELD_FIXED_HIDDEN_COVARIANCE_DIAGONAL_SCREEN',parent=str(P),parent_sha256=expected,candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    scope='L1..14 only via previous-layer PHYS_LAYER0..13; L0 Gram and L15 direct variance unchanged.',
    source_change='Clone last active flms/dc covariance functions; immediately after first=C32@w compute sum(float32(w*first),axis0,dtypefloat64); replace Cpre64 and retainedcov32 diagonals only. Star, offdiagonal and all other code retained.',
    arithmetic='Float32 products with float64 accumulation; first-product rounding is retained.',
    price_upper_flops=bill_upper,price_upper_percent_rank35=100*bill_upper/238836882388,
    nominal_added_calls_per_layer=4,nominal_added_calls_per_network=56,extra_numeric_temporary_bytes_upper=4*1024*1024+8*1024,
    proof_requirements=['Tiny original-vs-new branches and input ownership','Actual14layers, finite outputs, no rescue, strictguards on fixed35/200','No accuracy or residual eligibility claim before native measurements'])
with(D/'BUILD_v1.json').open('x')as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps({k:r[k]for k in ('candidate_sha256','price_upper_flops','price_upper_percent_rank35','nominal_added_calls_per_network')}))
