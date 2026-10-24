from pathlib import Path
import ast,json,hashlib,datetime
HERE=Path(__file__).resolve().parent;H=HERE.parent
FIT=H/'native_covariance_innovation_fit_v1';ORIGIN=H/'native_response_origin_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n')
candidate=FIT/'candidate.py';assert pin(candidate)==read(FIT/'result.json')['candidate']
assert read(ORIGIN/'covariance_replay.json')['classical32_locally_qualified']
original=candidate.read_text();nodes=ast.parse(original).body;lines=original.splitlines()
def function(name):
    n=[n for n in nodes if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    return '\n'.join(lines[n.lineno-1:n.end_lineno])+'\n'
cov=function('_flms_cov_star').replace('def _flms_cov_star(', 'def _cr_cov_star(')
assert cov.count('first = _mm(C32, w)')==1 and cov.count('cov = _sym_upper(wt, first)')==1
cov=cov.replace('first = _mm(C32, w)','first = C32 @ w').replace('cov = _sym_upper(wt, first)','cov = _cr_sym_upper(wt, first)')
diag=function('_flms_covariance_diagonal').replace('def _flms_covariance_diagonal(', 'def _cr_cov_diagonal(')
assert diag.count('_mm(_e2_H(C.astype(fnp.float32)), w)')==1
diag=diag.replace('_mm(_e2_H(C.astype(fnp.float32)), w)','(_e2_H(C.astype(fnp.float32)) @ w)')
helper='''
_CR_ON = True
_CR_COUNTS = dict(full=0, diagonal=0)
def _cr_sym_upper(A,B):
    m=int(A.shape[0]);h=m//2
    out=fnp.empty((m,m),dtype=fnp.float32)
    top=A[:h]@B
    fnp.copyto(out[:h,:],top);fnp.copyto(out[h:,:h],top[:,h:].T)
    part=A[h:]@B[:,h:];fnp.copyto(out[h:,h:],part)
    return out
def _cr_dispatch(l,w,C,A):
    if _CR_ON and 9<=l<=14:
        _CR_COUNTS['full']+=1
        return _cr_cov_star(w,C,A)
    return _flms_cov_star(w,C,A)
def _cr_diagonal_dispatch(l,w,C):
    if _CR_ON and l==15:
        _CR_COUNTS['diagonal']+=1
        return _cr_cov_diagonal(w,C)
    return _flms_covariance_diagonal(w,C)
'''
edits=[('Cpre, _bb_star_pre = _flms_cov_star(w32, C, star)','Cpre, _bb_star_pre = _cr_dispatch(l,w32,C,star)'),
    ('v, active_v = (_flms_covariance_diagonal(w32, C) if _FULL_LOWER_MOMENTS','v, active_v = (_cr_diagonal_dispatch(l,w32,C) if _FULL_LOWER_MOMENTS')]
source=original
for a,b in edits:assert source.count(a)==1,(a,source.count(a));source=source.replace(a,b)
prefix=source;footer='\n\n# Fixed covariance-only arithmetic intervention.\n'+helper+'\n'+cov+'\n'+diag
source+=footer;inverse=source[:-len(footer)]
for a,b in reversed(edits):assert inverse.count(b)==1;inverse=inverse.replace(b,a)
assert inverse==original
ast.parse(source);once(HERE/'candidate.py',source)
sources=[HERE/n for n in ('PROTOCOL.md','build.py','run.py','assess.py','launch.py','candidate.py')]
sources += [candidate,FIT/'result.json',ORIGIN/'covariance_replay.json',ORIGIN/'trace_seal.json',
    H/'native_innovation_readout_response_v1/basis.npz',H/'native_innovation_readout_response_v1/result.json',
    H/'recurrent_covariance_observer_v1/capture/complete.json']
for name in ('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus'):
    sources += [ORIGIN/name/'complete.json',ORIGIN/(name+'_guard.json')]
for name in ('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus','d7_plus','d7_minus','d7_halfplus','d7_halfminus'):
    sources.append(H/'native_innovation_readout_response_v1'/name/'complete.json')
once(HERE/'seal.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=[pin(p) for p in sources],
    source_inverse_exact=True,only_two_call_sites_changed=True,parent=pin(candidate),candidate=pin(HERE/'candidate.py')))
print(json.dumps(dict(candidate=pin(HERE/'candidate.py'),source_inverse_exact=True)),flush=True)
