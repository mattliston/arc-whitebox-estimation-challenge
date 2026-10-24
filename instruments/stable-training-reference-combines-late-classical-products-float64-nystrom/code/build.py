from pathlib import Path
import ast,json,hashlib,datetime
HERE=Path(__file__).resolve().parent;H=HERE.parent
PARENT=H/'classical_covariance_response_v1';TRACE=H/'current_nystrom_origin_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2,allow_nan=False)+'\n')
candidate=PARENT/'candidate.py';assert pin(candidate)==read(PARENT/'seal.json')['candidate']
original=candidate.read_text();lines=original.splitlines(keepends=True)
nodes=[n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push'
    and '_slr_require(_JOIN_NYSTROM' in ''.join(lines[n.lineno-1:n.end_lineno])]
assert len(nodes)==1;n=nodes[0];oldbody=''.join(lines[n.lineno-1:n.end_lineno])
block="""            Y = _shared_old384_mm(J, _e2_H(_e2_H(weight[:, None]) * _e2_H(_e2_H(J[_jn_rows, :]).T)))
            if old_columns:
                Y = Y + _e2_H(_shared_old384_mm(Qp, _e2_H(_shared_old384_mm(state['core'], _e2_H(_e2_H(Qp[_jn_rows, :]).T)))))
"""
assert oldbody.count(block)==1
replacement="""            if _SR_ON and layer >= 9:
                Y = _sr_wide_sketch(J,weight,Qp,state['core'] if old_columns else None,_jn_rows)
            else:
"""+''.join('    '+line for line in block.splitlines(keepends=True))
qline="        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')\n"
qreplacement=qline+"        if _SR_ON and layer >= 9:\n            Qn = Qn.astype(fnp.float32)\n"
assert oldbody.count(qline)==1
newbody=oldbody.replace(block,replacement).replace(qline,qreplacement)
assert newbody.replace(qreplacement,qline).replace(replacement,block)==oldbody
source=''.join(lines[:n.lineno-1])+newbody+''.join(lines[n.end_lineno:])
footer='''

# Fixed late numerical training reference.
_SR_ON = True
_SR_STATS = dict(late_dispatch=0, wide_sketch=0)
def _sr_wide_sketch(J,weight,Qp,core,rows):
    X=J.astype(fnp.float64)
    Y=X@(weight.astype(fnp.float64)[:,None]*X[rows].T)
    if core is not None:
        Q=Qp.astype(fnp.float64)
        Y=Y+Q@(core.astype(fnp.float64)@Q[rows].T)
    _SR_STATS['wide_sketch']+=1
    return Y
_SR_DISPATCH = _cr_dispatch
def _cr_dispatch(l,w,C,A):
    global _WINOGRAD
    if _SR_ON and l>=9:
        _WINOGRAD=False
        _SR_STATS['late_dispatch']+=1
    return _SR_DISPATCH(l,w,C,A)
_SR_PREDICT = Estimator.predict
def _sr_predict(self,*args,**kwargs):
    global _WINOGRAD
    previous_switch=_WINOGRAD
    _WINOGRAD=True
    try:
        return _SR_PREDICT(self,*args,**kwargs)
    finally:
        _WINOGRAD=previous_switch
Estimator.predict=_sr_predict
'''
source+=footer
inverse=source[:-len(footer)]
assert inverse.count(newbody)==1;inverse=inverse.replace(newbody,oldbody)
assert inverse==original;ast.parse(source)
once(HERE/'candidate.py',source)
paths=[HERE/n for n in ('PROTOCOL.md','build.py','prepare.py','run.py','assess.py','launch.py','candidate.py')]
paths += [candidate,PARENT/'seal.json',PARENT/'result.json',TRACE/'analysis.json',TRACE/'artifact_manifest.json',
    H/'native_innovation_readout_response_v1/basis.npz',H/'recurrent_covariance_observer_v1/capture/complete.json',H/'continuation_checkpoint_v50.json']
for name in ('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus'):
    paths += [TRACE/name/'complete.json',TRACE/(name+'_guard.json'),PARENT/name/'complete.json']
for name in ('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus','d7_plus','d7_minus','d7_halfplus','d7_halfminus'):
    paths.append(H/'native_innovation_readout_response_v1'/name/'complete.json')
once(HERE/'seal.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=[pin(p) for p in paths],
    source_inverse_exact=True,parent=pin(candidate),candidate=pin(HERE/'candidate.py'),training_reference_only=True))
print(json.dumps(dict(candidate=pin(HERE/'candidate.py'),source_inverse_exact=True)),flush=True)
