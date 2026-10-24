from pathlib import Path
import sys,types,time,gc,ctypes,hashlib,json
import numpy as np
import flopscope as flops
HERE=Path(__file__).resolve().parent;H2=HERE.parent;P2=H2.parent
PROBE=H2/'native_innovation_readout_response_v1';FIT=H2/'native_covariance_innovation_fit_v1'
BASE=H2/'recurrent_covariance_observer_v1';MINIS=(57,68)
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p).resolve();return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for e in read(HERE/'trace_seal.json')['sources']:assert pin(Path(e['path']))==e
name=sys.argv[1];assert name in ('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus')
old=read(PROBE/name/'complete.json');base={x['mini']:x for x in read(BASE/'capture/complete.json')['rows']}
previous={x['mini']:x for x in old['rows']}
first=previous[57]['packet'];assert pin(first['path'])==first
with np.load(first['path']) as z:readout=z['readout'].copy()
out=HERE/name;out.mkdir();candidate=FIT/'candidate.py'
assert pin(candidate)==read(FIT/'result.json')['candidate']
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('response_origin_'+name);mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
original=candidate.read_text();source=original
edits=[('Cpre, _bb_star_pre = _flms_cov_star(w32, C, star)','Cpre, _bb_star_pre = _origin_cov_star(l,w32,C,star)'),
    ('    Q = fnp.linalg.qr(A32[:, cols])[0]','    _origin_columns(A32,cols)\n    Q = fnp.linalg.qr(A32[:, cols])[0]'),
    ('            _PHYS_LAYER[0] = l','            _origin_pre(l,mu,Cpre,K21,k4,cp_pre,star_pre,bpre,live_indices,response_indices)\n            _PHYS_LAYER[0] = l'),
    ('            previous, previous_alive = (live_indices, alive)','            _origin_post(l,b,radial_new,birth,u,delta,t,s,selected,alive)\n            previous, previous_alive = (live_indices, alive)')]
for a,b in edits:assert source.count(a)==1,(a,source.count(a));source=source.replace(a,b)
inverse=source
for a,b in reversed(edits):assert inverse.count(b)==1;inverse=inverse.replace(b,a)
assert inverse==original
exec(compile(source,str(candidate)+':passive_origin','exec'),mod.__dict__)
mod._CF_PAYLOAD['output.weight']=readout.tolist();mod._CF_PAYLOAD['output.bias']=[0.]
saved={};active=[False];counts={}
def save(key,value):
    assert key not in saved,key
    saved[key]=np.asarray(value).copy()
    assert np.isfinite(saved[key]).all(),key
oldcov=mod._flms_cov_star;oldlr=mod._lr_factor;oldnonlinear=mod.nonlinear
oldstep=mod.gnn_step;oldfeedback=mod._cf_feedback
def cov_star(l,w,C,A):
    assert not active[0];active[0]=l==9
    if active[0]:save('C8',C);save('star8',A)
    result=oldcov(w,C,A)
    if active[0]:save('Cpre9',result[0]);save('star_pre9',result[1])
    active[0]=False;return result
def columns(A,cols):
    if active[0]:save('star_columns9',cols)
def factor(A):
    result=oldlr(A)
    if active[0]:save('star_factor_Q9',result[0]);save('star_factor_B9',result[1])
    return result
def pre(l,mu,C,K,k4,cp,star,b,live,response):
    if l!=9:return
    for key,value in [('mu9',mu),('K21_9',K),('k4_9',k4),('CPpre9',cp),('bpre9',b),('live9',live),('response9',response)]:save(key,value)
def post(l,b,radial,birth,u,delta,t,s,selected,alive):
    if l!=8:return
    for key,value in [('b8',b),('radial8',radial),('birth8',birth),('u8',u),('delta8',delta),('t8',t),('s8',s),('selected8',selected),('alive8',alive)]:save(key,value)
def nonlinear(*args,**kwargs):
    result=oldnonlinear(*args,**kwargs)
    if mod._PHYS_LAYER[0]==9:save('physical_mean9',result[0])
    return result
def step(l,x,sigma,w,state,P):
    if l==9:save('features9',x);save('previous_hidden8',state);save('sigma9',sigma)
    result=oldstep(l,x,sigma,w,state,P)
    if l==9:save('mean_delta9',result[0]);save('hidden9',result[1])
    return result
def feedback(l,m,C):
    if l==9:save('post_mean9',m)
    return oldfeedback(l,m,C)
mod._origin_cov_star=cov_star;mod._origin_columns=columns;mod._origin_pre=pre;mod._origin_post=post
mod._lr_factor=factor;mod.nonlinear=nonlinear;mod.gnn_step=step;mod._cf_feedback=feedback
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t;rows=[]
for mini in MINIS:
    assert not saved and not active[0];b=base[mini];assert pin(b['weights']['path'])==b['weights']
    w=np.load(b['weights']['path']);mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=b['identity']['mlp_name'])
    started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:pred=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
    bill=int(ctx.summary_dict()['flops_used']);oldrow=previous[mini];entry=oldrow['packet'];assert pin(entry['path'])==entry
    with np.load(entry['path']) as z:assert pred.tobytes()==z['prediction'].tobytes() and readout.tobytes()==z['readout'].tobytes()
    assert bill==oldrow['flops'],(mini,bill,oldrow['flops'])
    assert len(saved)==30,(len(saved),list(saved))
    assert mod._CF_STATS['calls']==7*(len(rows)+1)
    assert saved['post_mean9'].shape==(1024,)
    np.testing.assert_array_equal(saved['physical_mean9']+saved['mean_delta9'].astype('float64'),saved['post_mean9'])
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,prediction=pred,readout=readout,**saved)
    row=dict(mini=mini,packet=pin(p),flops=bill,wall_s=time.monotonic()-started,
        bytes=p.stat().st_size,all16_bytes_exact=True,complete_bill_exact=True,final_targets_read=False)
    rows.append(row);print(json.dumps(dict(event='traced',name=name,**row)),flush=True)
    saved.clear();del w,mlp,pred;gc.collect();relief(None,0)
once(out/'complete.json',dict(name=name,rows=rows,source_inverse_exact=True,passive=True,final_targets_read=False))
