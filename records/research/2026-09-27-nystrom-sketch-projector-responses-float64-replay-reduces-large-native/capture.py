from pathlib import Path
import sys,types,time,gc,ctypes,json,hashlib
import numpy as np
import flopscope as flops
HERE=Path(__file__).resolve().parent;H=HERE.parent
PREVIOUS=H/'classical_covariance_response_v1';BASE=H/'recurrent_covariance_observer_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for e in read(HERE/'seal.json')['sources']:assert pin(e['path'])==e
name=sys.argv[1];assert name in ('zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus')
prior={r['mini']:r for r in read(PREVIOUS/name/'complete.json')['rows']}
e=prior[57]['packet'];assert pin(e['path'])==e
with np.load(e['path']) as z:readout=z['readout'].copy()
out=HERE/name;out.mkdir();candidate=PREVIOUS/'candidate.py'
original=candidate.read_text();source=original
edits=[("        _slr_require(_JOIN_NYSTROM, 'Nystrom core requires Nystrom columns')",
        "        _ny_origin_input(layer,J,weight,coefficient,Qp,state,Y,_jn_rows,previous)\n        _slr_require(_JOIN_NYSTROM, 'Nystrom core requires Nystrom columns')"),
       ('        del K, unused_R, Y, Omega, _jn_rows',
        '        _ny_origin_output(layer,Qn,unused_R,K,core)\n        del K, unused_R, Y, Omega, _jn_rows')]
for a,b in edits:assert source.count(a)==1,(a,source.count(a));source=source.replace(a,b)
inverse=source
for a,b in reversed(edits):assert inverse.count(b)==1;inverse=inverse.replace(b,a)
assert inverse==original
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('nystrom_origin_'+name);mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(source,str(candidate)+':nystrom_passive_trace','exec'),mod.__dict__)
mod._CF_PAYLOAD['output.weight']=readout.tolist();mod._CF_PAYLOAD['output.bias']=[0.]
saved={}
def save(key,value):
    assert key not in saved and value is not None,key
    saved[key]=np.asarray(value).copy();assert np.isfinite(saved[key]).all(),key
def input_hook(layer,J,weight,coefficient,Qp,state,Y,rows,previous):
    if layer!=10:return
    assert not saved
    for k,v in [('J',J),('weight',weight),('coefficient',coefficient),('Qp',Qp),('core_in',state['core']),('Y',Y),('rows',rows),('previous',previous)]:save(k,v)
def output_hook(layer,Qn,R,K,core):
    if layer!=10:return
    for k,v in [('Qn',Qn),('R',R),('Ksym',K),('core_out',core)]:save(k,v)
mod._ny_origin_input=input_hook;mod._ny_origin_output=output_hook
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
assert mod._CR_ON
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t;rows=[]
for mini in (57,68):
    b=base[mini];assert pin(b['weights']['path'])==b['weights'];w=np.load(b['weights']['path'])
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=b['identity']['mlp_name'])
    started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:pred=np.asarray(est.predict(mlp,2**41),dtype='float64')
    bill=int(ctx.summary_dict()['flops_used']);wall=time.monotonic()-started
    e=prior[mini]['packet'];assert pin(e['path'])==e
    with np.load(e['path']) as z:assert pred.tobytes()==z['prediction'].tobytes() and readout.tobytes()==z['readout'].tobytes()
    assert bill==prior[mini]['flops'] and len(saved)==12
    assert mod._CF_STATS['calls']==7*(len(rows)+1)
    assert mod._CR_COUNTS==dict(full=6*(len(rows)+1),diagonal=len(rows)+1)
    assert np.array_equal(saved['weight'],saved['coefficient']*saved['coefficient'])
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,prediction=pred,readout=readout,**saved)
    assert p.stat().st_size<100_000_000
    row=dict(mini=mini,packet=pin(p),flops=bill,wall_s=wall,all16_bytes_exact=True,complete_bill_exact=True,
        shapes={k:list(v.shape) for k,v in saved.items()},final_targets_read=False)
    rows.append(row);print(json.dumps(dict(event='nystrom_traced',name=name,**row)),flush=True)
    saved.clear();del w,mlp,pred;gc.collect();relief(None,0)
once(out/'complete.json',dict(name=name,rows=rows,source_inverse_exact=True,passive=True,final_targets_read=False))
