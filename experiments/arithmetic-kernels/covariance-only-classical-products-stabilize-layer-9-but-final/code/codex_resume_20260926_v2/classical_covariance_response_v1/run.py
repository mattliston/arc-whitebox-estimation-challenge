from pathlib import Path
import sys,types,time,gc,ctypes,json,hashlib
import numpy as np
import flopscope as flops
HERE=Path(__file__).resolve().parent;H=HERE.parent
ORIGIN=H/'native_response_origin_v1';PROBE=H/'native_innovation_readout_response_v1'
BASE=H/'recurrent_covariance_observer_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for e in read(HERE/'seal.json')['sources']:assert pin(e['path'])==e
name=sys.argv[1];disabled=name=='disabled';parent_name='zero' if disabled else name
assert name in ('disabled','zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus','d7_plus','d7_minus','d7_halfplus','d7_halfminus')
if name!='disabled':
    g=read(HERE/'disabled_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    assert read(HERE/'disabled/complete.json')['all16_and_bills_exact']
old={r['mini']:r for r in read(PROBE/parent_name/'complete.json')['rows']}
e=old[57]['packet'];assert pin(e['path'])==e
with np.load(e['path']) as z:readout=z['readout'].copy()
with np.load(PROBE/'basis.npz') as z:directions=z['directions'].copy()
if parent_name=='zero':expected=np.zeros((1,32),dtype='float32')
else:
    direction=int(parent_name[1]);amp={'plus':.125,'minus':-.125,'halfplus':.0625,'halfminus':-.0625}[parent_name.split('_')[1]]
    expected=(amp*directions[:,direction]).astype('float32')[None,:]
assert readout.tobytes()==expected.tobytes()
out=HERE/name;out.mkdir();candidate=HERE/'candidate.py'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('classical_covariance_'+name);mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate),'exec'),mod.__dict__)
before={k:np.asarray(v,dtype='float32') for k,v in mod._CF_PAYLOAD.items()}
mod._CR_ON=not disabled;mod._CF_PAYLOAD['output.weight']=readout.tolist();mod._CF_PAYLOAD['output.bias']=[0.]
for k,v in mod._CF_PAYLOAD.items():
    if k not in ('output.weight','output.bias'):assert np.asarray(v,dtype='float32').tobytes()==before[k].tobytes()
saved={};old_dispatch=mod._cr_dispatch
def dispatch(l,w,C,A):
    if l==9:
        assert not saved
        saved['C8']=np.asarray(C).copy();saved['star8']=np.asarray(A).copy()
    result=old_dispatch(l,w,C,A)
    if l==9:
        saved['Cpre9']=np.asarray(result[0]).copy();saved['star_pre9']=np.asarray(result[1]).copy()
    return result
mod._cr_dispatch=dispatch
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
np.testing.assert_array_equal(np.asarray(mod._CF_PARAMS['output.weight']),readout)
assert np.asarray(mod._CF_PARAMS['output.bias']).item()==0
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
traces={r['mini']:r for r in read(ORIGIN/parent_name/'complete.json')['rows']} if not parent_name.startswith('d7') else None
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t;rows=[]
for mini in (57,68):
    b=base[mini];assert pin(b['weights']['path'])==b['weights'];w=np.load(b['weights']['path'])
    e=b['packet'];assert pin(e['path'])==e
    with np.load(e['path']) as z:parent=z['prediction'].copy()
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=b['identity']['mlp_name'])
    started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:pred=np.asarray(est.predict(mlp,2**41),dtype='float64')
    wall=time.monotonic()-started;bill=int(ctx.summary_dict()['flops_used'])
    assert pred.shape==(16,1024) and np.isfinite(pred).all() and np.min(pred)>=0
    assert pred[:9].tobytes()==parent[:9].tobytes()
    assert mod._CF_STATS['calls']==7*(len(rows)+1)
    assert mod._CR_COUNTS==dict(full=0 if disabled else 6*(len(rows)+1),diagonal=0 if disabled else len(rows)+1)
    if disabled:assert pred.tobytes()==parent.tobytes() and bill==old[mini]['flops']
    trace_checks=None
    if traces:
        e=traces[mini]['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:reference={k:z[k].copy() for k in ('C8','star8','Cpre9','star_pre9')}
        assert saved['C8'].tobytes()==reference['C8'].tobytes()
        assert saved['star8'].tobytes()==reference['star8'].tobytes()
        assert saved['star_pre9'].tobytes()==reference['star_pre9'].tobytes()
        if disabled:assert saved['Cpre9'].tobytes()==reference['Cpre9'].tobytes()
        else:
            W=w[9].astype('float32');first=saved['C8'].astype('float32')@W;h=512
            replay=np.empty((1024,1024),dtype='float32');top=W.T[:h]@first
            replay[:h]=top;replay[h:,:h]=top[:,h:].T;replay[h:,h:]=W.T[h:]@first[:,h:]
            assert saved['Cpre9'].tobytes()==replay.astype('float64').tobytes()
            del W,first,replay,top
        trace_checks=dict(input_covariance_exact=True,input_star_exact=True,output_star_exact=True,
            covariance_replay_exact=True)
        del reference
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,prediction=pred,readout=readout,Cpre9_diag=np.diag(saved['Cpre9']).copy())
    row=dict(mini=mini,packet=pin(p),flops=bill,parent_flops=old[mini]['flops'],bill_ratio=bill/old[mini]['flops'],
        wall_s=wall,first9_exact=True,trace_checks=trace_checks,final_targets_read=False)
    rows.append(row);print(json.dumps(dict(event='native_covariance_probe',name=name,**row)),flush=True)
    saved.clear();del w,mlp,pred,parent;gc.collect();relief(None,0)
once(out/'complete.json',dict(name=name,disabled=disabled,rows=rows,all16_and_bills_exact=disabled,
    first9_exact=True,covariance_only_replacement=True,final_targets_read=False,coefficients_fitted=False,
    replacement_counts=mod._CR_COUNTS))
