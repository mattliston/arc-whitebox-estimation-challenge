"""Target-free CPU64 derivative and native CPU/MPS neural qualification."""
import json,math,resource,time
from common import HERE,sha,read,seal_check,write,enc4_direction
seal_check('neural_sources_sealed.json')
from model import np,torch,KEYS,runtime,initialize,inputs,forward,export,retire

started=time.monotonic();runtime_info=runtime()
out=HERE/'neural_qualification';out.mkdir(exist_ok=False)
cap=read(HERE/'capture_qualification/result.json');replay=read(HERE/'fnp_replay/result.json')
for name in ('capture_qualification','fnp_replay'):
    g=read(HERE/(name+'_watch.json'));assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert replay['passed'] and replay['maximum_replay_flops']<=8_000_000_000
assert sha(cap['parent_P']['path'])==cap['parent_P']['sha256']
with np.load(cap['parent_P']['path'],allow_pickle=False) as z:P0={k:z[k].copy() for k in z.files}
records=[];gradient_records=[];driver_samples=[]
def load_capture(rec):
    assert sha(rec['path'])==rec['sha256'] and sha(rec['weights'])==rec['weights_sha256']
    with np.load(rec['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    return a,np.load(rec['weights'],allow_pickle=False)
def objective(data,p,fixed):
    _,delta,_=forward(data,p,fixed)
    q=torch.sin(.13*torch.arange(1,delta.numel()+1,dtype=delta.dtype,device=delta.device))/math.sqrt(delta.numel())
    return torch.dot(delta,q)

# Tiny n=8 is a CPU FP64 reference only, with actual captured local features
# and the first 8x8 submatrices. Eleven fixed normalized cosine directions.
a,w=load_capture(cap['records'][0])
tiny=dict(features=a['features'][:,:8].astype(np.float64),sigma=a['sigma'][:,:8].astype(np.float64),final_host=a['final_host'][:8].astype(np.float64))
tp,tf=initialize(P0,'cpu',dtype=torch.float64);td=inputs(tiny,w[:,:8,:8].astype(np.float64),'cpu',torch.float64)
value=objective(td,tp,tf);value.backward();tiny_rows=[]
for index,k in enumerate(KEYS):
    grad=tp[k].grad.detach();assert bool(torch.isfinite(grad).all()) and float(torch.linalg.vector_norm(grad))>0
    d=torch.cos(.019*torch.arange(1,grad.numel()+1,dtype=torch.float64)+.31*(index+1)).reshape(grad.shape)/math.sqrt(grad.numel())
    automatic=float(torch.sum(grad*d));assert abs(automatic)>1e-20
    base=tp[k].detach().clone();values=[]
    for sign in (1,-1):
        trial={name:x.detach() for name,x in tp.items()};trial[k]=base+sign*1e-5*d
        with torch.no_grad():values.append(float(objective(td,trial,tf)))
    finite=(values[0]-values[1])/(2e-5);relative=abs(finite-automatic)/max(abs(finite),abs(automatic),1e-30)
    tiny_rows.append(dict(parameter=k,automatic=automatic,central_difference=finite,relative=relative,gradient_norm=float(torch.linalg.vector_norm(grad))))
write(out/'tiny_gradient.json',dict(records=tiny_rows,epsilon=1e-5,passed=all(r['relative']<=1e-5 for r in tiny_rows),truth_accessed=False))
assert all(r['relative']<=1e-5 for r in tiny_rows),tiny_rows
del tp,tf,td,value,grad,d,base,trial,tiny;retire()

direction=enc4_direction(np)
for rec,rr in zip(cap['records'][:2],replay['records'][:2]):
    if rec['rank']!=35:a,w=load_capture(rec)
    assert rec['rank']==rr['rank'] and sha(rr['path'])==rr['sha256']
    with np.load(rr['path'],allow_pickle=False) as z:references={label:z[label+'_prediction'].copy() for label in ('zero','plus','minus')}
    gradients={}
    for device in ('cpu','mps'):
        data=inputs(a,w,device)
        for label,sign in [('zero',0),('plus',1),('minus',-1)]:
            proposal={k:v.copy() for k,v in P0.items()}
            if sign:proposal['enc4_w']=proposal['enc4_w']+np.float32(sign*2**-8)*direction
            p,fixed=initialize(proposal,device,False)
            assert all(export(proposal,p)[k].tobytes()==proposal[k].tobytes() for k in proposal)
            with torch.no_grad():served,delta,_=forward(data,p,fixed)
            y=served.cpu().numpy().copy();dd=delta.cpu().numpy().copy()
            assert np.isfinite(y).all() and np.isfinite(dd).all()
            diff=y.astype(np.float64)-references[label].astype(np.float64)
            row=dict(rank=rec['rank'],device=device,variant=label,max_abs=float(np.max(np.abs(diff))),rms=float(np.sqrt(np.mean(diff*diff))))
            row['passed']=row['max_abs']<=1e-6 and row['rms']<=1e-7
            records.append(row);print(json.dumps(row),flush=True)
            np.save(out/f"rank{rec['rank']}_{device}_{label}.npy",y)
            del p,fixed,served,delta,proposal,y,dd,diff
            if device=='mps':
                torch.mps.synchronize();driver_samples.append(int(torch.mps.driver_allocated_memory()));retire()
        if rec['rank']==35:
            p,fixed=initialize(P0,device,True);scalar=objective(data,p,fixed);scalar.backward()
            arrays={k:p[k].grad.detach().cpu().numpy().copy() for k in KEYS}
            assert all(np.isfinite(v).all() and np.linalg.norm(v.astype(np.float64))>0 for v in arrays.values())
            gradients[device]=arrays
            np.savez(out/f'gradient_{device}.npz',**arrays)
            del p,fixed,scalar
        del data;retire()
    if rec['rank']==35:
        c=np.concatenate([gradients['cpu'][k].ravel().astype(np.float64) for k in KEYS]);m=np.concatenate([gradients['mps'][k].ravel().astype(np.float64) for k in KEYS])
        relative=float(np.linalg.norm(c-m)/np.linalg.norm(c));assert np.linalg.norm(c)>0 and np.linalg.norm(m)>0
        gradient_records.append(dict(rank=35,cpu_norm=float(np.linalg.norm(c)),mps_norm=float(np.linalg.norm(m)),relative=relative,passed=relative<=.01,
            per_parameter=[dict(parameter=k,relative=float(np.linalg.norm(gradients['cpu'][k].astype(np.float64)-gradients['mps'][k].astype(np.float64))/np.linalg.norm(gradients['cpu'][k].astype(np.float64)))) for k in KEYS]))
        del c,m,gradients
    del a,w,references;retire()
passed=all(r['passed'] for r in records+gradient_records)
write(out/'result.json',dict(completed=True,passed=passed,runtime=runtime_info,parameter_count=44210,records=records,
    gradients=gradient_records,tiny_gradient_max_relative=max(r['relative'] for r in tiny_rows),
    sampled_driver_bytes_max=max(driver_samples),driver_samples_are_not_transient_peak=True,
    truth_accessed=False,new_physical_calls=0,source_seal_sha256=sha(HERE/'neural_sources_sealed.json'),
    elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
assert passed,'Fixed native neural qualification gates failed; no fit.'
