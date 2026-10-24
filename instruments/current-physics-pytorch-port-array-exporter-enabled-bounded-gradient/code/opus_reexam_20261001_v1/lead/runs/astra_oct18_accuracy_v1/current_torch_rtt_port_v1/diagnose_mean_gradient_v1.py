"""Saved TRAIN8 mean-gradient coherence and untouched CHECK header availability only."""
from pathlib import Path
import hashlib,json,math,zipfile
import numpy as np
import torch
P=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1');R=P.parents[1];O=R.parents[1];D=P/'train8_selection_diagnosis_v1'
torch.set_num_threads(1);torch.set_num_interop_threads(1);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((P/'offline_train8_v2/GRADIENT_MANIFEST_v1.json').read_text());basepath=O/'trainer/cert/RAD/S56/point_v1/params.pt';assert sha(basepath)==m['base_params_sha256'];base=torch.load(basepath,map_location='cpu',weights_only=True)['arrays']
r=[]
for x in m['rows']:
 assert sha(x['result'])==x['result_sha256'];r.append(torch.load(x['result'],map_location='cpu',weights_only=False))
keys=sorted(r[0]['grads']);ESCAL={'e.cov_factor','e.third_scale'};groups={mode:{g:np.zeros((8,8)) for g in ('arrp','escal')} for mode in ('raw','rms2')};arrays={}
def stats(K):
 n=len(K);off=K.copy();np.fill_diagonal(off,0);held=off.sum(1)/(n-1);U=float(held.mean());u_minus=[]
 for i in range(n):
  sel=[j for j in range(n) if j!=i];v=off[np.ix_(sel,sel)];u_minus.append(float(v.sum()/((n-1)*(n-2))))
 jse=float(np.sqrt((n-1)/n*np.square(np.asarray(u_minus)-np.mean(u_minus)).sum()))
 plug=float(K.sum()/n**2);noise=float((np.trace(K)-K.sum()/n)/((n-1)*n))
 return dict(unbiased_cross_network_inner_product=U,jackknife_SE=jse,heldout_benefits=held.tolist(),heldout_mean=U,heldout_naive_SE=float(held.std(ddof=1)/math.sqrt(n)),heldout_positive=int((held>0).sum()),plugin_mean_squared_norm=plug,estimated_mean_noise=noise,signal_fraction=U/plug if plug else None,identity_error=U-(plug-noise))
for key in keys:
 G=torch.stack([x['grads'][key].double().reshape(-1) for x in r]);K=(G@G.T).numpy();rms=float(base[key].double().square().mean().sqrt());group='escal' if key in ESCAL else 'arrp'
 arrays[key]=dict(group=group,rms=rms,raw=stats(K),rms2=stats(K*rms*rms))
 groups['raw'][group]+=K;groups['rms2'][group]+=K*rms*rms
out=dict(status='SAVED_TRAIN8_UNTHRESHOLDED_MEAN_ONLY',definition=dict(raw='d=-mean_gradient(other7)',rms2='d=-array_RMS**2*mean_gradient(other7); natural Euclidean descent in per-array relative coordinates',benefit='positive means descent: -held_gradient dot d',unbiased='mean pairwise cross-network dot product, excludes self pairs',SE='U-statistic delete-one jackknife; naive heldout SE also shown, folds overlap'),groups={mode:{g:stats(K) for g,K in block.items()} for mode,block in groups.items()},arrays=arrays,no_coordinate_threshold=True,no_optimizer=True,no_check_outcomes=True)
with (D/'MEAN_COHERENCE_v1.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
inv=[]
for seed in range(7103480,7103488):
 path=O/f'analyst/fresh/truth_seed{seed}.npz';row=dict(seed=seed,path=str(path),exists=path.exists())
 if path.exists():
  with zipfile.ZipFile(path) as z:
   heads={}
   for name in ('N.npy','mc_post.npy','mc_post_var.npy'):
    if name not in z.namelist():heads[name]=None;continue
    with z.open(name) as f:
     version=np.lib.format.read_magic(f);reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
     shape,fortran,dtype=reader(f);heads[name]=dict(shape=list(shape),dtype=str(dtype))
     if name=='N.npy':assert not dtype.hasobject;row['N']=int(np.frombuffer(f.read(dtype.itemsize),dtype=dtype)[0])
  row.update(headers=heads,eligible=row.get('N',0)>=7995392 and heads['mc_post.npy']['shape']==[16,1024] and heads['mc_post_var.npy']['shape']==[16,1024])
 inv.append(row)
with (D/'NEXT_CHECK_HEADERS_v1.json').open('x') as f:json.dump(dict(rows=inv,outcomes_read=False,disjoint_previous_check=True),f,indent=2)
print(json.dumps(dict(groups=out['groups'],perarray=[dict(key=k,raw_U=v['raw']['unbiased_cross_network_inner_product'],raw_SE=v['raw']['jackknife_SE'],rms2_U=v['rms2']['unbiased_cross_network_inner_product'],positive=v['raw']['heldout_positive']) for k,v in arrays.items()],next_check_eligible=sum(x.get('eligible',False) for x in inv))))
