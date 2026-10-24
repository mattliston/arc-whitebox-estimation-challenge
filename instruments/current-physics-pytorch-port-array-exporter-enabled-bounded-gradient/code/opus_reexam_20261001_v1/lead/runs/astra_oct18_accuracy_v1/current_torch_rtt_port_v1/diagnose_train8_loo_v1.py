"""Saved TRAIN8 gradients only; never reads CHECK outcomes or runs an estimator."""
from pathlib import Path
import hashlib,json,math,zipfile
import numpy as np
import torch
P=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1');R=P.parents[1];O=R.parents[1];D=P/'train8_selection_diagnosis_v1';D.mkdir()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((P/'offline_train8_v2/GRADIENT_MANIFEST_v1.json').read_text());assert m['status']=='ALL_EIGHT_GRADIENTS_COMPLETE_AND_RETIRED'
basepath=O/'trainer/cert/RAD/S56/point_v1/params.pt';assert sha(basepath)==m['base_params_sha256'];base=torch.load(basepath,map_location='cpu',weights_only=True)['arrays']
point=json.loads((R/'astra_oct18_lead_v1/current_array_graft_v1/point_train8_larger_v1/FIXED_STEP_v1.json').read_text());mse_nc=point['training_noise_corrected_mse']
records=[]
for x in m['rows']:
 assert sha(x['result'])==x['result_sha256']
 z=torch.load(x['result'],map_location='cpu',weights_only=False);assert z['rank']==x['seed'] and z['mode']=='grad' and z['scale']==1e8;records.append(z)
keys=sorted(records[0]['grads']);assert all(sorted(x['grads'])==keys for x in records)
ESCAL={'e.cov_factor','e.third_scale'}
folds={g:np.zeros(8,dtype=np.float64) for g in ('arrp','escal')};full={g:dict(uncorrected=0.,corrected=0.,moving=0) for g in folds};arrays={}
for key in keys:
 group='escal' if key in ESCAL else 'arrp';G=torch.stack([x['grads'][key].double().reshape(-1) for x in records]);rms=float(base[key].double().square().mean().sqrt())
 held=[];counts=[]
 for out in range(8):
  train=G[[i for i in range(8) if i!=out]];mean=train.mean(0);se=train.std(0)/math.sqrt(7);tv=torch.where(se>0,mean/se.clamp_min(1e-300),torch.zeros_like(mean));keep=tv.abs()>2.5
  direction=-torch.sign(mean)*keep*rms
  benefit=-float((G[out]*direction).sum());held.append(benefit);counts.append(int(keep.sum()));folds[group][out]+=benefit
 mean=G.mean(0);se=G.std(0)/math.sqrt(8);tv=torch.where(se>0,mean/se.clamp_min(1e-300),torch.zeros_like(mean));keep=tv.abs()>2.5
 unc=float((mean.abs()*keep).sum())*rms;corrected=float((mean.abs()*(1-1/tv.square().clamp_min(1e-12))*keep).sum())*rms
 full[group]['uncorrected']+=unc;full[group]['corrected']+=corrected;full[group]['moving']+=int(keep.sum())
 v=np.asarray(held);arrays[key]=dict(group=group,elements=G.shape[1],rms=rms,full8_selected=int(keep.sum()),full8_first_order_uncorrected=unc,full8_first_order_corrected=corrected,loo_benefits=v.tolist(),loo_mean=float(v.mean()),loo_SE=float(v.std(ddof=1)/math.sqrt(8)),loo_positive=int((v>0).sum()),loo_selected_counts=counts)
groups={}
for name,v in folds.items():
 size=point['group_sizes'][name]
 groups[name]=dict(full8=full[name],loo_benefits=v.tolist(),loo_mean=float(v.mean()),loo_SE=float(v.std(ddof=1)/math.sqrt(8)),loo_positive=int((v>0).sum()),same_fixed_larger_step=size,loo_predicted_gain_pct=float(v.mean()*size/(mse_nc*1e8)*100),loo_predicted_gain_SE_pct=float(v.std(ddof=1)/math.sqrt(8)*size/(mse_nc*1e8)*100),optimistic_corrected_gain_pct=full[name]['corrected']*size/(mse_nc*1e8)*100)
result=dict(status='SAVED_TRAIN8_LOO_DIAGNOSIS_ONLY',train=[x['rank'] for x in records],threshold=2.5,definition='Positive benefit=-held_gradient dot direction; each direction uses other7 only, direction=-sign(mean)*I(abs(t)>2.5)*arrayRMS. No fitting or CHECK use.',scaled_gradient=1e8,groups=groups,arrays=arrays,training_noise_corrected_mse=mse_nc,baseline_manifest_sha256=sha(P/'offline_train8_v2/GRADIENT_MANIFEST_v1.json'),parameters_updated=False)
with (D/'LOO_v1.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
# Only NPZ headers and scalar sample count; never load mc_post or mc_post_var data.
inv=[]
for seed in range(7100276,7100300):
 path=O/f'analyst/fresh/truth_seed{seed}.npz';row=dict(seed=seed,path=str(path),exists=path.exists())
 if path.exists():
  with zipfile.ZipFile(path) as z:
   heads={}
   for name in ('N.npy','mc_post.npy','mc_post_var.npy'):
    if name not in z.namelist():heads[name]=None;continue
    with z.open(name) as f:
     version=np.lib.format.read_magic(f);reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
     shape,fortran,dtype=reader(f);heads[name]=dict(shape=list(shape),dtype=str(dtype),fortran=fortran)
     if name=='N.npy':
      assert not dtype.hasobject and math.prod(shape or (1,))==1
      row['N']=int(np.frombuffer(f.read(dtype.itemsize),dtype=dtype)[0])
  row.update(headers=heads,file_bytes=path.stat().st_size,eligible=(row.get('N',0)>=7995392 and heads.get('mc_post.npy',{}).get('shape')==[16,1024] and heads.get('mc_post_var.npy',{}).get('shape')==[16,1024]))
 inv.append(row)
with (D/'TRAIN_CANDIDATE_HEADERS_v1.json').open('x') as f:json.dump(dict(rows=inv,outcomes_read=False,unused_status='Metadata inventory only; not a proof of no historical training use.',check_excluded=True),f,indent=2)
print(json.dumps(dict(groups=groups,eligible=sum(x.get('eligible',False) for x in inv),arrays_ranked=sorted([dict(key=k,loo_mean=v['loo_mean'],loo_SE=v['loo_SE'],positive=v['loo_positive'],full=v['full8_first_order_corrected']) for k,v in arrays.items()],key=lambda x:-abs(x['loo_mean']))[:8])))
