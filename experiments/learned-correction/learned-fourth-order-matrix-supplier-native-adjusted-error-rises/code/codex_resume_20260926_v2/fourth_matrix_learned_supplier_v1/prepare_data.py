from common import *
import time
import resource
import numpy as np
from data import cell
verify();u=read(HERE/'unit_result.json');g=read(HERE/'unit_guard.json')
assert u['passed'] and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
rng=np.random.default_rng(20260927391);started=time.monotonic();F=[];ii=[];jj=[];edges=[];targets=[];diags=[]
fs=np.zeros(140);rs=ss=off=diag=0.;rows=[];external=[];tri=np.triu_indices(1024,1)
for ordinal in TRAIN:
    for layer in LAYERS:
        c=cell(ordinal,layer);D=c['projected_D'];base=len(F)*1024;f=c['F'];R=c['R'];S=c['S']
        choose=rng.choice(len(tri[0]),16384,replace=False);i=tri[0][choose];j=tri[1][choose]
        ii.append((i+base).astype(np.int64));jj.append((j+base).astype(np.int64));F.append(f)
        edges.append(np.stack((R[i,j],R[j,i],S[i,j],S[j,i]),axis=1))
        targets.append(D[i,j].astype(np.float32));diags.append(np.diag(D).astype(np.float32))
        fs+=np.mean(f.astype(np.float64)**2,axis=0)
        rs+=(float(np.sum(R.astype(np.float64)**2))-float(np.sum(np.diag(R).astype(np.float64)**2)))/(1024*1023)
        ss+=(float(np.sum(S.astype(np.float64)**2))-float(np.sum(np.diag(S).astype(np.float64)**2)))/(1024*1023)
        dv=float(np.sum(np.diag(D)**2));off+=(float(np.sum(D*D))-dv)/(1024*1023);diag+=dv/1024
        rows.append(dict(ordinal=ordinal,layer=layer,node_offset=base,pairs=16384,projection=c['projection_info']))
        external+=c['external_pins']
        print(json.dumps(dict(event='prepared',ordinal=ordinal,layer=layer,elapsed_s=time.monotonic()-started)),flush=True)
normalizers=dict(feature_scale=np.maximum(np.sqrt(fs/42),1e-5).astype(np.float32),
                 rho_scale=np.array(max(np.sqrt(rs/42),1e-5),np.float32),third_scale=np.array(max(np.sqrt(ss/42),1e-5),np.float32),
                 off_scale=np.array(np.sqrt(off/42),np.float32),diag_scale=np.array(np.sqrt(diag/42),np.float32))
assert all(np.isfinite(v).all() and np.min(v)>0 for v in normalizers.values())
with (HERE/'normalizers.npz').open('xb') as out:np.savez(out,**normalizers)
f=np.clip(np.concatenate(F)/normalizers['feature_scale'],-8,8)
e=np.concatenate(edges);e[:,:2]/=normalizers['rho_scale'];e[:,2:]/=normalizers['third_scale']
dataset=dict(F=f,i=np.concatenate(ii),j=np.concatenate(jj),edges=e,
             target=np.concatenate(targets)/normalizers['off_scale'],diagonal=np.concatenate(diags)/normalizers['diag_scale'])
with (HERE/'training_data.npz').open('xb') as out:np.savez(out,**dataset)
assert dataset['F'].shape==(43008,140) and dataset['target'].shape==(688128,)
result=dict(completed=True,rows=rows,dataset=pin(HERE/'training_data.npz'),normalizers=pin(HERE/'normalizers.npz'),
            external_inputs=external,validation_labels_accessed=False,seal_sha256=sha(HERE/'seal.json'),elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
once(HERE/'prepared.json',result);print(json.dumps(dict(event='complete',bytes=result['dataset']['bytes'],elapsed_s=result['elapsed_s'])),flush=True)
