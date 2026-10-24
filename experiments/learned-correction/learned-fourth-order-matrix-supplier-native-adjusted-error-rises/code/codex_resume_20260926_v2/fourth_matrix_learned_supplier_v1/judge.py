from common import *
import time
import resource
import numpy as np
import torch
from model import Supplier,numpy_pair,numpy_diag
from data import cell,assess
torch.set_num_threads(1);torch.set_num_interop_threads(1)
verify();frozen=read(HERE/'model_frozen.json');g=read(HERE/'fit_guard.json')
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
for name in ('model','normalizers','training'):check(frozen[name])
with np.load(frozen['model']['path'],allow_pickle=False) as z:params={k:z[k] for k in z.files}
with np.load(frozen['normalizers']['path'],allow_pickle=False) as z:norm={k:z[k] for k in z.files}
model=Supplier();model.load_state_dict({k:torch.from_numpy(v) for k,v in params.items()});model.eval()
started=time.monotonic();rows=[];external=[];replay=0.;rng=np.random.default_rng(20260927393)
with (HERE/'judged_cells.jsonl').open('x') as stream:
    for ordinal in TRAIN+VALID:
        for layer in LAYERS:
            c=cell(ordinal,layer);F=np.clip(c['F']/norm['feature_scale'],-8,8);R=c['R']/norm['rho_scale'];S=c['S']/norm['third_scale']
            with torch.no_grad():D=model.matrix(torch.from_numpy(F),torch.from_numpy(R),torch.from_numpy(S),float(norm['off_scale']),float(norm['diag_scale'])).numpy().astype(np.float64)
            assert np.isfinite(D).all() and np.array_equal(D,D.T)
            # Independent double arithmetic on64 fixed sampled entries and every diagonal.
            i=rng.integers(1024,size=64);j=(i+rng.integers(1,1024,size=64))%1024
            e=np.stack((R[i,j],R[j,i],S[i,j],S[j,i]),axis=1).astype(np.float64)
            pd={k:v.astype(np.float64) for k,v in params.items()}
            rp=numpy_pair(F[i].astype(np.float64),F[j].astype(np.float64),e,pd)*float(norm['off_scale'])
            rd=numpy_diag(F.astype(np.float64),pd)*float(norm['diag_scale'])
            error=max(float(np.max(np.abs(D[i,j]-rp)))/float(norm['off_scale']),float(np.max(np.abs(np.diag(D)-rd)))/float(norm['diag_scale']))
            assert error<=2e-5,error;replay=max(replay,error)
            metrics,direct=assess(c,D)
            row=dict(ordinal=ordinal,layer=layer,role='train' if ordinal in TRAIN else 'excluded_reused',metrics=metrics,
                     finite_consumer_max_abs=direct,independent_model_normalized_max_abs=error)
            rows.append(row);external+=c['external_pins'];stream.write(json.dumps(row)+'\n');stream.flush()
            print(json.dumps(dict(event='judged',ordinal=ordinal,layer=layer,ratios={k:v['ratio'] for k,v in metrics.items()},elapsed_s=time.monotonic()-started)),flush=True)
names=tuple(rows[0]['metrics'])
energies={str(layer):{name:sum(r['metrics'][name]['baseline_energy'] for r in rows if r['ordinal'] in TRAIN and r['layer']==layer)/6 for name in names} for layer in LAYERS}
networks=[]
for ordinal in TRAIN+VALID:
    rec=dict(ordinal=ordinal,role='train' if ordinal in TRAIN else 'excluded_reused',ratios={})
    rr=[r for r in rows if r['ordinal']==ordinal]
    for name in names:
        num=sum(r['metrics'][name]['residual_energy']/energies[str(r['layer'])][name] for r in rr)
        den=sum(r['metrics'][name]['baseline_energy']/energies[str(r['layer'])][name] for r in rr)
        rec['ratios'][name]=num/den
    networks.append(rec)
primary=('joint_faces','covariance','next_variance_mean','next_total_mean')
gates={str(r['ordinal']):{name:r['ratios'][name]<=.80 for name in primary} for r in networks if r['ordinal'] in VALID}
result=dict(passed=all(all(v.values()) for v in gates.values()),gates=gates,networks=networks,cells=rows,
            train_layer_baseline_energies=energies,independent_model_normalized_max_abs=replay,external_inputs=external,
            model_frozen=pin(HERE/'model_frozen.json'),model=frozen['model'],normalizers=frozen['normalizers'],
            seal_sha256=sha(HERE/'seal.json'),cell_log_sha256=sha(HERE/'judged_cells.jsonl'),
            original_parent='331684',current_S31_native_candidate=False,valid40_accessed=False,new_samples=False,
            elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
once(HERE/'result.json',result);print(json.dumps(dict(event='complete',passed=result['passed'],gates=gates,networks=networks)),flush=True)
