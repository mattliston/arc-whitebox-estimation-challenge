"""Fixed ten exact-source science identities; no model calls or parity gate."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2];SCI=O/'lead/runs/astra_prefix_row_energy22_science_v1'
SOURCE_SHA='7b1266280f75edd8e44800e6afdcc5981d9f5d27f1a8226e73cac6f1f1abc84a'
BASE_SHA='19fc42a4d64b6721a91719f87cd4e81eb3b6f6b54aa24aed92ae4a5facdd8f5f'
SEEDS=list(range(7103248,7103258))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def pinned(e):
 assert sha(e['path'])==e['sha256'];return read(e['path'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root_grant',type=Path);a=ap.parse_args();g=read(a.root_grant)
 assert g['approved']is True and g['reviewed_by']=='root'and g['action']=='export_fixed_prefix22_science_ten'
 assert g['numerical_slots']==0 and g['controller_sha256']==sha(__file__)and g['source_sha256']==SOURCE_SHA
 assert g['seeds']==SEEDS and g['no_old_physics_parity_gate']is True and g['fresh_production_counts_required']is True
 for p,h in g['source_pins'].items():assert sha(p)==h
 accept=pinned(g['science24_root_acceptance'])
 assert accept['approved']is True and accept['reviewed_by']=='root'and accept['candidate_sha256']==SOURCE_SHA
 decision=pinned(accept['decision'])
 assert decision['n']==24 and decision['passed']is True and decision['all_actual_retired']is True and decision['source']['sha256']==SOURCE_SHA
 assert all(decision['means'][role]['net_log_pct']<0 and decision['groups']['OOS8'][role]['net_log_pct']<0 for role in ('PRIMARY','e406'))
 protocol=SCI/'PROTOCOL_REGISTERED_astra_v1.json';c=read(protocol);evidence={str(protocol):sha(protocol),str(a.root_grant):sha(a.root_grant),accept['decision']['path']:accept['decision']['sha256']}
 evaluations=[]
 # Whole-phase actual ownership barriers precede numerical certificate reads.
 for phase in ('PILOT','REMAINING'):
  p=SCI/f'{phase}_RETIREMENT_astra_v1.json';r=read(p)
  assert r['approved']is True and r['reviewed_by']=='root'and r['all_owned_PID_PPID_PGID_absent']is True
  assert r['all_actual_exits_observed']is True and r['all_guards_clean']is True and r['candidate_sha256']==SOURCE_SHA
  assert r['protocol_sha256']==sha(protocol)
  for f,h in r['evidence_sha256'].items():assert sha(f)==h
  evidence[str(p)]=sha(p);evaluations.extend(r['evaluations'])
 assert sorted(e['seed']for e in evaluations)==list(range(7103248,7103272))
 out=D/'native_ten_from_science_v1';out.mkdir()
 for role,label,source_sha,entries in [('ON','arm',SOURCE_SHA,evaluations),('PRIMARY','base',BASE_SHA,c['stack_control_reuse'])]:
  rows=[];source=None
  for seed in SEEDS:
   e=next(x for x in entries if x['seed']==seed)
   for key in ('certificate','guard'):assert sha(e[key])==e[key+'_sha256'];evidence[e[key]]=e[key+'_sha256']
   p=Path(e['certificate']);pp=p.with_name(p.stem+'_pred.npz');assert sha(pp)==e['prediction_npz_sha256'];evidence[str(pp)]=sha(pp)
   cert=read(p);guard=read(e['guard']);r=cert['rows'][0]
   assert cert['sha256']==source_sha and sha(cert['candidate'])==source_sha
   assert guard['completed']is True and guard['exit_code']==0 and guard['whole_group_retired']is True and not guard['remaining_rows']
   assert cert['environment']['PB_ALL']=='1'and cert['setup']['pb_wall_s']==1e9 and r['seed']==seed
   with np.load(pp,allow_pickle=False)as z:pred=z[str(seed)].copy()
   assert pred.shape==(16,1024)and pred.dtype==np.float32 and np.isfinite(pred).all()
   assert hashlib.sha256(pred.tobytes()).hexdigest()[:16]==r['pred_sha']
   assert len(r['mse_layers'])==16 and all(math.isfinite(v)and v>0 for v in r['mse_layers'])and r['mse_final']==r['mse_layers'][-1]
   assert type(r['flops'])is int and r['flops']>0 and type(r['ops'])is int and r['ops']>0
   if role=='ON':
    hf=r['stats']['_ASTRA_HF_STATS'];assert hf['calls']==28 and hf['retired']is True and hf['ba']==hf['packed']==14
    assert r['hermite_route']==dict(fusion_on=True,extra31_inactive=True)and r['hermite_cleanup']==dict(row_absent=True)
   pf=out/f'quick_{label}_ten_pred_{seed}.npy'
   with pf.open('xb')as f:np.save(f,pred,allow_pickle=False)
   evidence[str(pf)]=sha(pf);source=cert['candidate']
   rows.append(dict(rank=seed,seed=seed,mse=r['mse_final'],flops=r['flops'],ops=r['ops'],pred_sha=r['pred_sha']))
  q=out/f'quick_{label}_ten.json';write(q,dict(candidate=source,candidate_sha256=source_sha,rows=rows,profile='Exact-source PB_ALL1/PB_WALL1e9 native science; production bill/timing from separate fresh normal/slow. No rounding-parity gate.'))
  evidence[str(q)]=sha(q)
 write(D/'NATIVE_TEN_SCIENCE_EXPORT_v1.json',dict(source_sha256=SOURCE_SHA,baseline_sha256=BASE_SHA,seeds=SEEDS,all10_all16_finite=True,exact_final_source=True,no_new_model_calls=True,no_old_physics_parity_gate=True,no_timing_or_production_bill_transfer=True,science24_root_acceptance=g['science24_root_acceptance'],root_grant_sha256=sha(a.root_grant),evidence_sha256=evidence,status='EXACT_FINAL_SOURCE_TEN_VALIDATED_SCIENCE_RUNTIME_SEAL_RELEASE_REQUIRED'))
 print('Exact final-source native ten exported; no models, old-physics parity gate, or production timing transfer.')
if __name__=='__main__':main()
