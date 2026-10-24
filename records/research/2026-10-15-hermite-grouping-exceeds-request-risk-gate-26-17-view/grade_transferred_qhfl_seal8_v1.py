"""After root whole-owner/production barriers, grade fixed8 against pinned e406."""
import argparse,hashlib,importlib.util,json,math,statistics,subprocess
from pathlib import Path
import numpy as np
D=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_qhfl_delivery_v1/seal');OUT=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def write(p,x):
 with Path(p).open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('root_grant',type=Path);a=ap.parse_args();g=read(a.root_grant)
 assert g['approved']is True and g['reviewed_by']=='root'and g['action']=='grade_fixed_qhfl_private8'
 assert g['numerical_slots']==0 and g['controller_sha256']==sha(__file__)
 p=D/'finite_qhfl_seal_arms_v1.py';assert sha(p)==g['seal_controller_sha256']
 spec=importlib.util.spec_from_file_location('_qhfl_seal_grade',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 assert g['measurement_candidate_sha256']==m.m.SOURCE_SHA;m.pins()
 transfer=g['semantics_transfer'];assert sha(transfer['path'])==transfer['sha256'];t=read(transfer['path'])
 assert t['approved']is True and t['reviewed_by']=='root'and t['candidate_sha256']==g['candidate_sha256']
 assert t['measurement_candidate_sha256']==m.m.SOURCE_SHA
 assert all(t[k]is True for k in ('pure_basic_view_handle_cache_only','native_ten_all16_bytes_equal','native_ten_charged_FLOPs_equal','prediction_and_bill_semantics_unchanged'))
 for f,h in t['evidence_sha256'].items():assert sha(f)==h

 evidence={str(a.root_grant):sha(a.root_grant),str(p):sha(p)}
 prod=g['production_root_acceptance'];assert sha(prod['path'])==prod['sha256'];production=read(prod['path'])
 assert production['approved']is True and production['reviewed_by']=='root'and production['candidate_sha256']==g['candidate_sha256']
 assert all(production[k]is True for k in ('normal_pass','slow_pass','science24_accepted','all_owned_PID_PPID_PGID_absent'))
 for f,h in production['evidence_sha256'].items():assert sha(f)==h
 evidence[prod['path']]=prod['sha256']
 assert set(g['phase_retirements'])==set(map(str,range(4)))
 allids=set();groups=set()
 for shard in range(4):
  e=g['phase_retirements'][str(shard)];assert sha(e['path'])==e['sha256'];r=read(e['path']);ready=D/f'PHASE_READY_initial8_s{shard}_v1.json';x=read(ready)
  assert r['approved']is True and r['reviewed_by']=='root'and r['actual_managed_exit_observed']is True and r['exit_code']==0
  assert r['all_owned_PID_PPID_PGID_absent']is True and r['ready_sha256']==sha(ready)
  assert r['controller_pid']==x['controller_pid']and x['source_sha256']==m.m.SOURCE_SHA and x['seeds']==m.m.seeds_for('initial8',shard)
  assert not(D/f'OWNER_LOCK_initial8_s{shard}_v1').exists()
  for f,h in x['evidence_sha256'].items():assert sha(f)==h
  allids.update(r['owned_pids_checked_absent']);groups.update(r['owned_pgids_checked_absent'])
  evidence[e['path']]=e['sha256'];evidence[str(ready)]=sha(ready)
 live=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()if s.strip()]
 assert not[row for row in live if row[0]in allids or row[1]in allids or row[2]in groups]
 cached=read(D/'CACHED_E406_SEAL8_v1.json');pairs={'base':[],'arm':[]}
 for shard,item in enumerate(cached):
  for e in (item['ready'],item['retirement']):assert sha(e['path'])==e['sha256'];evidence[e['path']]=e['sha256']
  assert read(item['retirement']['path'])['all_owned_PID_PPID_PGID_absent']is True
  for f,h in item['evidence_sha256'].items():assert sha(f)==h;evidence[f]=h
  cp=next(Path(f)for f in item['evidence_sha256']if f.endswith('/cert.json'));bc=read(cp)
  assert bc['sha256']==m.m.BASE_SHA and bc['candidate']==str(m.m.BASE)and bc['seeds']==item['seeds']
  pp=cp.with_name('cert_pred.npz')
  with np.load(pp,allow_pickle=False)as z:
   for row in bc['rows']:
    y=z[str(row['seed'])];assert y.shape==(16,1024)and y.dtype==np.float32 and np.isfinite(y).all()
    assert hashlib.sha256(y.tobytes()).hexdigest()[:16]==row['pred_sha']
  rows,ev,_=m.checked('initial8','arm',shard);evidence.update(ev)
  pairs['base'].extend(bc['rows']);pairs['arm'].extend(rows)
 seeds=list(range(7190080,7190088));assert all([r['seed']for r in pairs[role]]==seeds for role in pairs)
 diffs=[math.log(a['mse_final']/b['mse_final'])+math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))for b,a in zip(pairs['base'],pairs['arm'])]
 mean=statistics.mean(diffs);se=statistics.stdev(diffs)/len(diffs)**.5;verdict='PASS'if mean<0 else('EXTEND'if mean<se else'FAIL')
 write(OUT/'DECISION_initial8_v1.json',dict(candidate_sha256=g['candidate_sha256'],measurement_candidate_sha256=m.m.SOURCE_SHA,semantics_transfer=transfer,baseline_sha256=m.m.BASE_SHA,seeds=seeds,n=8,mean_log_net=mean,se_log_net=se,verdict=verdict,per_seed_log_net=diffs,evidence_sha256=evidence,production=prod,original_rule_preserved=True,automatic_extension_authorized=False,release_or_fire_authorized=False))
 print(json.dumps(dict(n=8,verdict=verdict,mean_log_net=mean,se_log_net=se)))
if __name__=='__main__':main()
