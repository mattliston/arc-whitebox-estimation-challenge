"""Fixed paired no-grad CHECK8; never fits parameters or selects another point."""
from pathlib import Path
import argparse,hashlib,json,os,subprocess,math,time
P=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_oct18_accuracy_v1/current_torch_rtt_port_v1');R=P.parents[1];ROOT=Path('/Users/matt/arc-white-box-estimation');O=R.parents[1]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--point',type=Path,required=True);ap.add_argument('--point-sha',required=True);ap.add_argument('--out-dir',type=Path,required=True);a=ap.parse_args()
 binding=json.loads((P.parent/'post336935_training_audit_v1/BINDING_v1.json').read_text());physics=json.loads((P/'offline_train8_v2/PLAN_v1.json').read_text())['source_pins']
 assert sha(a.point)==a.point_sha
 point=json.loads(a.point.with_name('FIXED_STEP_v1.json').read_text());assert point['status']=='LARGER_POINT_AWAITING_TRAIN_SCREEN' and point['check']==list(range(7103472,7103480)) and point['base_params_sha256']==binding['params_sha256']
 harness=P/'run_fixed_check8_forward_v2.py';assert sha(harness)=='7f8e317c11b25b2db87eafba952da38bf82245e9b325d51c6ab51964eafea608'
 a.out_dir.mkdir();rows=[];start=time.monotonic()
 env=dict(os.environ,TRAINER_THREADS='1',TRAINER_PL='1',TRAINER_CARD='M',TRAINER_RELIEF='1',TRAINER_STAR_ATTACH='0',TRAINER_BIRTH_ATTACH='0')
 for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):env[name]='1'
 write(a.out_dir/'RUN_PLAN_v1.json',dict(check=point['check'],point=str(a.point.resolve()),point_sha256=a.point_sha,harness_sha256=sha(harness),source_pins=physics,no_optimizer=True,arms=['control','point'],order='per-seed control then point, fresh processes',memory_limit=7500000000))
 try:
  for seed in point['check']:
   pair={'seed':seed}
   for arm,params,params_sha in [('control',Path(binding['params']),binding['params_sha256']),('point',a.point,a.point_sha)]:
    for path,h in physics.items():assert sha(path)==h,path
    out=a.out_dir/f'{seed}_{arm}';guard=a.out_dir/f'GUARD_{seed}_{arm}_v1.json'
    cmd=[str(ROOT/'.venv/bin/python'),str(R/'astra_crunch_physics_v1/guard_memory50_v1.py'),'--record',str(guard),'--seconds','180','--footprint-limit','7500000000','--rss-backstop','7500000000','/Users/matt/anaconda3/bin/python','-B',str(harness),'--rank',str(seed),'--w3module',str(R/'astra_oct18_lead_v1/current_torch_jsk_port_v1/torch_w3_current95c_v2.py'),'--params',str(params.resolve()),'--params-sha',params_sha,'--arm',arm,'--out-dir',str(out)]
    print(json.dumps(dict(event='START',seed=seed,arm=arm)),flush=True)
    with (a.out_dir/f'LOG_{seed}_{arm}_v1.txt').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert r.returncode==0,(seed,arm,r.returncode)
    g=json.loads(guard.read_text());assert g['completed'] and g['whole_group_retired'] and not g['error'] and not g['remaining_rows'] and g['min_pressure_free_pct']>=50
    c=json.loads((out/'COMPLETE_v1.json').read_text());assert sha(c['result'])==c['result_sha256'] and c['params_sha256']==params_sha
    data=json.loads(Path(c['result']).read_text());assert data['mode']=='eval' and data['rank']==seed and data['peak_rss']<7500000000 and math.isfinite(data['mse'])
    assert all(math.isfinite(v) for layer in data['prediction'] for v in layer)
    pair[arm]=dict(result=c['result'],result_sha256=c['result_sha256'],mse=data['mse'],guard=str(guard),peak_rss=data['peak_rss'])
   rows.append(pair)
   print(json.dumps(dict(event='PAIR_COMPLETE',seed=seed)),flush=True)
  write(a.out_dir/'PAIRS_v1.json',dict(status='ALL_EIGHT_PAIRS_COMPLETE_AND_RETIRED',rows=rows,point_sha256=a.point_sha,elapsed_seconds=time.monotonic()-start,parameters_updated=False,source_pins=physics))
 except BaseException as e:
  write(a.out_dir/'STOP_v1.json',dict(error=repr(e),completed_seeds=[x['seed'] for x in rows],parameters_updated=False));raise
if __name__=='__main__':main()
