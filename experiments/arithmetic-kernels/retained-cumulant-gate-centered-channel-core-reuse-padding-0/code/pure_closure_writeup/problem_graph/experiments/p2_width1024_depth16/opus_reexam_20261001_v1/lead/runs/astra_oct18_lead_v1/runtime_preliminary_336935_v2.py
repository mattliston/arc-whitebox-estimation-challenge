"""Read-only preliminary runtime screen against graded95c/#336935.

CLI: CANDIDATE_DIRECTORY SOURCE_SHA256 {normal,slow13} OUTPUT
No SDK, model, setup gate, or submission is run. PASS is not release approval.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
O = RUNS.parents[1]
ACCURACY = RUNS / 'astra_oct18_accuracy_v1'
GATE_SHA = 'e8119395c599d48a1033d8107fe37cc7732e8b6403cdd9bba2806cc7dacd93cc'
BASE_SHA = '95c8717670a5b4fbc2dcf57e289a5a0f5a8c6f9c656bee3a638655deb94b0cdf'
EVAL_SHA = '14751b61bc89618a31c9b99c8132929bd94b4716e7d32e3bbcf7e18f2d299407'
LIMITS = dict(residual_median_s=.3452224775, residual_max_s=.390682602, roundtrip_max_s=118.)
CONTROL_PINS = {
 'normal': {'manifest.json':'13bc9a6199fc505af2491ac806ed6b1d6e4cd3355b52f327612b172773db5771',
  'launcher.json':'951af6434c14b9830a2644f4d12e335215e451681aa3e4a160aede6385919e3e',
  'GUARD_v1.json':'71553142e329719f30151c197f8e3682a3886bd15f86f271adcf72159f8badd3',
  'rows.jsonl':'ed3a544ceb634521de5d6bfb8348a2bf3d62ea9741017371553976cec04754de'},
 'slow13': {'manifest.json':'7c0058709013e039a7bb8733fad4d045c756920c8ee3be52dbc73c7f7c0cb9de',
  'launcher.json':'694d6010d5b8bc4f5affc51ab0876201348a70f84f01f32b91d9be34ed6b9e53',
  'GUARD_v1.json':'725e1965a353152f02b075d8ef24bc347ce94216e0abcebfcbc7230a70dc24d8',
  'rows.jsonl':'9f3e00da4867ff9df403f0664db1ffe6194a02729af5a0e23d3648f2c0e8762a'}}

def sha(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def require(value, message):
 if not value: raise ValueError(message)

def read(path):
 return json.loads(Path(path).read_text())

def sdk_pins():
 root = RUNS/'astra_crunch_runtime_v1/failure336631_v1'
 records = []
 for env, dist in [('boundary310_v1','flopscope_client'),('server310_openblas_v1','flopscope'),('server310_openblas_v1','flopscope_server')]:
  path=root/env/f'lib/python3.10/site-packages/{dist}-0.12.3.dist-info/METADATA'
  require('Version: 0.12.3' in path.read_text().splitlines(), 'Runtime SDK metadata mismatch')
  records.append(dict(path=str(path),sha256=sha(path)))
 return records

def telemetry():
 path=O/'fire/astra_runtime_v65u_95c87176/eval_336935.json'
 require(sha(path)==EVAL_SHA,'Frozen336935 hosted telemetry changed')
 public=read(path);e=public['evaluation'];r=e['results']
 require(int(public['submission_id'])==336935 and int(e['submission_id'])==336935,'Wrong hosted submission')
 require(e['state']=='completed' and e['error'] is None and e['progress']['mlps_failed']==0,'Hosted336935 did not complete cleanly')
 require(len(r['per_mlp'])==100 and len({x['mlp_index']for x in r['per_mlp']})==100,'Hosted100 incomplete/duplicate')
 require(all(x['state']=='completed' and x['error_code'] is None for x in r['per_mlp']),'Hosted MLP failure')
 require(len(r['setup_executions'])==5 and all(x['error_code']=='SUCCESS'for x in r['setup_executions']),'Hosted setups incomplete')
 env=e['runtime_environment']
 require(all(env[k]=='0.12.3'for k in ('flopscope_version','flopscope_client_version','flopscope_server_version')),'Hosted SDK mismatch')
 ts=[x['telemetry']for x in r['per_mlp']]
 require(all(t['residual_wall_time_s']>0 and t['participant_roundtrip_wall_s']>0 for t in ts),'Invalid hosted times')
 require(sum(t.get('participant_setup_wall_s') is not None for t in ts)==5,'Expected five cold hosted MLPs')
 return ts,path

def evaluate(directory, source_sha, mode):
 gate_path=ACCURACY/'runtime_gate_thread_v6.py'
 require(sha(gate_path)==GATE_SHA,'Frozen group validator changed')
 sys.path.insert(0,str(ACCURACY))
 import runtime_gate_thread_v6 as gate
 require(all(gate.LIMITS[k]==v for k,v in LIMITS.items()),'Original336854 limits changed')
 versions=sdk_pins();pace=1. if mode=='normal'else 1.3
 reference=HERE/'compact10_repair_qualification_v1'/('candidate_'+mode)
 for name, digest in CONTROL_PINS[mode].items():require(sha(reference/name)==digest,'Frozen95c control changed: '+name)
 candidate=gate.group(directory,source_sha,pace)
 control=gate.group(reference,BASE_SHA,pace)
 # group verifies source/manifest/launcher hashes, retired7.5GB/free50 guards,
 # .12.3 client, exact thread-counter/helper/runner pins, no profiling or rescue.
 for group in (candidate,control):
  command=group['guard']['command']
  require(str(Path(group['manifest']['candidate']).resolve()) in command and group['directory'] in command,'Guard candidate/output path mismatch')
  require(group['manifest']['dual_counter']['thread_counter_scope']=='measurement thread only','Wrong thread counter scope')
 ts,tele_path=telemetry()
 ratios={k:candidate['rows'][k]['thread_residual_instructions']/control['rows'][k]['thread_residual_instructions']for k in (35,200)}
 process_ratios={k:candidate['rows'][k]['residual_instructions']/control['rows'][k]['residual_instructions']for k in (35,200)}
 residuals=[t['residual_wall_time_s']*ratios[35 if t.get('participant_setup_wall_s')is not None else 200]for t in ts]
 wall_ratio=max(candidate['rows'][k]['wall_s']/control['rows'][k]['wall_s']for k in (35,200))
 projected=dict(residual_median_s=statistics.median(residuals),residual_max_s=max(residuals),roundtrip_max_s=max(t['participant_roundtrip_wall_s']for t in ts)*wall_ratio)
 return dict(status='PASS'if all(projected[k]<=v for k,v in LIMITS.items())else'HOLD',base_sid=336935,base_source_sha256=BASE_SHA,
  source_sha256=source_sha,mode=mode,candidate_directory=candidate['directory'],control_directory=str(reference),control_receipt_pins=CONTROL_PINS[mode],
  telemetry_path=str(tele_path),telemetry_sha256=EVAL_SHA,runtime_metadata=versions,group_validator_sha256=GATE_SHA,
  projected=projected,thread_instruction_ratios=ratios,process_instruction_ratios_diagnostic=process_ratios,wall_ratio=wall_ratio,limits=LIMITS,
  measured_hosted_baseline=dict(residual_median_s=statistics.median(t['residual_wall_time_s']for t in ts),residual_max_s=max(t['residual_wall_time_s']for t in ts),roundtrip_max_s=max(t['participant_roundtrip_wall_s']for t in ts)),
  limitation='Preliminary projection using archived95c same-harness controls, not contemporaneous hardware qualification or release approval. Requires exact-source science32, cold259, both modes/repeatability, official five-start setup gate, memory and final release checks. No automatic fire.')

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('candidate_directory',type=Path);p.add_argument('source_sha256');p.add_argument('mode',choices=('normal','slow13'));p.add_argument('output',type=Path)
 a=p.parse_args();require(not a.output.exists(),'Never overwrite decision receipt')
 try: result=evaluate(a.candidate_directory,a.source_sha256,a.mode)
 except Exception as exc: result=dict(status='FAIL',reason=repr(exc),source_sha256=a.source_sha256,mode=a.mode,base_sid=336935)
 result['gate_source_sha256']=sha(__file__)
 with a.output.open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
 return 0 if result['status']=='PASS'else 1

if __name__=='__main__':raise SystemExit(main())
