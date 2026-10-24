from pathlib import Path
import json,hashlib,ast,sys,types
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'lead/runs/astra_rtt2_prefix_z1_late_v1';Z=D.parent/'astra_rtt2_prefix_z1_v1';P=D.parent/'astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();s=P.read_text();assert sha(P)=='4d7739a5ad4444be8c20bdd5c5bc86f1cee5536a2c21e956d86dd95e6c83d8d9'
n=[n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)and n.name=='_rtt_read'][-1];lines=s.splitlines(keepends=True);start=sum(map(len,lines[:n.lineno-1]));stop=sum(map(len,lines[:n.end_lineno]));body=s[start:stop]
old='floor=.5*_rc_math.erfc(1.5/_rc_math.sqrt(2.0))';new='floor=.5*_rc_math.erfc((1.0 if 10<=layer<=14 else 1.5)/_rc_math.sqrt(2.0))';assert body.count(old)==1
edited=body.replace(old,new);out=s[:start]+edited+s[stop:];assert out[:start]+out[start:start+len(edited)].replace(new,old)+out[start+len(edited):]==s
candidate=D/'candidate_QR_prefix_z1_late_ON_HELD_v1.py';candidate.write_text(out);ast.parse(out)
# Exact constant expression coverage: only L10..14 differ, no numeric knobs.
import math
checks=[]
for layer in range(16):
 cutoff=1.0 if 10<=layer<=14 else 1.5
 assert cutoff==(1.0 if layer in (10,11,12,13,14) else 1.5)
 checks.append(dict(layer=layer,cutoff=cutoff,gate_floor=.5*math.erfc(cutoff/math.sqrt(2))))
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub;m=types.ModuleType('_pzl_import');m.__file__=str(candidate);sys.modules[m.__name__]=m
exec(compile(out,str(candidate),'exec'),m.__dict__);assert m._RC_PARENT_READ is m._rtt_read and m._RTP_ON is True and m._RTT_ACTIVE is False and m._RTT_CONTEXT==[None]
ready=dict(status='FROZEN_SINGLE_LATE_BAND',candidate=str(candidate),candidate_sha256=sha(candidate),parent=str(P),parent_sha256=sha(P),exact_inverse_recovers_parent=True,changed_layers=list(range(10,15)),unchanged_prediction_prefix_layers=list(range(10)),all_other_source_bytes_unchanged=True,full_import_only_pass=True,tiny_fixed_cutoff_checks=checks,no_setup_predict_models=True,cost_forecast_mean_pct=.4841316701687607,cost_receipt_sha256=sha(D/'SAVED8_LATE_COST_v1.json'),root_exception='Root expressly accepted0.484% as close enough; retain fixed L10..14, no widening.')
r=D/'SOURCE_READY_HELD_v1.json';r.write_text(json.dumps(ready,indent=2)+'\n')
purpose='pzl_qr_e406_pilot8_fresh16_oos8'
runner=(Z/'finite_pz1_astra_v1.py').read_text().replace('pz1','pzl').replace('PZ1','PZL')
runner=runner.replace('import hashlib,importlib.util,json,os,sys','import hashlib,importlib.util,json,os,sys,signal\nimport numpy as np')
needle="assert st['_YR7_STATS']['retired'] and st['_YR7_STATS']['value_cache_entries']==0"
runner=runner.replace(needle,needle+'''
            control=next(x for x in c['stack_control_reuse'] if x['seed']==seed)
            cp=Path(control['certificate']);pp=cp.with_name(cp.stem+'_pred.npz')
            assert sha(cp)==control['certificate_sha256'] and sha(pp)==control['prediction_npz_sha256']
            bc=json.loads(cp.read_text());assert bc['sha256']==c['candidates']['stack']['sha256']
            with np.load(pp,allow_pickle=False)as z:assert a['pred'][:10].tobytes()==z[str(seed)][:10].tobytes()
            assert a['cert']['setup']['pb_single_sha256']==bc['setup']['pb_single_sha256']
''')
runner=runner.replace("if __name__=='__main__':run(sys.argv[1],int(sys.argv[2]))",'''def interrupted(signum,frame):raise KeyboardInterrupt('Owned late-prefix collector signal%s'%signum)
if __name__=='__main__':
    old=signal.signal(signal.SIGTERM,interrupted)
    try:run(sys.argv[1],int(sys.argv[2]))
    finally:signal.signal(signal.SIGTERM,old)''')
f=D/'finite_pzl_astra_v1.py';f.write_text(runner)
j=D/'native_pzl_judge_astra_v1.py';j.write_text((Z/'native_pz1_judge_astra_v1.py').read_text().replace('pz1','pzl'))
c=json.loads((Z/'PROTOCOL_REGISTERED_astra_v1.json').read_text());c['purpose']=purpose;c['candidates']['ON'].update(path=str(candidate),sha256=sha(candidate));c['source_pins']={k:v for k,v in c['source_pins'].items()if not str(k).startswith(str(Z))};c['source_pins'].update({str(p):sha(p)for p in [P,f,j,r,D/'SAVED8_LATE_COST_v1.json']})
c.update(executed_by='compute_hunt',approved=True,reviewed_by='root',approved_by='root',conditional_root_authorization='Root expressly authorized fixedL10..14 z1 pilot8; negative vsQR ande406 plus actualallretirement automatically authorizes16ONremaining; finalall24/OOS8 negativeboth, fresh16report. Four slots/cap7/unchangedguards; no tuning.',numerically_authorized_phase='pilot_then_conditional_remaining',run_names={phase:{str(i):f'run_pzl_{phase}_s{i}_v1'for i in range(4)}for phase in ('pilot','remaining')},decision_names=dict(pilot='PILOT_DECISION_astra_v1.json',remaining='ALL24_DECISION_astra_v1.json'),source_readiness=dict(path=str(r),sha256=sha(r)),root_source_decision='fixedband10..14 costexception0.484 accepted',readiness=ready,no_followup_authorization=False)
c['rule']=dict(pilot='meanlog adjusted strictly negative vsQR ande406',final='all24 andOOS8 meanlog adjusted strictly negative vsQR ande406; fresh16 reported only',continuation='only after fullpilot8 actualretirement and dualPASS',source='fixedL10..14 z1, all earlierL1..9 z1.5, no tuning')
p=D/'PROTOCOL_REGISTERED_astra_v1.json';p.write_text(json.dumps(c,indent=2)+'\n')
print(json.dumps(dict(candidate_sha256=sha(candidate),protocol_sha256=sha(p),runner_sha256=sha(f),judge_sha256=sha(j),ready_sha256=sha(r))))
