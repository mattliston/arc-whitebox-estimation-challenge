"""Freeze final helper transfer. Metadata only; source, cost, and lifecycle peer are required."""
from pathlib import Path
import argparse,datetime,hashlib,json
D=Path(__file__).resolve().parent;R=D.parent
S=R/'astra_smirnov_rpe_science_v1';H=R/'astra_prefix22_uvp_split_native_ten_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
 p=D/name
 with p.open('x')as f:
  if isinstance(value,str):f.write(value)
  else:json.dump(value,f,indent=2);f.write('\n')
 return p
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--peer',required=True);p.add_argument('--cost',required=True);p.add_argument('--contract',required=True);a=p.parse_args()
source=Path(a.source).resolve();peer=Path(a.peer).resolve();cost=Path(a.cost).resolve();contract=Path(a.contract).resolve()
cm=json.loads(cost.read_text());assert cm['candidate_sha256']==sha(source)
assert [r['seed']for r in sorted(cm['rows'],key=lambda r:r['seed'])]==list(range(7103248,7103272))
assert all(type(r['final_flops'])is int and type(r['science_flops'])is int and r['delta_flops']==r['final_flops']-r['science_flops']for r in cm['rows'])
rule=S/'ROOT_EXACT_HELPER_BILL_TRANSFER_RULE_v1.json';assert json.loads(rule.read_text())['approved']
prefetch=S/'ROOT_OPTIONAL_YR7_PREFETCH_REVIEW_v1.json';assert json.loads(prefetch.read_text())['approved']

runner=(H/'finite_uvp_astra_v1.py').read_text().replace('accepted exact7b126 controls','accepted exact204880a5 controls').replace('prefix22_uvp_split_exact_native_ten','smirnov_final_exact_native_ten')
runner=runner.replace('native_uvp_judge_astra_v1.py','native_final_judge_astra_v1.py').replace('run_uvp_{phase}','run_final_{phase}')
runner=runner.replace("'_RE22_ACTIVE')","'_RE22_ACTIVE','_ASTRA_SM40_STATS','_ASTRA_SM40_BIND_STATS')")
runner=runner.replace("from rpe_source_contract_v1 import gate","from final_source_contract_v1 import gate")
runner=runner.replace("st['_YR7_BIND_STATS']==dict(keys=17,retired=True)","st['_YR7_BIND_STATS']==dict(keys=0,retired=True)")
old="            assert a['row']['flops']==bc['rows'][0]['flops'], 'Administrative view repair changed FLOPs'"
assert runner.count(old)==1
runner=runner.replace(old,"""            assert a['row']['mse_layers']==bc['rows'][0]['mse_layers']
            costrow=next(x for x in c['final_cost_rows'] if x['seed']==seed)
            assert bc['rows'][0]['flops']==costrow['science_flops']
            assert a['row']['flops']==costrow['final_flops'], 'Actual final bill differs from frozen exact metadata repricing'""")
write('finite_final_astra_v1.py',runner)
judge=(S/'native_sm40_judge_astra_v1.py').read_text().replace('smirnov_rpe_fixed_primary_incremental_pilot8_fresh16_oos8','smirnov_final_exact_native_ten').replace('sm40_rpe_source_contract_v1','final_source_contract_v1')
old="assert set(phase_seeds)=={'pilot','remaining'} and len(phase_seeds['pilot'])==8 and len(phase_seeds['remaining'])==16\nallowed=phase_seeds['pilot']+phase_seeds['remaining'];assert len(set(allowed))==24"
assert judge.count(old)==1
judge=judge.replace(old,"assert set(phase_seeds)=={'parity'} and len(phase_seeds['parity'])==10\nallowed=phase_seeds['parity'];assert len(set(allowed))==10")
write('native_final_judge_astra_v1.py',judge)
write('final_source_contract_v1.py',contract.read_text())

base=json.loads((S/'PROTOCOL_REGISTERED_astra_v1.json').read_text());c=dict(base);seeds=list(range(7103248,7103258));controls=[]
for phase in ('pilot','remaining'):
 for shard in range(4):
  ready=json.loads((S/f'run_sm40_{phase}_s{shard}_v1/READY_v1.json').read_text());controls.extend(ready['evaluations'])
controls.sort(key=lambda e:e['seed']);assert [e['seed']for e in controls]==list(range(7103248,7103272))
for e in controls:
 for kind in ('certificate','guard'):assert sha(e[kind])==e[kind+'_sha256']
 cert=json.loads(Path(e['certificate']).read_text());assert cert['sha256']==base['candidates']['ON']['sha256']
 pp=Path(e['certificate']);assert sha(pp.with_name(pp.stem+'_pred.npz'))==e['prediction_npz_sha256']
 cr=next(r for r in cm['rows']if r['seed']==e['seed']);assert cr['science_flops']==cert['rows'][0]['flops']
c.update(status='REGISTERED',purpose='smirnov_final_exact_native_ten',approved_by='root',approval_authority='root',executed_by='fresh_algebra',phase_seeds={'parity':seeds},identity_groups={'parity':seeds,'oos':list(range(7103264,7103272))},shards=[0,1],research_slots=2,native_calls_max=10,stack_control_reuse=controls,primary_control_reuse=base['stack_control_reuse'],final_cost_rows=cm['rows'],final_cost_metadata=dict(path=str(cost),sha256=sha(cost)),rule=dict(scope='Fixed ten exact helper transfer; no numerical selection.',parity='All16 prediction bytes and all layer MSEs equal frozen204880a5 science.',cost='Actual first10 final bills equal exact fixed24 metadata. Repriced all24 and OOS8 adjusted log negative versus PRIMARY and RPE.',failure='Stop on any difference; no automatic retry.'),no_followup_authorization=True,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),run_names=['run_final_parity_s0_v1','run_final_parity_s1_v1'],decision_names=['PARITY_RETIREMENT_astra_v1.json','NATIVE_TEN_PARITY_v1.json'])
c['candidates']=dict(ON=dict(path=str(source),sha256=sha(source),mode=1),stack=base['candidates']['ON'],primary=base['candidates']['stack'],reference=base['candidates']['reference'])
c['source_readiness']=dict(path=str(peer),sha256=sha(peer));c['source_pins']=dict(base['source_pins'])
pins=[source,peer,cost,contract,rule,prefetch,Path(__file__),D/'finite_final_astra_v1.py',D/'native_final_judge_astra_v1.py',D/'final_source_contract_v1.py',D/'grade_native_ten_v1.py']
c['source_pins'].update({str(p.resolve()):sha(p)for p in pins})
for e in controls:
 pp=Path(e['certificate']);c['source_pins'][str(pp)]=e['certificate_sha256'];c['source_pins'][str(pp.with_name(pp.stem+'_pred.npz'))]=e['prediction_npz_sha256'];c['source_pins'][e['guard']]=e['guard_sha256']
for path,digest in c['source_pins'].items():assert sha(path)==digest
write('PROTOCOL_REGISTERED_astra_v1.json',c)
print(json.dumps(dict(protocol_sha256=sha(D/'PROTOCOL_REGISTERED_astra_v1.json'),runner_sha256=sha(D/'finite_final_astra_v1.py'),judge_sha256=sha(D/'native_final_judge_astra_v1.py'),pins=len(c['source_pins']),candidate_sha256=sha(source))))
