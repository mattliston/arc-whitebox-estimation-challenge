"""Freeze one native full384 implementation/whole-bill check. No model call."""
from pathlib import Path
import hashlib,json,datetime
D=Path(__file__).resolve().parent;R=D.parent;H=R/'astra_smirnov_final_native_ten_v1';V=R/'astra_full384_smirnov_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
 p=D/name
 with p.open('x')as f:
  if isinstance(value,str):f.write(value)
  else:json.dump(value,f,indent=2);f.write('\n')
 return p
source=V/'candidate_full384_smirnov_HELD_v2.py';peer=V/'LARGER_ACCURACY_SOURCE_PEER_v1.json'
assert sha(source)=='e33563bc0135b08226fcfcef15033ca4a8e46264b95708711832e23200c744be'
assert sha(peer)=='0c84caf29aeae655b60f300c50ce28b566abdfefa05a0ee725e46f778b27d893'
runner=(H/'finite_final_astra_v1.py').read_text().replace('smirnov_final_exact_native_ten','full384_smirnov_fixed_bill1').replace('native_final_judge_astra_v1.py','native_bill1_judge_astra_v1.py').replace('ROOT_NATIVE_TEN_GRANT_v1.json','ROOT_BILL1_GRANT_v1.json').replace('run_final_{phase}','run_full384_{phase}')
runner=runner.replace("phase == 'parity' and shard in range(2)","phase == 'bill' and shard == 0")
runner=runner.replace("grant['native_calls_max']==10 and grant['research_slots']==2","grant['native_calls_max']==1 and grant['research_slots']==1")
runner=runner.replace("[shard::2]","[shard::1]")
runner=runner.replace("a['pred'].tobytes()==z[str(seed)].tobytes()","a['pred'][:10].tobytes()==z[str(seed)][:10].tobytes()")
runner=runner.replace("assert a['row']['mse_layers']==bc['rows'][0]['mse_layers']","assert a['row']['mse_layers'][:10]==bc['rows'][0]['mse_layers'][:10]")
old="""            costrow=next(x for x in c['final_cost_rows'] if x['seed']==seed)
            assert bc['rows'][0]['flops']==costrow['science_flops']
            # Actual final bill is recorded unchanged; exact metadata transfer is graded after retirement."""
assert old in runner;runner=runner.replace(old,"            # Whole bill is graded only after the actual owned terminal/guard exit.")
runner=runner.replace('final_source_contract_v1','full384_source_contract_v1')
write('finite_full384_bill1_v1.py',runner)
judge=(H/'native_final_judge_astra_v1.py').read_text().replace('smirnov_final_exact_native_ten','full384_smirnov_fixed_bill1').replace('final_source_contract_v1','full384_source_contract_v1')
judge=judge.replace("set(phase_seeds)=={'parity'} and len(phase_seeds['parity'])==10","set(phase_seeds)=={'bill'} and len(phase_seeds['bill'])==1").replace("allowed=phase_seeds['parity'];assert len(set(allowed))==10","allowed=phase_seeds['bill'];assert allowed==[7103248]")
judge=judge.replace("sm40_lifecycle_before=sm40_before,","full384_rule=[[k,mod._R320_RULE[k]]for k in (7,10,13)],\n         sm40_lifecycle_before=sm40_before,")
write('native_bill1_judge_astra_v1.py',judge)
contract=(H/'final_source_contract_v1.py').read_text()+'''
# The fixed full384 candidate transports the same cost-selected plans to rank384.
PLAN_KEYS=frozenset(k[:3]+(384,) if k[3]==320 else k for k in PLAN_KEYS)
_full384_parent_gate=gate
def gate(row):
    _full384_parent_gate(row)
    assert row['full384_rule']==[[7,384],[10,384],[13,384]]
    reads=row['stats']['_RTT_STATS']['reads']
    assert [r['layer']for r in reads]==list(range(1,15))
    assert all(r['old_rank']==(0 if r['layer']<7 else 384)for r in reads)
'''
write('full384_source_contract_v1.py',contract)
c=json.loads((H/'PROTOCOL_REGISTERED_astra_v1.json').read_text());controls=[]
for shard in (0,1):controls.extend(json.loads((H/f'run_final_parity_s{shard}_v1/READY_v1.json').read_text())['evaluations'])
control=next(e for e in controls if e['seed']==7103248)
assert json.loads(Path(control['certificate']).read_text())['sha256']=='2b10843a76fc033325ceee9dd30703eff29a1fa535cdc094c5feef72023a0c3c'
c.update(status='REGISTERED',purpose='full384_smirnov_fixed_bill1',phase_seeds={'bill':[7103248]},identity_groups={'bill':[7103248]},shards=[0],research_slots=1,native_calls_max=1,stack_control_reuse=[control],remaining_pilot_seeds=list(range(7103249,7103256)),no_followup_authorization=True,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rule=dict(scope='One fixed full384 whole-bill and implementation check; no numerical selection.',identity='First10 prediction bytes/layerMSE exact against cached actual current2b108; all16 finite; all hidden old ranks384; existing cleanup/ownership gates.',continuation='Root decides after actual wholebill. No more models under this grant; the fixed next7 would complete pilot8.',MSE='Not used for bill1 admission/continuation.'),source_readiness=dict(path=str(peer),sha256=sha(peer)),run_names=['run_full384_bill_s0_v1'],decision_names=['BILL1_RETIREMENT_v1.json','BILL1_DECISION_v1.json'])
c['candidates']['stack']=c['candidates']['ON'];c['candidates']['ON']=dict(path=str(source),sha256=sha(source),mode=1)
for k in ('final_cost_rows','final_cost_metadata'):c.pop(k,None)
pins=[source,peer,V/'RANK_BASE_SOURCE_READY_v1.json',V/'FIXED_SOURCE_READY_v2.json',D/'finite_full384_bill1_v1.py',D/'native_bill1_judge_astra_v1.py',D/'full384_source_contract_v1.py',D/'grade_bill1_v1.py',Path(__file__)]
c['source_pins'].update({str(p.resolve()):sha(p)for p in pins})
for kind in ('certificate','guard'):c['source_pins'][control[kind]]=control[kind+'_sha256']
cp=Path(control['certificate']);c['source_pins'][str(cp.with_name(cp.stem+'_pred.npz'))]=control['prediction_npz_sha256']
for path,digest in c['source_pins'].items():assert sha(path)==digest
write('PROTOCOL_REGISTERED_v1.json',c)
# Existing runner locates this fixed filename. Both files are immutable and equal.
write('PROTOCOL_REGISTERED_astra_v1.json',c)
for name in ('finite_full384_bill1_v1.py','native_bill1_judge_astra_v1.py','full384_source_contract_v1.py'):compile((D/name).read_text(),str(D/name),'exec')
print(json.dumps(dict(protocol_sha256=sha(D/'PROTOCOL_REGISTERED_astra_v1.json'),runner_sha256=sha(D/'finite_full384_bill1_v1.py'),judge_sha256=sha(D/'native_bill1_judge_astra_v1.py'),pins=len(c['source_pins']))))
