"""Freeze one prospective uniform64 native check; preparation makes no model calls."""
from pathlib import Path
import hashlib,json,datetime,sys
D=Path(__file__).resolve().parent;R=D.parent;H=R/'astra_smirnov_final_native_ten_v1';F=R/'astra_full384_smirnov_bill1_v1';V=R/'astra_read256_inner_cost_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    p=D/name
    with p.open('x') as f:
        if isinstance(value,str):f.write(value)
        else:json.dump(value,f,indent=2);f.write('\n')
    return p
source=V/'candidate_uniform64_read_HELD_v1.py';ready=V/'UNIFORM64_SOURCE_HELD_v1.json';tiny=Path(sys.argv[1]).resolve()
assert sha(source)=='24b13a6763b83a9bf75186cc9aad5d4d424c5a2e9516c37b8676c839c964b9f4'
assert json.loads(ready.read_text())['candidate_sha256']==sha(source)
assert tiny.is_file()
runner=(F/'finite_full384_bill1_v1.py').read_text().replace('full384_smirnov_fixed_bill1','uniform64_read_fixed_bill1').replace('full384_source_contract_v1','uniform64_source_contract_v1').replace('run_full384_','run_uniform64_')
runner=runner.replace("D.parent/'astra_full384_smirnov_v1/ROOT_ONE_MODEL_GRANT_v1.json'","D.parent/'astra_read256_inner_cost_v1/ROOT_ONE_NATIVE_GRANT_v1.json'")
runner=runner.replace("grant['candidate_sha256']", "grant['source_sha256']").replace("grant['models']==[7103248]", "grant['seed']==7103248").replace("grant['max_simultaneous_native']==1", "grant['maximum_native_calls']==1")
runner=runner.replace("'_PCV_STATS')","'_PCV_STATS','_ASTRA_READ64_STATS')")
runner=runner.replace('[:10]','[:7]')
runner=runner.replace('"""Fixed ten ON-only administrative parity against accepted exact204880a5 controls."""','"""One uniform64 read candidate, compared against cached current2b108 after retirement."""')
write('finite_uniform64_bill1_v1.py',runner)
judge=(F/'native_bill1_judge_astra_v1.py').read_text().replace('full384_smirnov_fixed_bill1','uniform64_read_fixed_bill1').replace('full384_source_contract_v1','uniform64_source_contract_v1')
judge=judge.replace("         full384_rule=[[k,mod._R320_RULE[k]]for k in (7,10,13)],\n",'')
write('native_bill1_judge_astra_v1.py',judge)
contract=(H/'final_source_contract_v1.py').read_text()+'''
# Fixed prospective output-coordinate truncation only in old-bank vt reads.
_uniform64_parent_gate=gate
def gate(row):
    _uniform64_parent_gate(row)
    st=row['stats']['_ASTRA_READ64_STATS']
    assert st['retired'] is True
    events=st['events']
    expected=[(7,384),(8,384),(9,384),(10,320),(11,320),(12,320),(13,320),(13,384),(14,320),(14,384)]
    assert [(e[0],e[3]) for e in events]==expected
    assert all(len(e)==6 and e[4]==e[3]-64 and e[5]==256 and e[1]>0 and e[2]>0 for e in events)
    assert row['sm40_lifecycle_before']['prebind_count']==14
'''
write('uniform64_source_contract_v1.py',contract)
grader=(F/'grade_bill1_v1.py').read_text().replace('run_full384_','run_uniform64_').replace('full384_source_contract_v1','uniform64_source_contract_v1').replace('first-ten','first-seven').replace('first10','first7').replace('[:10]','[:7]')
write('grade_bill1_v1.py',grader)
c=json.loads((H/'PROTOCOL_REGISTERED_astra_v1.json').read_text());controls=[]
for shard in (0,1):controls.extend(json.loads((H/f'run_final_parity_s{shard}_v1/READY_v1.json').read_text())['evaluations'])
control=next(e for e in controls if e['seed']==7103248)
assert json.loads(Path(control['certificate']).read_text())['sha256']=='2b10843a76fc033325ceee9dd30703eff29a1fa535cdc094c5feef72023a0c3c'
c.update(status='REGISTERED',purpose='uniform64_read_fixed_bill1',phase_seeds={'bill':[7103248]},identity_groups={'bill':[7103248]},shards=[0],research_slots=1,native_calls_max=1,stack_control_reuse=[control],remaining_pilot_seeds=list(range(7103249,7103256)),no_followup_authorization=True,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rule=dict(scope='One fixed uniform64-coordinate vt read cut at every affected L7–14 bank. No rank/dial/grid selection.',identity='First7 prediction bytes/layerMSE exact against cached current2b108; all16 finite; ten fixed cut events; unchanged current lifecycle gates.',continuation='Root decides after actual wholebill and retired one-model result. No further models under this grant.',MSE='Single-model error reported after actual ownership retirement. No conditional automatic continuation.'),source_readiness=dict(path=str(ready),sha256=sha(ready)),run_names=['run_uniform64_bill_s0_v1'],decision_names=['BILL1_RETIREMENT_v1.json','BILL1_DECISION_v1.json'])
c['candidates']['stack']=c['candidates']['ON'];c['candidates']['ON']=dict(path=str(source),sha256=sha(source),mode=1)
for k in ('final_cost_rows','final_cost_metadata'):c.pop(k,None)
pins=[source,ready,tiny,V/'CACHE_UNIFORM64_v1.json',V/'ROOT_ONE_NATIVE_GRANT_v1.json',V/'uniform64_source_diff_v1.patch',D/'finite_uniform64_bill1_v1.py',D/'native_bill1_judge_astra_v1.py',D/'uniform64_source_contract_v1.py',D/'grade_bill1_v1.py',Path(__file__)]
c['source_pins'].update({str(p.resolve()):sha(p) for p in pins})
for kind in ('certificate','guard'):c['source_pins'][control[kind]]=control[kind+'_sha256']
cp=Path(control['certificate']);c['source_pins'][str(cp.with_name(cp.stem+'_pred.npz'))]=control['prediction_npz_sha256']
for path,digest in c['source_pins'].items():assert sha(path)==digest
write('PROTOCOL_REGISTERED_astra_v1.json',c)
for name in ('finite_uniform64_bill1_v1.py','native_bill1_judge_astra_v1.py','uniform64_source_contract_v1.py','grade_bill1_v1.py'):compile((D/name).read_text(),str(D/name),'exec')
print(json.dumps(dict(protocol_sha256=sha(D/'PROTOCOL_REGISTERED_astra_v1.json'),runner_sha256=sha(D/'finite_uniform64_bill1_v1.py'),judge_sha256=sha(D/'native_bill1_judge_astra_v1.py'),pins=len(c['source_pins']))))
