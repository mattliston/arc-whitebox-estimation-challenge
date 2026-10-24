"""Freeze the opus U64 ten-screen (9 new native calls vs exact 2b108 native-ten controls). No model calls."""
from pathlib import Path
import hashlib,json,datetime
D=Path(__file__).resolve().parent;R=D.parent;A=R/'astra_uniform64_bill1_v1';H=R/'astra_smirnov_final_native_ten_v1';V=R/'astra_read256_inner_cost_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    p=D/name
    with p.open('x') as f:
        if isinstance(value,str):f.write(value)
        else:json.dump(value,f,indent=2);f.write('\n')
    return p
SEEDS=list(range(7103249,7103258))
source=V/'candidate_uniform64_read_HELD_v1.py';assert sha(source)=='24b13a6763b83a9bf75186cc9aad5d4d424c5a2e9516c37b8676c839c964b9f4'
# judge: same body, purpose/phase/seed allowlist only
j=(A/'native_bill1_judge_astra_v1.py').read_text()
for a,b in [("science_protocol['purpose']=='uniform64_read_fixed_bill1'","science_protocol['purpose']=='opus_u64_screen_v1'"),
            ("assert set(phase_seeds)=={'bill'} and len(phase_seeds['bill'])==1","assert set(phase_seeds)=={'screen'} and len(phase_seeds['screen'])==9"),
            ("allowed=phase_seeds['bill'];assert allowed==[7103248]","allowed=phase_seeds['screen'];assert allowed==list(range(7103249,7103258))"),
            ("uniform64_source_contract_v1","u64_screen_contract_v1")]:
    assert j.count(a)>=1,a;j=j.replace(a,b)
write('native_u64_screen_judge_v1.py',j)
# contract: accepted 2b108 final contract + actual-scope U64 gate (7-event L10-14 scope per ACTUAL_COVERAGE_ACCEPTANCE_v2)
k=(H/'final_source_contract_v1.py').read_text()+'''
# Actual uniform64 scope (root-accepted coverage correction): old-bank vt reads in L10-14 only.
_uniform64_parent_gate=gate
def gate(row):
    _uniform64_parent_gate(row)
    st=row['stats']['_ASTRA_READ64_STATS']
    assert st['retired'] is True
    events=st['events']
    assert events and all(len(e)==6 and 10<=e[0]<=14 and e[3] in (320,384) and e[4]==e[3]-64 and e[5]==256 and e[1]>0 and e[2]>0 for e in events)
    assert row['sm40_lifecycle_before']['prebind_count']==14
    return [(e[0],e[3]) for e in events]
'''
write('u64_screen_contract_v1.py',k)
r=(A/'finite_uniform64_bill1_v1.py').read_text()
for a,b in [('"""One uniform64 read candidate, compared against cached current2b108 after retirement."""','"""Opus U64 ten-screen: candidate vs exact cached 2b108 native-ten controls, two shards."""'),
            ("q.guard.JUDGE=D/'native_bill1_judge_astra_v1.py'","q.guard.JUDGE=D/'native_u64_screen_judge_v1.py'"),
            ("c['purpose']=='uniform64_read_fixed_bill1'","c['purpose']=='opus_u64_screen_v1'"),
            ("assert phase == 'bill' and shard == 0","assert phase == 'screen' and shard in (0,1)"),
            ("grant=json.loads((D.parent/'astra_read256_inner_cost_v1/ROOT_ONE_NATIVE_GRANT_v1.json').read_text())","grant=json.loads((D/'OPUS_LEAD_GRANT_v1.json').read_text())"),
            ("grant['source_sha256']==c['candidates']['ON']['sha256'] and grant['seed']==7103248","grant['source_sha256']==c['candidates']['ON']['sha256'] and grant['seeds']==c['phase_seeds']['screen']"),
            ("assert grant['maximum_native_calls']==1 and","assert grant['maximum_native_calls']==9 and"),
            ("seeds=c['phase_seeds'][phase][shard::1]","seeds=c['phase_seeds'][phase][shard::2]"),
            ("h=D/f'run_uniform64_{phase}_s{shard}_v1'","h=D/f'run_u64screen_{phase}_s{shard}_v1'"),
            ("from uniform64_source_contract_v1 import gate","from u64_screen_contract_v1 import gate"),
            ("gate(a['row'])","scope=gate(a['row'])"),
            ("evaluations.append(dict(role='ON',seed=seed,**item))","evaluations.append(dict(role='ON',seed=seed,scope=scope,**item))")]:
    assert r.count(a)==1,a;r=r.replace(a,b)
write('finite_u64_screen_v1.py',r)
controls=[]
for shard in (0,1):controls.extend(json.loads((H/f'run_final_parity_s{shard}_v1/READY_v1.json').read_text())['evaluations'])
ctl=[next(e for e in controls if e['seed']==s) for s in SEEDS]
for e in ctl:assert json.loads(Path(e['certificate']).read_text())['sha256']=='2b10843a76fc033325ceee9dd30703eff29a1fa535cdc094c5feef72023a0c3c'
grant=dict(approved=True,reviewed_by='root',root='opus lead (sole lead after Astra; directive 2026-10-16 step1)',source_sha256=sha(source),seeds=SEEDS,maximum_native_calls=9,max_concurrent_native=2,no_fire_authority=True)
write('OPUS_LEAD_GRANT_v1.json',grant)
c=json.loads((A/'PROTOCOL_REGISTERED_astra_v1.json').read_text())
c.update(status='REGISTERED',purpose='opus_u64_screen_v1',approved_by='root',executed_by='opus-u64',phase_seeds={'screen':SEEDS},identity_groups={'screen':SEEDS},shards=[0,1],research_slots=2,native_calls_max=9,stack_control_reuse=ctl,registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
         rule=dict(scope='Unchanged U64 candidate 24b13a67 (actual 7-event L10-14 scope). No rank/dial/grid selection.',identity='First7 prediction bytes/layerMSE exact vs cached 2b108 native-ten; all16 finite.',decision='Lead decides on ten-screen (with graded 7103248): mean net log<0 and not single-network carried.'),
         run_names=['run_u64screen_screen_s0_v1','run_u64screen_screen_s1_v1'],decision_names=['SCREEN10_DECISION_v1.json'])
c.pop('remaining_pilot_seeds',None)
pins=[source,D/'finite_u64_screen_v1.py',D/'native_u64_screen_judge_v1.py',D/'u64_screen_contract_v1.py',D/'OPUS_LEAD_GRANT_v1.json',Path(__file__)]
for old in [str(A/n) for n in ('finite_uniform64_bill1_v1.py','native_bill1_judge_astra_v1.py','uniform64_source_contract_v1.py','grade_bill1_v1.py','prepare_bill1_v1.py','prepare_bill1_v2.py')]:c['source_pins'].pop(old,None)
c['source_pins'].update({str(p.resolve()):sha(p) for p in pins})
for e in ctl:
    for kind in ('certificate','guard'):c['source_pins'][e[kind]]=e[kind+'_sha256']
    cp=Path(e['certificate']);c['source_pins'][str(cp.with_name(cp.stem+'_pred.npz'))]=e['prediction_npz_sha256']
for path,digest in c['source_pins'].items():assert sha(path)==digest,path
write('PROTOCOL_REGISTERED_opus_v1.json',c)
for n in ('finite_u64_screen_v1.py','native_u64_screen_judge_v1.py','u64_screen_contract_v1.py'):compile((D/n).read_text(),str(D/n),'exec')
print(json.dumps(dict(protocol_sha256=sha(D/'PROTOCOL_REGISTERED_opus_v1.json'),runner=sha(D/'finite_u64_screen_v1.py')[:8],judge=sha(D/'native_u64_screen_judge_v1.py')[:8],pins=len(c['source_pins']))))
