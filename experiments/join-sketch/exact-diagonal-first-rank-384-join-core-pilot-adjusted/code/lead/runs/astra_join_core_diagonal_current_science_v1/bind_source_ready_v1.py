"""Register the fixed read-energy featujcdp source and cached current controls."""
from pathlib import Path
import hashlib,json,datetime,sys
D=Path(__file__).resolve().parent;O=D.parents[2];sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
ready=Path(sys.argv[1]).resolve();peer=D/'SOURCE_PEER_PASS_v1.json';r=read(ready);pr=read(peer);assert pr['passed']
cp=Path(pr['candidate_path']);cs=pr['candidate_sha256'];assert sha(cp)==cs
S=O/'lead/runs/astra_hermite_group_fusion_final_science_v1';donor=read(S/'PROTOCOL_REGISTERED_astra_v1.json')
controls=[]
for phase in ('PILOT','REMAINING'):
 owner=read(S/(phase+'_RETIREMENT_astra_v1.json'));assert owner['approved']and owner['all_actual_exits_observed']and owner['all_owned_PID_PPID_PGID_absent'];controls.extend(owner['evaluations'])
controls.sort(key=lambda x:x['seed']);assert [x['seed']for x in controls]==list(range(7103248,7103272))
parity=O/'lead/runs/astra_hf_view_repair_native_ten_v1/NATIVE_TEN_PARITY_v1.json';v=read(parity);assert v['passed']and v['all16_byte_identical']and v['all_flops_identical']
pins=dict(donor['source_pins']);pins.update(pr['evidence_sha256'])
authority=D/'ROOT_DIRECT_PILOT_AUTHORITY_v1.json';pins[str(authority)]=sha(authority)
for p in [ready,peer,cp,D/'finite_jcdp_astra_v1.py',D/'native_jcdp_judge_astra_v1.py',D/'grade_jcdp_retired_phase_v1.py',D/'jcdp_source_contract_v1.py',Path(__file__).resolve(),parity,S/'PILOT_RETIREMENT_astra_v1.json',S/'REMAINING_RETIREMENT_astra_v1.json',S/'ALL24_DECISION_astra_v1.json']:
 pins[str(p)]=sha(p)
for e in controls+donor['reference_control_reuse']:
 for key in ('certificate','guard'):assert sha(e[key])==e[key+'_sha256'];pins[e[key]]=e[key+'_sha256']
 p=Path(e['certificate']);pp=p.with_name(p.stem+'_pred.npz');assert sha(pp)==e['prediction_npz_sha256'];pins[str(pp)]=e['prediction_npz_sha256']
c=dict(status='REGISTERED',purpose='jcdp_primary_pilot8_fresh16_oos8',approved_by='root',approved=True,reviewed_by='root',executed_by='bigger_gain_hunt',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),identity_groups=donor['identity_groups'],phase_seeds=donor['phase_seeds'],shards=[0,1,2,3],native_calls_max=24,research_slots=4,candidates=dict(ON=dict(path=str(cp),sha256=cs,mode=1),stack=dict(donor['candidates']['ON']),reference=dict(donor['candidates']['reference'])),source_pins=pins,truth_metadata=donor['truth_metadata'],stack_control_reuse=controls,reference_control_reuse=donor['reference_control_reuse'],rule=dict(pilot='all8 adjusted meanlog strictly negative vs currentbest PRIMARY19fc/equivalent1c913; e406 report-only',remaining='Only after all8 actualretired and PRIMARY PASS; exactly16 ON-only cachedcontrols',final='all24 AND OOS8 adjusted meanlog strictlynegative versus currentbest PRIMARY; fresh16 and e406 reported only',source='One fixed pair-aware missing young-tail energy on L7–14 featujcdp only, vectorized complete signed partners, exact pair-local diagonal exclusion; original physical read/covariance untouched locally; no tuning; first10 exactPRIMARY; no newparents/truths'),resource_policy=dict(entry_free_percent_min=60,entry_counted_max_exclusive=7,live_free_percent_min=50,live_counted_max=10,native_rss_and_footprint_max=7500000000,threads_each=1,native_nice=10),source_readiness=dict(path=str(ready),sha256=sha(ready)),control_equivalence=dict(path=str(parity),sha256=sha(parity),scope='19fc prediction/FLOP proxy for submitted1c913, source-transparent view-only repair and accepted native-ten exact parity'),runtime_not_transferred=True)
p=D/'PROTOCOL_REGISTERED_astra_v1.json';p.open('x').write(json.dumps(c,indent=2)+'\n')
g=dict(approved=True,reviewed_by='root',executed_by='bigger_gain_hunt',conditional_root_authorization=read(authority)['conditional_root_authorization'],protocol_sha256=sha(p),runner_sha256=sha(D/'finite_jcdp_astra_v1.py'),candidate_sha256=cs,native_calls_max=24,research_slots=4,no_retry=True)
gp=D/'ROOT_PILOT_AND_CONDITIONAL_REMAINING_GRANT_v1.json';gp.open('x').write(json.dumps(g,indent=2)+'\n');print(json.dumps(dict(candidate=cs,protocol=sha(p),grant=sha(gp),runner=g['runner_sha256'])))
