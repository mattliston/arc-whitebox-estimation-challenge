"""Freeze one paired estimator and fixed independent24; never executes models."""
from pathlib import Path
import ast,datetime,hashlib,json
D=Path(__file__).resolve().parent;R=D.parent;H=R/'astra_smirnov_final_native_ten_v1';V=R/'astra_smirnov_unseen_shapes_v1';A=R/'astra_smirnov_fresh3816_proposal_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):
 p=D/name
 with p.open('x')as f:
  if isinstance(v,str):f.write(v)
  else:json.dump(v,f,indent=2);f.write('\n')
 return p
source=V/'candidate_smirnov_unseen_table_R7_HELD_v2.py';parent=R/'astra_smirnov_current24_virtual_v1/candidate_smirnov_arena_RPE_SPLIT_COV_HELD_v15.py'
assert sha(source)=='b7090c35a185309c73633d66dbaa06d48b2d173c3f6b25ff941a54c224b2ba04'
assert sha(parent)=='2b10843a76fc033325ceee9dd30703eff29a1fa535cdc094c5feef72023a0c3c'
alloc=A/'ROOT_REASSIGN_FIXED24_v1.json';al=json.loads(alloc.read_text());assert al['approved']and al['seeds']==list(range(7103816,7103840))
judge=(H/'native_final_judge_astra_v1.py').read_text().replace('smirnov_final_exact_native_ten','smirnov_table_r7_fixed_paired24').replace('final_source_contract_v1','table_r7_source_contract_v1')
judge=judge.replace("assert set(phase_seeds)=={'parity'} and len(phase_seeds['parity'])==10\nallowed=phase_seeds['parity'];assert len(set(allowed))==10", "assert set(phase_seeds)=={'pilot','remaining'}\nallowed=phase_seeds['pilot']+phase_seeds['remaining'];assert allowed==list(range(7103816,7103840))")
write('native_table_r7_judge_v1.py',judge)
contract=(H/'final_source_contract_v1.py').read_text();ct=ast.parse(contract);node=next(n for n in ct.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='PLAN_KEYS')
tree=ast.parse(source.read_text());plans=next(ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_ASTRA_SM40_PLANS'for t in n.targets));assert len(plans)==167
lines=contract.splitlines(keepends=True);contract=''.join(lines[:node.lineno-1])+'PLAN_KEYS=frozenset('+repr(tuple(plans))+')\n'+''.join(lines[node.end_lineno:]);write('table_r7_source_contract_v1.py',contract)
c=json.loads((H/'PROTOCOL_REGISTERED_astra_v1.json').read_text());metadata=json.loads((A/'CACHED24_PROPOSAL_v1.json').read_text())['truth_metadata'];assert [r['seed']for r in metadata]==al['seeds']
for r in metadata:assert sha(r['path'])==r['sha256']
c.update(status='REGISTERED',approved_by='root',purpose='smirnov_table_r7_fixed_paired24',registered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates={'parent':dict(path=str(parent),sha256=sha(parent),mode=1,prebound=14),'ON':dict(path=str(source),sha256=sha(source),mode=1,prebound=9)},phase_seeds={'pilot':list(range(7103816,7103824)),'remaining':list(range(7103824,7103840))},identity_groups={'fresh':list(range(7103816,7103832)),'oos':list(range(7103832,7103840))},truth_metadata=metadata,shards=[0,1,2,3],research_slots=4,native_calls_max=48,pilot_native_calls=16,remaining_native_calls=32,role_order='Even seed parent then ON; odd seed ON then parent.',allocation=dict(path=str(alloc),sha256=sha(alloc)),rule=dict(pilot='After all8 pairs and actual retirement, negative mean adjusted log ratio vs current2b108 permits unchanged remaining16.',final='All24 and OOSlast8 adjusted mean log ratios must both be strictly negative.',metric='100*log(MSE_ON/MSE_parent)+100*log(max(flops_ON/2**41,.1)/max(flops_parent/2**41,.1))',no_tuning=True,no_release_authority=True),implementation='All16 finite; first1 prediction bytes/layerMSE exact; current ownership/physics and per-source Smirnov/R7 lifecycle; rank schedule unchanged.',initial_slots='Launch at most3 workers while accuracy baseline capture owns1; fourth only after its actual retirement or root clear.')
for k in ('stack_control_reuse','primary_control_reuse','reference_control_reuse','final_cost_rows','final_cost_metadata','remaining_pilot_seeds','no_followup_authorization'):c.pop(k,None)
pins=[source,parent,alloc,A/'S60_STAGE2_ABSENCE_v1.json',V/'STACK_SOURCE_HELD_v2.json',V/'PREFLIGHT_TABLE_R7_v2.json',V/'UNSEEN32_PRICE_v1.json',V/'LARGER_ACCURACY_STACK_PEER_v1.json',R/'astra_r7_firstread_reallocation_v1/TINY_SOURCE_TARIFF_v2.json',R/'astra_r7_firstread_reallocation_v1/SOURCE_HELD_v3.json',D/'native_table_r7_judge_v1.py',D/'table_r7_source_contract_v1.py',D/'finite_table_r7_v1.py',D/'grade_table_r7_v1.py',Path(__file__)]
c['source_pins'].update({str(p.resolve()):sha(p)for p in pins});c['source_pins'].update({r['path']:r['sha256']for r in metadata})
for path,digest in c['source_pins'].items():assert sha(path)==digest
write('PROTOCOL_REGISTERED_astra_v1.json',c)
for name in ('native_table_r7_judge_v1.py','table_r7_source_contract_v1.py','finite_table_r7_v1.py','grade_table_r7_v1.py'):compile((D/name).read_text(),str(D/name),'exec')
print(json.dumps(dict(protocol_sha256=sha(D/'PROTOCOL_REGISTERED_astra_v1.json'),runner_sha256=sha(D/'finite_table_r7_v1.py'),judge_sha256=sha(D/'native_table_r7_judge_v1.py'),pins=len(c['source_pins']))))
