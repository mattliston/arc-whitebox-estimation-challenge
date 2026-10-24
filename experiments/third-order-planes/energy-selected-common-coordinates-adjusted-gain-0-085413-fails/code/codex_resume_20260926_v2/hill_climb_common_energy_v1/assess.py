from pathlib import Path
import hashlib,json,subprocess,sys
S=Path(__file__).resolve().parent;H=S.parent
p=json.loads((S/'followup_protocol.json').read_text())
candidate=p['candidate'];assert hashlib.sha256(Path(candidate['path']).read_bytes()).hexdigest()==candidate['sha256']
for block in ['train2','train6','extra8']:
    g=json.loads((S/(block+'_guard.json')).read_text())
    assert g['completed'] and g['exit_code']==0 and g['whole_group_retired'] and not g['remaining_rows'] and g['error'] is None and g['max_swap_growth_bytes']==0 and g['max_rss_bytes']<7500000000
parent=['hill_climb_star_projection32_train2_v1','hill_climb_star_projection32_train6_v1','hill_climb_star_projection32_extra8_candidate_v1']
child=['hill_climb_common_energy_train2_v1','hill_climb_common_energy_train6_v1','hill_climb_common_energy_extra8_v1']
def compare(name,left,right):
    path=S/(name+'.json')
    subprocess.run([sys.executable,'-B',str(H/'compare_native.py'),','.join(left),','.join(right),str(path.relative_to(H))],check=True)
    return json.loads(path.read_text())
r16=compare('train16_incremental',parent,child)
r14=compare('additional14_incremental',parent[1:],child[1:])
r8=compare('train8_vs_s31',['tail_cut_short_parent_train2_v1','last_join_shift_parent_train6_v1'],child[:2])
gates=dict(increment=r16['adjusted_pct']<=-.1,wins=r16['adjusted_wins']>=10,ci=r16['adjusted_ci95_pct'][1]<0,additional14=r14['adjusted_pct']<0,bank_wins=r8['adjusted_wins']>=6,bank_ci=r8['adjusted_ci95_pct'][1]<0,bank_max_regression=max(x['adjusted_pct'] for x in r8['per_network'])<=5)
passed=all(gates.values())
result=dict(passed=passed,stage='TRAIN8' if passed and r8['adjusted_pct']<=-2 else 'TRAIN16_COMPONENT',candidate=candidate,gates=gates,train2_gate_failed_preserved=False,extension_protocol='followup_protocol.json',incremental_adjusted_pct=r16['adjusted_pct'],incremental_wins=r16['adjusted_wins'],incremental_ci95=r16['adjusted_ci95_pct'],additional14_pct=r14['adjusted_pct'],bank_train8_adjusted_pct=r8['adjusted_pct'],bank_train8_wins=r8['adjusted_wins'],bank_train8_ci95=r8['adjusted_ci95_pct'],broad_validation_eligible=passed and r8['adjusted_pct']<=-2,worker_qualified=False,submitted=False)
with (S/'result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
