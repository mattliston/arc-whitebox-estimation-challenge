"""Minimal resource-only native screen, saved FG controls; no models/arrays."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent
FG=H/'hill_climb_scatter_forward_gather_v1';AE=H/'hill_climb_all_eager_fusion_v1'
NATIVE=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def write(name,x):
 p=D/name
 with p.open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2)+'\n')
 return pin(p)
candidate=pin(D/'candidate.py');assert candidate['sha256']=='f9f4790b3c65f730404425e8923c767dc2f895fdcc1e14f14cc649286ad3b69f'
fgsource=pin(FG/'candidate.py');assert fgsource['sha256']=='1056a1b79acedd1ab75bd711c4880b0507318db15448db48638e618baa7dcf4d'
evidence=[];normal=[]
def clean_guard(p,limit):
 g=read(p);assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None
 assert not g['remaining_rows'] and g['max_swap_growth_bytes']==0 and g['rss_trigger_bytes']==7500000000
 assert 0<g['elapsed_s']<=limit and 0<g['max_rss_bytes']<7500000000
 evidence.append(pin(p));return g
for i in (1,2,3):
 p=H/f'{D.name}_normal_setup{i}_v1_setup.json';s=read(p)
 assert s['candidate_sha256']==candidate['sha256'] and s['mode']=='normal' and s['phase1_deadline']==4.0
 assert s['setup_wall_s']<=4.1 and s['fused_entries']==649 and all(x['fused'] for x in s['essential'])
 assert [x[1] for x in s['setup_log'] if x[0]=='lvfuse']==[521]
 clean_guard(D/f'normal_setup{i}_guard.json',60);evidence.append(pin(p));normal.append(s['setup_wall_s'])
base_tag=AE.name+'_phase1_3s_base_v1';child_tag=D.name+'_phase1_3s_candidate_v1'
b=read(H/(base_tag+'_census.json'));c=read(H/(child_tag+'_census.json'))
assert [r['rank'] for r in b]==[r['rank'] for r in c]==[35,100]
per=[]
for x,y in zip(b,c):
 assert x['flops']==y['flops'] and x['prediction_sha256']==y['prediction_sha256'] and y['requests']<x['requests']
 assert x['prediction_shape']==y['prediction_shape']==[16,1024] and x['nonfinite']==y['nonfinite']==0
 for tag in (base_tag,child_tag):evidence.append(pin(H/f"{tag}_pred_{x['rank']}.bin"))
 assert (H/f"{base_tag}_pred_{x['rank']}.bin").read_bytes()==(H/f"{child_tag}_pred_{x['rank']}.bin").read_bytes()
 per.append(dict(rank=x['rank'],flops=x['flops'],requests_saved=x['requests']-y['requests'],base_residual_s=x['residual_s'],candidate_residual_s=y['residual_s']))
for tag,source in ((base_tag,fgsource),(child_tag,candidate)):
 s=read(H/(tag+'_setup.json'));assert s['candidate_sha256']==source['sha256'] and s['phase1_deadline']==4.0
 assert s['phase1_simulation']['requested_phase1_budget_s']==3.0 and s['phase1_simulation']['restored_deadline_s']==4.0
 evidence.extend(pin(H/(tag+x)) for x in ('_setup.json','_census.json','_launcher.json'))
clean_guard(AE/'phase1_3s_base_guard_v1.json',240);clean_guard(D/'degraded_candidate_guard.json',240)
cheap=write('cheap_resource_gate_v1.json',dict(candidate=candidate,base=fgsource,passed=True,normal_setup_seconds=normal,
    all16_bytes_equal=True,flops_equal=True,requests_lower_each_rank=True,per_network=per,evidence=evidence,
    interpretation='Pass of the predeclared cheap mechanism gate only; no official reliability, full40, serving or release claim.'))
parent_engine=H/'native_screen_engine_v1/screen.py';old=parent_engine.read_text()
assert pin(parent_engine)['sha256']=='1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
edits=[dict(before="    result = metrics(rows['base'], rows['candidate']); limits = c['stages'][str(n)]['gates']\n",
 after="    assert {r['rank']: r['flops'] for r in rows['candidate']} == {r['rank']: r['flops'] for r in rows['base']}, 'resource-only per-network FLOP equality'\n    result = metrics(rows['base'], rows['candidate']); limits = c['stages'][str(n)]['gates']\n"),
 dict(before='        all16_predictions_finite=True, configured_parent_bytes_exact=bool(c[\'parity_layers\']),\n',
 after='        per_network_flops_equal=True, all16_predictions_finite=True, configured_parent_bytes_exact=bool(c[\'parity_layers\']),\n')]
s=old
for e in edits:assert s.count(e['before'])==1;s=s.replace(e['before'],e['after'])
inv=s
for e in reversed(edits):inv=inv.replace(e['after'],e['before'])
assert inv==old;ast.parse(s)
engine=write('screen.py',s)
inverse=write('engine_inverse.json',dict(engine=engine,parent=pin(parent_engine),edits=edits,exact_inverse=True))
write('screen.diff',''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile=str(parent_engine),tofile='screen.py')))
parent_config=read(FG/'native_screen_continuation.json');base=copy.deepcopy(parent_config['arms']['candidate'])
assert base['source']==fgsource
for size in ('2','6'):
 block=base['blocks'][size];folder=H/block['tag']
 block['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(r):pin(folder/f'pred_{r}.npy') for r in block['ranks']})
child=copy.deepcopy(base);child['source']=candidate;child['setup']['counts']['lvfuse']=521
for size in ('2','6'):
 block=child['blocks'][size];block.pop('artifacts');block['tag']=D.name+f'_train{size}_v1';block['guard']=str(D/f'native{size}_guard.json')
gates=dict(adjusted_pct_at_most=0.0,strict_adjusted=False,wins_min=0,ci_upper_below=None,max_per_network_regression_pct=0.0,additional6_gain_required=False)
protocol=write('native_resource_protocol_v1.json',dict(candidate=candidate,comparator=fgsource,cheap_gate=cheap,
 source_build=pin(D/'build.json'),engine_inverse=inverse,resource_only=True,
 interpretation='No score gain is requested or claimed: all16 prediction bytes, per-rank FLOPs and resulting scores must equal FG. Original resource limits remain.',
 setup_counts=dict(bound=687,lvfuse_deferred=521,eager_prefix=128,total_fused=649),
 native2_and8_gates=gates,parity_layers=list(range(16)),own_normal_boundary_required=True,
 full40_admission_requires=['actual degraded cheap gate pass','own normal N2/N8 all16/FLOP equality','own normal boundary pass'],
 full40_bank_gain_pct=1.0,fg_local_reference_gain_pct=1.38916,new_actual_full40_required=True,no_release_authority=True))
c=dict(schema='native-screen-v1',prospective_authorization=True,future_only=True,raw_error_veto=False,engine=engine,
 provenance=[pin(__file__),pin(D/'build.json'),pin(D/'protocol.json'),cheap,protocol,inverse,pin(D/'independent_source_review_v1.json'),pin(FG/'native_screen_continuation.json')],
 arms=dict(candidate=child,base=base,parity_parent=copy.deepcopy(base)),parity_layers=list(range(16)),
 stages={str(n):dict(result=str(D/f'native{n}_assessment.json'),assessment_guard=str(D/f'native{n}_assess_guard.json'),assessment_limits=dict(seconds=120,rss_bytes=7500000000),gates=copy.deepcopy(gates)) for n in (2,8)},
 boundary_before8=copy.deepcopy(parent_config['boundary_before8']),
 prospective_scope='native2_only_until_own_boundary_continuation',resource_protocol=protocol,exact_flops_required=True,
 interpretation='Prefix128 setup scheduling; resource-only equality against actual FG2+6. Deferred521 and eager128 are distinct. No current release claim.')
# Existing assessor pin permits normal N2 configure, but nonexistent own receipt blocks N8.
c['boundary_before8']['receipt']=str(D/'boundary_assessment.json')
c['boundary_before8']['assessment_guard']=str(D/'boundary_assess_guard.json')
config=write('native_screen.json',c)
commands={f'native{n}_assessment':[str(NATIVE),'-B',engine['path'],config['path'],config['sha256'],str(n)] for n in (2,8)}
for n in (2,6):
 block=child['blocks'][str(n)];commands[f'native{n}']=[str(NATIVE),'-B',block['driver']['path'],candidate['path'],','.join(map(str,block['ranks'])),block['tag']]
cp=write('native_commands_v1.json',dict(config=config,resource_protocol=protocol,commands=commands,
 native_guards={str(n):child['blocks'][str(n)]['guard'] for n in (2,6)},native_limits={str(n):child['blocks'][str(n)]['limits'] for n in (2,6)},
 assessment_limits=dict(seconds=120,rss_bytes=7500000000),native6_and8_held_until_own_boundary_continuation=True))
print(json.dumps(dict(config=config,engine=engine,engine_inverse=inverse,cheap_gate=cheap,protocol=protocol,commands=cp)))
