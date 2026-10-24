"""One prospectively fixed128-key setup schedule; source-only construction."""
from pathlib import Path
import ast,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent
AE=H/'hill_climb_all_eager_fusion_v1';NINE=H/'hill_climb_essential_fusion_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(name,x):
 p=D/name
 with p.open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2)+'\n')
 return pin(p)
parent=H/'hill_climb_scatter_forward_gather_v1/candidate.py';pp=pin(parent)
assert pp['sha256']=='1056a1b79acedd1ab75bd711c4880b0507318db15448db48638e618baa7dcf4d'
s=parent.read_text()
before='    ops = _lvf_ops(kind, L, levels, chunks, groups[0]) if _LVF_EAGER[0] else None\n'
after='    ops = _lvf_ops(kind, L, levels, chunks, groups[0]) if (_LVF_EAGER[0] or (_LVF_KEEP[0] and (kind, L, rows, cols) in _PB_SINGLE[:128])) else None\n'
assert s.count(before)==1;t=s.replace(before,after);assert t.replace(after,before)==s;ast.parse(t)
candidate=write('candidate.py',t)
write('source_diff.patch',''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile=str(parent),tofile='candidate.py')))
ns={'tuple':tuple,'len':len}
for node in ast.parse(s).body:
 if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('_PB_SINGLE','_PB_WALL_S') for x in node.targets):
  exec(compile(ast.Module(body=[node],type_ignores=[]),'<roster-only>','exec'),ns)
assert len(ns['_PB_SINGLE'])==len(set(ns['_PB_SINGLE']))==649 and ns['_PB_WALL_S']==4.0
evidence=[];rows={}
for role in ('base','candidate'):
 tag=AE.name+'_phase1_3s_'+role+'_v1'
 evidence.extend(pin(H/(tag+x)) for x in ('_setup.json','_census.json','_launcher.json'))
 evidence.extend(pin(H/f'{tag}_pred_{r}.bin') for r in (35,100))
 evidence.append(pin(AE/f'phase1_3s_{role}_guard_v1.json'))
 rows[role]=json.loads((H/(tag+'_census.json')).read_text())
failure=[]
for b,c in zip(rows['base'],rows['candidate']):
 assert b['rank']==c['rank'] and b['prediction_sha256']==c['prediction_sha256'] and b['flops']!=c['flops']
 failure.append(dict(rank=b['rank'],extra_flops=c['flops']-b['flops'],requests_saved=b['requests']-c['requests'],base_residual_s=b['residual_s'],candidate_residual_s=c['residual_s']))
protocol=write('protocol.json',dict(
 status='Prospective fixed128-key resource-only schedule; no model execution',parent=pp,candidate=candidate,
 selection='Root selects exactly first128 existing keys after nine-key small/noisy residual benefit and all-eager strict degraded FLOP gate failure. No other prefix/cutoff selected.',
 prior_all_eager_failure=failure,prior_failed_evidence=evidence,
 prefix_count=128,prefix_keys=ns['_PB_SINGLE'][:128],all649_bind_order_unchanged=True,source_deadline_s=4.0,
 same_existing_ops_kernels=True,same_arenas_and_cache_limits=True,lazy_prediction_misses_unchanged=True,
 coverage='Only keys actually bound while KEEP is true can fuse eagerly. All other existing keys retain ordinary phase2 behavior; partial setup is measured.',
 complete_normal_setup_counts=dict(bound=687,deferred_lvfuse=521,eager_prefix=128,total_fused=649,psdb_fused=9,pair_block_indices=1,read7_fused=4),
 degraded_probe=pin(AE/'protocol_degraded_v1.json'),
 gates=dict(normal_setup_repetitions=3,normal_setup_seconds_at_most=4.1,degraded_phase1_seconds=3.0,restored_deadline_s=4.0,
    ranks=[35,100],all16_bytes_equal=True,flops_equal=True,adjusted_regression_pct_at_most=0,requests_strictly_lower_each_rank=True,
    rss_bytes=7500000000,added_swap_bytes=0,guard_seconds=240),
 no_full40_serving_release_authority=True,no_parameter_grid=True,limitations='A request reduction is not itself an official reliability fix; no failed gate is relaxed.'))
build=write('build.json',dict(builder=pin(__file__),candidate=candidate,protocol=protocol,
 source_inverse=dict(parent=pp,edits=[dict(before=before,after=after)],exact_inverse=True),
 ast_parsed=True,only_one_condition_changed=True,prefix_count=128,bind_order_unchanged=True))
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python';normal=NINE/'probe_driver.py';degraded=AE/'probe_driver_degraded_v1.py'
commands={f'normal_setup_{i}':[str(native),'-B',str(normal),candidate['path'],'-',D.name+f'_normal_setup{i}_v1','normal'] for i in (1,2,3)}
commands['degraded_candidate']=[str(native),'-B',str(degraded),candidate['path'],'35,100',D.name+'_phase1_3s_candidate_v1','phase1_3s']
tag=AE.name+'_phase1_3s_base_v1'
cp=write('commands.json',dict(build=build,protocol=protocol,
 probes=[pin(normal),pin(NINE/'probe_client.py'),pin(degraded),pin(AE/'probe_client_degraded_v1.py')],commands=commands,
 actual_degraded_base=dict(census=pin(H/(tag+'_census.json')),setup=pin(H/(tag+'_setup.json')),guard=pin(AE/'phase1_3s_base_guard_v1.json')),
 limits=dict(normal_setup_seconds=60,degraded_candidate_seconds=240,rss_bytes=7500000000,added_swap_bytes=0),
 root_executes_serially=True,no_new_harness=True))
print(json.dumps(dict(candidate=candidate,build=build,protocol=protocol,commands=cp)))
