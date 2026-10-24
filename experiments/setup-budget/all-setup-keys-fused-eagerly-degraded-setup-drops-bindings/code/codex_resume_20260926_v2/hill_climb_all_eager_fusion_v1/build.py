"""Source-only all-existing-key eager setup variant. No numerical execution."""
from pathlib import Path
import ast, difflib, hashlib, json
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent
def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(name,x):
    p=D/name
    with p.open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2)+'\n')
    return pin(p)
parent=H/'hill_climb_scatter_forward_gather_v1/candidate.py'
assert pin(parent)['sha256']=='1056a1b79acedd1ab75bd711c4880b0507318db15448db48638e618baa7dcf4d'
old=parent.read_text()
before='    ops = _lvf_ops(kind, L, levels, chunks, groups[0]) if _LVF_EAGER[0] else None\n'
after='    ops = _lvf_ops(kind, L, levels, chunks, groups[0]) if (_LVF_EAGER[0] or _LVF_KEEP[0]) else None\n'
assert old.count(before)==1
new=old.replace(before,after);assert new.replace(after,before)==old;ast.parse(new)
candidate=write('candidate.py',new)
write('source_diff.patch',''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(parent),tofile='candidate.py')))
prior=H/'hill_climb_essential_fusion_v1';c=json.loads((prior/'commands.json').read_text())
base_tag=prior.name+'_forced_base_v1';prior_tag=prior.name+'_forced_candidate_v1'
evidence=[]
for tag in (base_tag,prior_tag):
    evidence.extend(pin(H/(tag+suffix)) for suffix in ('_setup.json','_census.json','_launcher.json'))
    evidence.extend(pin(H/f'{tag}_pred_{r}.bin') for r in (35,100))
evidence.extend(pin(prior/f) for f in ('forced_base_guard.json','forced_candidate_guard.json','protocol.json','build.json'))
base_rows=json.loads((H/(base_tag+'_census.json')).read_text())
nine_rows=json.loads((H/(prior_tag+'_census.json')).read_text())
measured=[]
for a,b in zip(base_rows,nine_rows):
    assert a['rank']==b['rank'] and a['prediction_sha256']==b['prediction_sha256'] and a['flops']==b['flops']
    assert a['requests']-b['requests']==870
    measured.append(dict(rank=a['rank'],requests_saved=870,base_residual_s=a['residual_s'],nine_residual_s=b['residual_s']))
protocol=write('protocol.json',dict(
    status='prospective resource-only extension selected by root after limited measured nine-key benefit',
    parent=pin(parent),candidate=candidate,prior_trial_evidence=evidence,prior_measured=measured,
    scope='All existing phase1 keys fuse when bound; no new keys, kernels, arithmetic, arenas or caches.',
    bind_keys=649,bind_order_unchanged=True,deadline_seconds=4.0,lazy_prediction_misses_unchanged=True,
    prospective_revision='Root supersedes earlier nine-key-only scope based on its measured 870-request improvement with small/noisy residual benefit. Earlier artifacts remain unchanged.',
    complete_setup_counts=dict(bound=687,deferred_lvfuse=0,eager_fused=649,total_fused=649,psdb_fused=9,pair_block_indices=1,read7_fused=4),
    ownership='Existing cache entries own the identical ops tuples; existing eviction and reset behavior retained. No new response-value state.',
    limitations=['Eager work may bind fewer keys before the unchanged deadline; actual coverage must be recorded.','Request reduction alone does not establish recovery of official runtime/residual failures.'],
    gates=dict(normal_setup_repetitions=3,setup_seconds_at_most=4.1,native_ranks=[35,100],all16_bytes_equal=True,flops_equal=True,
        adjusted_regression_pct_at_most=0,forced_phase2_skip_requests_strictly_lower=True,rss_bytes=7500000000,added_swap_bytes=0,
        own_boundary_required=True,residual_seconds_strictly_below=.4,projected_wall_seconds_at_most=105,cpu_agreement_pp=5),
    degraded_phase1_test='Pending independent agreement; harness-only fixedprefix128 proposed, not yet implemented or executed.',
    no_full40_serving_release_authority=True,models_executed=False))
build=write('build.json',dict(builder=pin(__file__),candidate=candidate,protocol=protocol,
    source_inverse=dict(parent=pin(parent),edits=[dict(before=before,after=after)],exact_inverse=True),
    ast_parsed=True,only_one_condition_changed=True,roster_deadline_and_other_functions_byte_identical=True))
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
driver=prior/'probe_driver.py'
commands={f'normal_setup_{i}':[str(native),'-B',str(driver),candidate['path'],'-',D.name+f'_normal_setup{i}_v1','normal'] for i in (1,2,3)}
commands['forced_candidate']=[str(native),'-B',str(driver),candidate['path'],'35,100',D.name+'_forced_candidate_v1','skip_phase2']
commands['normal_native2']=[str(native),'-B',str(H/'hill_climb_e1_ynt_psdb_d5_v1/native_probe.py'),candidate['path'],'35,100',D.name+'_train2_v1']
write('commands.json',dict(build=build,protocol=protocol,probe_driver=pin(driver),probe_client=pin(prior/'probe_client.py'),
    commands=commands,existing_forced_base=dict(census=pin(H/(base_tag+'_census.json')),setup=pin(H/(base_tag+'_setup.json')),guard=pin(prior/'forced_base_guard.json')),
    limits=c['limits'],root_executes_serially=True,
    gate='Same source-bound probe. Compare full16 bytes/FLOPs and requests against the actual saved unfused FG base. Real four-second phase1 cutoff unchanged. Record coverage and measured residual; no implied official repair.'))
print(json.dumps(dict(candidate=candidate,build=build,commands=pin(D/'commands.json'))))
