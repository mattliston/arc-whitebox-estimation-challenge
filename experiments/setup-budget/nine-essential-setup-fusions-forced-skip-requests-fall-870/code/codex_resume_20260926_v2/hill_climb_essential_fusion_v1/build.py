"""Source-only, one setup scheduling edit; no arrays or model execution."""
from pathlib import Path
import ast, difflib, hashlib, json

D = Path(__file__).resolve().parent
H = D.parent
P = H.parent
ROOT = P.parents[3]
def pin(p):
    p = Path(p).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
def write(name, value):
    p = D/name
    with p.open('x') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,indent=2)+'\n')
    return pin(p)
parent = H/'hill_climb_scatter_forward_gather_v1/candidate.py'
assert pin(parent)['sha256']=='1056a1b79acedd1ab75bd711c4880b0507318db15448db48638e618baa7dcf4d'
s = parent.read_text()
keys = ((0,3,1024,1024),(1,3,1024,32),(2,3,1024,32),
        (0,3,32,1024),(1,3,1024,1024),(2,3,32,1024),
        (0,3,1024,32),(1,3,32,1024),(2,3,1024,1024))
before = '    ops = _lvf_ops(kind, L, levels, chunks, groups[0]) if _LVF_EAGER[0] else None\n'
after = ('    # Essential star keys fuse during existing phase 1; lazy prediction misses stay unchanged.\n'
         '    ops = _lvf_ops(kind, L, levels, chunks, groups[0]) if (_LVF_EAGER[0] or\n'
         '        (_LVF_KEEP[0] and (kind, L, rows, cols) in '+repr(keys)+')) else None\n')
assert s.count(before)==1
t=s.replace(before,after); assert t.replace(after,before)==s
ast.parse(t)
candidate=write('candidate.py',t)
write('source_diff.patch',''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile=str(parent),tofile='candidate.py')))
# Evaluate only literal/order assignments needed for the existing bind roster.
ns={'tuple':tuple,'len':len}
for node in ast.parse(s).body:
    if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id in ('_PB_SINGLE','_PB_WALL_S') for x in node.targets):
        exec(compile(ast.Module(body=[node],type_ignores=[]),'<roster-only>','exec'),ns)
roster=ns['_PB_SINGLE']; assert len(roster)==649 and len(set(roster))==649
assert [roster.index(k) for k in keys]==list(range(40,49)) and ns['_PB_WALL_S']==4.0
protocol=write('protocol.json',dict(
    status='prospective resource-only trial; no execution or release claim',
    parent=pin(parent),candidate=candidate,essential_keys=keys,
    bind_order_unchanged=True,deadline_seconds=4.0,essential_zero_based_positions=list(range(40,49)),
    operation_identity='Only existing fused-view tuple construction is scheduled earlier. Kernel arithmetic, routing, arena, capacities and prediction-lazy misses unchanged.',
    request_derivation=dict(star_factor_products=15*3,covariance_star_products=14*3,total_products=87,
        requests_saved_each_when_unfused=dict(A=3,B=3,C=4),conditional_network_requests_saved=870,
        eager_free_view_requests=dict(A=3*7,B=3*9,C=3*9,total=75)),
    limitation='Coverage requires phase 1 to reach key49. No guarantee below that cutoff; actual setup wall may change. Worker concentration is not proof of host cause.',
    complete_setup_counts=dict(bound=687,lvfuse_deferred=640,essential_fused=9,total_fused=649,psdb_fused=9,pair_block_indices=1,read7_fused=4),
    gates=dict(normal_setup_repetitions=3,setup_seconds_at_most=4.1,native_ranks=[35,100],all16_bytes_equal=True,
        flops_equal=True,adjusted_regression_pct_at_most=0,forced_phase2_skip_requests_strictly_lower=True,
        forced_phase2_skip_same_phase1_deadline=True,rss_bytes=7500000000,added_swap_bytes=0,
        own_boundary_required=True,residual_seconds_strictly_below=.4,projected_wall_seconds_at_most=105,cpu_agreement_pp=5),
    no_long_validation_or_submission_authority=True,simulation_is_separate_from_candidate=True,
    official_failure_evidence=pin(H/'official_s35_worker_concentration_summary_v1.json')))
build=write('build.json',dict(builder=pin(__file__),candidate=candidate,protocol=protocol,
    source_inverse=dict(parent=pin(parent),edits=[dict(before=before,after=after)],exact_inverse=True),
    ast_parsed=True,roster_identical=True,bind_order=list(roster),models_executed=False))

# Separate controlled boundary harness, preserving the existing real phase-1 clock.
client_parent=H/'census_boundary_setup_snapshot_v1.py'
c=client_parent.read_text()
c=c.replace('HERE = Path(__file__).resolve().parent; P2 = HERE.parent','STUDY = Path(__file__).resolve().parent; HERE = STUDY.parent; P2 = HERE.parent')
c=c.replace("[int(x) for x in sys.argv[2].split(',')]", "([] if sys.argv[2]=='-' else [int(x) for x in sys.argv[2].split(',')])")
anchor='by_site = collections.Counter(); by_func = collections.Counter(); total = [0]; active = [False]'
inject='''mode = sys.argv[4]
assert mode in ('normal','skip_phase2')
if mode == 'skip_phase2':
    def phase1_only(t0):
        mod._LVF_KEEP[0] = True
        try:
            mod._LVF_OLD_PB_BIND(t0)
        finally:
            mod._LVF_KEEP[0] = False
            mod._LVF_PENDING.clear()
            mod._PB_LOG.append(('forced_phase2_unavailable',0,round(mod._pb_time.monotonic()-t0,3)))
            mod._BH_HOLD.clear()
    mod._KSF_INNER = phase1_only
'''
assert c.count(anchor)==1;c=c.replace(anchor,inject+anchor)
setup_anchor='KERNEL = ("_packed_product_cached", "_packed_frame", "_exact_view", "_a_frame", "_b_frame", "_o_frame")'
capture='''import hashlib
_essential = ESSENTIAL_KEYS
_coverage = [dict(key=k, bound=dict.get((mod._IPT_A,mod._IPT_B,mod._IPT_C)[k[0]],tuple(k[1:])) is not None,
    fused=(dict.get((mod._IPT_A,mod._IPT_B,mod._IPT_C)[k[0]],tuple(k[1:])) or (None,None,None,None))[3] is not None) for k in _essential]
_setup = dict(candidate=str(cand),candidate_sha256=hashlib.sha256(cand.read_bytes()).hexdigest(),mode=mode,
    setup_wall_s=time.time()-t_s,setup_requests=total[0],setup_log=[list(x) for x in mod._PB_LOG],
    phase1_deadline=mod._PB_WALL_S,essential=_coverage,
    fused_entries=sum(e[3] is not None for cache in (mod._IPT_A,mod._IPT_B,mod._IPT_C) for e in cache.values()))
with (HERE/f'{tag}_setup.json').open('x') as _fp: json.dump(_setup,_fp,indent=2)
assert mod._PB_WALL_S==4.0 and all(r['bound'] for r in _coverage), 'phase1 did not reach all nine keys'
'''.replace('ESSENTIAL_KEYS',repr(keys))
assert c.count(setup_anchor)==1;c=c.replace(setup_anchor,capture+setup_anchor)
byte_anchor="    rec['n_out'] = len(_a); rec['nonfinite'] = sum(1 for _x in _a if not _math.isfinite(_x))"
assert c.count(byte_anchor)==1
c=c.replace(byte_anchor,byte_anchor+"\n    rec['prediction_sha256']=hashlib.sha256(_data).hexdigest(); rec['prediction_shape']=list(_shape); rec['prediction_dtype']=str(_dt)\n    with (HERE/f'{tag}_pred_{rank}.bin').open('xb') as _fp: _fp.write(_data)")
ast.parse(c); client=write('probe_client.py',c)
driver_parent=H/'run_boundary_census.py'; d=driver_parent.read_text()
d=d.replace('HERE=Path(__file__).resolve().parent\nP2=HERE.parent','STUDY=Path(__file__).resolve().parent\nHERE=STUDY.parent\nP2=HERE.parent')
lo=d.index("source=P2/'fable_porter_v1/census_boundary_setup_v2.py'"); hi=d.index('server_python=',lo)
d=d[:lo]+"source=STUDY/'probe_client.py'\nclient_script=source\npayload=source.read_bytes()\nmode=sys.argv[4]\nassert mode in ('normal','skip_phase2')\n"+d[hi:]
d=d.replace('str(candidate),ranks,tag],env=env)', 'str(candidate),ranks,tag,mode],env=env)')
ast.parse(d); driver=write('probe_driver.py',d)
write('probe_source_diffs.patch',''.join(difflib.unified_diff(client_parent.read_text().splitlines(True),c.splitlines(True),fromfile=str(client_parent),tofile='probe_client.py'))+'\n'+''.join(difflib.unified_diff(driver_parent.read_text().splitlines(True),d.splitlines(True),fromfile=str(driver_parent),tofile='probe_driver.py')))
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
cmds={f'normal_setup_{i}':[str(native),'-B',str(D/'probe_driver.py'),candidate['path'],'-',D.name+f'_normal_setup{i}_v1','normal'] for i in (1,2,3)}
for role,source in [('base',pin(parent)),('candidate',candidate)]:
    cmds['forced_'+role]=[str(native),'-B',str(D/'probe_driver.py'),source['path'],'35,100',D.name+f'_forced_{role}_v1','skip_phase2']
cmds['normal_native2']=[str(native),'-B',str(H/'hill_climb_e1_ynt_psdb_d5_v1/native_probe.py'),candidate['path'],'35,100',D.name+'_train2_v1']
write('commands.json',dict(build=build,protocol=protocol,probe_client=client,probe_driver=driver,
    original_client=pin(client_parent),original_driver=pin(driver_parent),commands=cmds,
    limits=dict(normal_setup_seconds=60,forced_pair_arm_seconds=240,normal_native2_seconds=150,rss_bytes=7500000000,added_swap_bytes=0),
    root_executes_serially=True,gate='Forced base/candidate must have all16 identical bytes, equal FLOPs, fewer candidate requests on each rank; normal N2 versus actual FG also equal bytes/FLOPs. All setup3/boundary limits unchanged.'))
print(json.dumps(dict(candidate=candidate,build=build,commands=pin(D/'commands.json'))))
