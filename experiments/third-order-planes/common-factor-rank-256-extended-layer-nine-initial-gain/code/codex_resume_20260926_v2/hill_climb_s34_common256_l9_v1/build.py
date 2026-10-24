"""One current-S34 common256 L9 extension; stdlib source/scalar proof only."""
from pathlib import Path
import ast, copy, hashlib, json
D=Path(__file__).resolve().parent
H=D.parent
SOURCE=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
TEMPLATE=H/'hill_climb_s34_ynt_zonal3_v1/native_screen.json'
PRIOR=H/'hill_climb_common256_l7_v1/result.json'

def pin(path):
    path=Path(path).resolve(); raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

def once(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n')

def rewrite(value):
    if isinstance(value,dict): return {k:rewrite(v) for k,v in value.items()}
    if isinstance(value,list): return [rewrite(v) for v in value]
    if isinstance(value,str): return value.replace(TEMPLATE.parent.name,D.name)
    return value

assert pin(SOURCE)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(TEMPLATE)['sha256']=='442bf11d03eebcb1ae3b6090d2280d57984386573551e665ae4a71f57d8c18a6'
source=SOURCE.read_text(); fragment=(D/'fragment.py').read_text()
tree=ast.parse(source); edits=ast.parse(fragment)
assert len(edits.body)==2 and all(isinstance(n,ast.Assign) for n in edits.body)
assert [n.targets[0].id for n in edits.body]==['_LCR_LAYERS','_UVP_ONLY_LAYERS']
names={'_LCR_RU','_LCR_LAYERS','_UVP_ONLY_LAYERS'}
assignments=[n for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1
    and isinstance(n.targets[0],ast.Name) and n.targets[0].id in names]
scope={}
exec(compile(ast.Module(body=assignments,type_ignores=[]),'<scalar-base>','exec'),scope)
assert scope['_LCR_RU']==256 and scope['_LCR_LAYERS']==(10,11,12,13,14)
assert scope['_UVP_ONLY_LAYERS']==(10,11,12,13,14,15)
exec(compile(edits,'<scalar-extension>','exec'),scope)
assert scope['_LCR_RU']==256 and scope['_LCR_LAYERS']==(9,10,11,12,13,14)
assert scope['_UVP_ONLY_LAYERS']==(9,10,11,12,13,14,15)
candidate=source+'\n'+fragment
assert candidate[:-len('\n'+fragment)]==source
newtree=ast.parse(candidate)
assert len(newtree.body)==len(tree.body)+2
assert all(ast.dump(a)==ast.dump(b) for a,b in zip(tree.body,newtree.body))
assert [ast.dump(n) for n in newtree.body[-2:]]==[ast.dump(n) for n in edits.body]
with (D/'candidate.py').open('x') as f: f.write(candidate)
prior=json.loads(PRIOR.read_text())
assert prior['passed'] is False and prior['stage']=='TRAIN16_COMPONENT'
assert prior['incremental_adjusted_pct']==0.18031003789613642 and prior['incremental_wins']==7
protocol=dict(schema='s34-common256-l9-v1',root_selected=True,baseline=pin(SOURCE),
    hypothesis='L9 may benefit from the incumbent late common256 approximation with less prefix distortion than the failed L7-9 extension.',
    added_layer=9,common_rank=256,LCR_layers=list(scope['_LCR_LAYERS']),UVP_layers=list(scope['_UVP_ONLY_LAYERS']),
    parity_layers=list(range(9)),first_possible_changed_output=9,
    parity_reason='_shared_old384_push sets _UVP_CUR to the current layer before inherited reconstruction/read; layer9 precedes the layer9 pair map and output.',
    signed_difference_directions_unchanged=True,carried_F_core_Q_unchanged=True,
    full_factor_trace_diagonal_radial_repairs_unchanged=True,
    joins_ranks_cutoffs_births_learned_parameters_unchanged=True,
    cache_scope='Existing UBC remains restricted to L10-15; L9 takes the ordinary uncached UVP path.',
    wrappers_captures_reset_unchanged=True,scratch_arena_and_setup_keys_unchanged=True,
    prior_failure=dict(result=pin(PRIOR),adjusted_pct=prior['incremental_adjusted_pct'],wins=7,n=16,
        ci95=prior['incremental_ci95'],additional14_pct=prior['additional14_pct'],failure_preserved=True),
    no_current_gain_claim=True,fit_or_sweep=False,no_further_LCR_windows_or_ranks_if_failed=True,
    roster=[35,100],native2_gates=dict(adjusted_pct_at_most=-.1,wins_min=1,max_per_network_regression_pct=1.),
    scope='native2_only',boundary_authorized=False,broad_authorized=False,serving_qualified=False,
    numpy_torch_or_model_executed=False)
once(D/'protocol.json',protocol)
build=dict(candidate=pin(D/'candidate.py'),parent=pin(SOURCE),builder=pin(__file__),fragment=pin(D/'fragment.py'),
    protocol=pin(D/'protocol.json'),source_inverse_exact=True,exact_parent_prefix=True,
    original_statements_ast_exact=True,appended_assignments=2,scalar_scope_check_passed=True,
    changed_layer_membership_only=[9],common_rank_unchanged=256,numpy_torch_or_model_executed=False,
    serving_qualified=False,submitted=False)
once(D/'build.json',build)
base=json.loads(TEMPLATE.read_text()); config=rewrite(copy.deepcopy(base))
config['arms']['candidate']['source']=build['candidate']
assert config['arms']['base']==base['arms']['base'] and config['arms']['parity_parent']==base['arms']['parity_parent']
assert config['arms']['base']['source']==pin(SOURCE)
config['parity_layers']=list(range(9))
config['provenance']=[pin(p) for p in (TEMPLATE,SOURCE,D/'build.py',D/'fragment.py',D/'protocol.json',D/'build.json',
    H/'official_bank_S34_v1.json',PRIOR)]
config['interpretation']=('One fixed S34 common256 L9 extension, no rank/cutoff/capture changes. Prior combined L7-9 TRAIN16 failure preserved. '
    'Exact L0-8 parity, original N2 gates/resources and genuine S34 controls. No further LCR window/rank rescue if this fails; '
    'no boundary/additional-six/broad/serving authority.')
assert config['prospective_scope']=='native2_only'
for s in ('2','8'):
    assert config['stages'][s]['gates']==base['stages'][s]['gates']
once(D/'native_screen.json',config)
print(json.dumps(dict(candidate=build['candidate'],config=pin(D/'native_screen.json'),build=pin(D/'build.json'),protocol=pin(D/'protocol.json'))))
