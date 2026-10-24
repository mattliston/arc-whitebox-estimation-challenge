"""One held edge-bias reassociation pilot. Stdlib/source only."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;BASE=H/'hill_climb_prefix128_fusion_v1'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p):return json.loads(Path(p).read_text())
def write(n,x):
 p=D/n
 with p.open('x') as f:f.write(x if isinstance(x,str) else json.dumps(x,indent=2)+'\n')
 return pin(p)
parent=BASE/'candidate.py';assert pin(parent)['sha256']=='f9f4790b3c65f730404425e8923c767dc2f895fdcc1e14f14cc649286ad3b69f'
s=parent.read_text();tree=ast.parse(s)
node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_g5_message_edge_base'][-1]
oldfn=ast.get_source_segment(s,node)
before="    h = _e2_H(_e2_H(left[:, None, :]) + right) + t['bias']\n"
after="    h = _e2_H(left[:, None, :]) + _e2_H(right + t['bias'])\n"
assert oldfn.count(before)==1
newfn=oldfn.replace(before,after);assert s.count(oldfn)==1
t=s.replace(oldfn,newfn);assert t.replace(newfn,oldfn)==s;ast.parse(t)
candidate=write('candidate.py',t)
write('source_diff.patch',''.join(difflib.unified_diff(s.splitlines(True),t.splitlines(True),fromfile=str(parent),tofile='candidate.py')))
history=[H/'RESULTS.md',P/'fable_engineer2_v1/RESULT.md',P/'fable_porter_v1/RESULT.md']
duplicate=write('bounded_duplicate_check.json',dict(searched=[pin(p) for p in history],
 patterns=['bias.{0,35}hoist','hoist.{0,35}bias','right.{0,15}bias','edge.{0,15}bias'],matches=0,
 matching_study_names=['edge_lookup_tie_bias_source_note_v1.json (unrelated table quantization)'],
 interpretation='No exact prior found in these bounded logs/name inventory; not an exhaustive history claim. Existing tanh approximation failures remain closed.'))
parent_cfg=read(BASE/'native_screen_continuation.json');base=copy.deepcopy(parent_cfg['arms']['candidate'])
assert base['source']==pin(parent)
b=base['blocks']['2'];folder=H/b['tag']
b['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(b['guard']),predictions={str(r):pin(folder/f'pred_{r}.npy') for r in b['ranks']})
# N6 is intentionally not an authorized acquisition or control panel for this pilot.
base['blocks']['6'].pop('artifacts',None)
rows=[json.loads(x) for x in (folder/'rows.jsonl').read_text().splitlines()]
saved=7*16*(1024*1024-1024)
price=dict(calls=7,layers=list(range(8,15)),width=1024,channels=16,old_add_elements_per_call=2*1024*1024*16,
 new_add_elements_per_call=(1024*1024+1024)*16,saved_flops_per_network=saved,
 forecast_bill_pct_vs_parent_train2=-100*saved/(sum(r['flops'] for r in rows)/2),
 previous_eight_FG_forecast_pct=-.04454930803005142,
 mechanism='One dense n*n*16 FP32 add moves to an n*16 add; same two public additions, all work paid.',
 memory='One dense pre-bias temporary becomes one n*16 vector; no persistent storage/cache/arena change.')
protocol=write('protocol.json',dict(status='HELD_SOURCE_ONLY_NATIVE2_PILOT',candidate=candidate,parent=pin(parent),duplicate_check=duplicate,
 scope='Only active learned covariance edge helper L8–14 bias association; PI/setup/cache/weights/constants unchanged.',
 real_arithmetic_identity='(left_i+right_j)+bias == left_i+(right_j+bias)',
 numerical_caveat='FP32 association changes rounding; not an exact-prediction component. Prior edge tanh approximations were sensitive and failed. No expected accuracy gain asserted.',
 price=price,parity_layers=list(range(8)),all16_finite=True,
 native2_gates=dict(adjusted_pct_strictly_below=0.0,max_per_network_adjusted_regression_pct=.1,minimum_wins=0),
 interpretation='A strictly positive pooled adjusted gain is required; either per-network regression above0.1% rejects. This only selects a complementary component for future validation, not a>=1% bank.',
 one_fixed_scope=True,no_parameter_variants=True,no_grid=True,no_native6_or_boundary_or_broad_authority=True,
 root_execution_held_until_parent_qualification=True,no_models_executed=True))
build=write('build.json',dict(builder=pin(__file__),candidate=candidate,protocol=protocol,source_inverse=dict(parent=pin(parent),
 edits=[dict(before=oldfn,after=newfn)],exact_inverse=True),only_changed_function='_g5_message_edge_base',ast_parsed=True,
 active_target_occurrences=1,untouched_other_same_expression=True))
original=H/'native_screen_engine_v1/screen.py';es=original.read_text()
eb="    assert value['command'] == command\n"
ea="    actual_command = value['command'][1:] if value['command'][:1] == ['--'] else value['command']\n    assert actual_command == command\n"
assert es.count(eb)==1;new=es.replace(eb,ea);ast.parse(new)
engine=write('screen.py',new)
ei=write('engine_inverse.json',dict(engine=engine,parent=pin(original),edits=[dict(before=eb,after=ea)],exact_inverse=True,
 reason='Actual P128 training2 guard stores one leading nice option separator; source command/limits unchanged. No FLOP equality for this prospective cost/reassociation trial.'))
child=copy.deepcopy(base);child['source']=candidate
for size in ('2','6'):
 block=child['blocks'][size];block.pop('artifacts',None);block['tag']=D.name+f'_train{size}_v1';block['guard']=str(D/f'native{size}_guard.json')
stages={}
for n in (2,8):
 gates=(dict(adjusted_pct_at_most=0.0,strict_adjusted=True,wins_min=0,ci_upper_below=None,max_per_network_regression_pct=.1,additional6_gain_required=False)
        if n==2 else dict(adjusted_pct_at_most=-.1,strict_adjusted=False,wins_min=6,ci_upper_below=0,max_per_network_regression_pct=1.0,additional6_gain_required=True))
 stages[str(n)]=dict(result=str(D/f'native{n}_assessment.json'),assessment_guard=str(D/f'native{n}_assess_guard.json'),assessment_limits=dict(seconds=120,rss_bytes=7500000000),gates=gates)
boundary=copy.deepcopy(parent_cfg['boundary_before8']);boundary['receipt']=str(D/'NATIVE8_NOT_AUTHORIZED.json');boundary['assessment_guard']=str(D/'NATIVE8_NOT_AUTHORIZED_guard.json')
c=dict(schema='native-screen-v1',prospective_authorization=True,future_only=True,raw_error_veto=False,engine=engine,
 provenance=[pin(__file__),build,protocol,duplicate,ei,pin(BASE/'native_screen_continuation.json'),pin(BASE/'build.json')],
 arms=dict(candidate=child,base=base,parity_parent=copy.deepcopy(base)),parity_layers=list(range(8)),stages=stages,boundary_before8=boundary,
 prospective_scope='HELD_NATIVE2_ONLY',no_further_stage_authority=True,root_execution_held_until_parent_qualification=True,
 interpretation='Only stage2 is selected. Stage8 structural fields are dormant original approximate defaults, not future authorization; absent required receipt and unbound base6 prevent use.')
cp=write('native_screen.json',c)
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python';b=child['blocks']['2']
commands=dict(native2=[str(native),'-B',b['driver']['path'],candidate['path'],'35,100',b['tag']],
 native2_assessment=[str(native),'-B',engine['path'],cp['path'],cp['sha256'],'2'])
cmd=write('commands.json',dict(config=cp,commands=commands,native2_guard=b['guard'],native2_limits=b['limits'],
 assessment_guard=stages['2']['assessment_guard'],assessment_limits=stages['2']['assessment_limits'],
 HELD=True,root_executes_serially_only_after_parent_qualification=True,no_other_commands_authorized=True))
print(json.dumps(dict(candidate=candidate,protocol=protocol,build=build,config=cp,commands=cmd)))
