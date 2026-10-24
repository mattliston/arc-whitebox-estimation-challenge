"""Immutable base transpose cache, one fixed source/config; stdlib only."""
from pathlib import Path
import ast,copy,difflib,hashlib,json,shlex
D=Path(__file__).resolve().parent;H=D.parent;P=H.parent;R=P.parents[3]
BASE=H/'hill_climb_combo_bd_linear_delta_v1/candidate.py';PARENT=BASE
T=BASE.parent/'native_screen.json';ENGINE=H/'native_screen_engine_v1/screen.py'
CLOSE=H/'s34_additional_uvp_cache_source_closeout_v1.json'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(BASE)['sha256']=='6f104e9db5ca32dc5e7724d64e03fd969bfff9d6cb48ce2b55695a998eecd1bb'
assert pin(T)['sha256']=='2b0c97146cb387d961c2c786841777e2ba41656346371d2398528a66025bc545'
assert pin(ENGINE)['sha256']=='1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
s=BASE.read_text();candidate=s;edits=[]
def change(before,after):
 global candidate
 assert candidate.count(before)==1
 candidate=candidate.replace(before,after,1);edits.append(dict(before=before,after=after))
change('_UBC_LIMIT = 512000000','_UBC_LIMIT = 640000000')
change("    inner = _shared_old384_mm(Dv, Fv2.T)\n    inner_u = _shared_old384_mm(Su, Fu.T)\n", "    inner = _ubc_transpose_product(Dv, Fv2, uv, 'vt')\n    inner_u = _ubc_transpose_product(Su, Fu, uv, 'ut')\n")
old="""            if (L == 6 and k == rank == (256 if leg == 'u' else 320)
                    and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32'
                    and (main or tail)):
                key = (run['generation'], leg, offset, valid, L, k, n, 'float32')
"""
new="""            if (leg in ('ut', 'vt') and L == 5 and k == rank == width
                    and offset == 0 and valid == stored == n == (256 if leg == 'ut' else 320)
                    and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32'):
                key = (run['generation'], leg, offset, valid, L, k, n, 'float32')
            elif (L == 6 and k == rank == (256 if leg == 'u' else 320)
                    and tuple(b.shape) == (k, n) and str(b.dtype) == 'float32'
                    and (main or tail)):
                key = (run['generation'], leg, offset, valid, L, k, n, 'float32')
"""
change(old,new)
change("len(run['cache']) < 3", "len(run['cache']) < 5")
change("hit_slots={'u:0': 0, 'v:0': 0, 'v:4096': 0}","hit_slots={'u:0': 0, 'v:0': 0, 'v:4096': 0, 'ut:0': 0, 'vt:0': 0}")
fragment=(D/'fragment.py').read_text();candidate+='\n'+fragment
inverse=candidate[:-len('\n'+fragment)]
for edit in reversed(edits):assert inverse.count(edit['after'])==1;inverse=inverse.replace(edit['after'],edit['before'],1)
assert inverse==s;compile(candidate,str(D/'candidate.py'),'exec');ast.parse(candidate)
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(s.splitlines(True),candidate.splitlines(True),fromfile=str(BASE),tofile=str(D/'candidate.py'))))
once(D/'source_inverse.json',dict(parent=pin(BASE),edits=edits,fragment=pin(D/'fragment.py'),exact_inverse=True))
e=ENGINE.read_text();before="            assert 0 <= ubc['max_resident_bytes'] <= 512000000";after="            assert 0 <= ubc['max_resident_bytes'] <= (640000000 if role == 'candidate' else 512000000)"
assert e.count(before)==1;modified=e.replace(before,after,1);assert modified.replace(after,before,1)==e
compile(modified,str(D/'screen.py'),'exec')
with (D/'screen.py').open('x') as f:f.write(modified)
with (D/'engine_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(e.splitlines(True),modified.splitlines(True),fromfile=str(ENGINE),tofile=str(D/'screen.py'))))
once(D/'engine_inverse.json',dict(parent=pin(ENGINE),engine=pin(D/'screen.py'),before=before,after=after,exact_inverse=True))
forecast=json.loads(CLOSE.read_text())['base_transpose']
protocol=dict(schema='bd-delta-uvp-fulltranspose-cache-v1',root_selected=True,parent=pin(BASE),scope='native2_only',
 intervention='Only the existing _uvp_inner Fu.T/Fv.T products gain explicit immutable-base ownership. Cache full unsliced depth5 transforms for ut/vt at the same current routes; original three depth6 reconstruction slots remain. No new kernel, padding policy, rank, layer window, recent-join caching or partial transpose reuse.',
 ownership='Original immutable base F object, generation, uv identity, and matching exact factor object. A new transpose view is explicitly registered. Existing slice/padding ancestry hooks can identify descendants, but the full-transpose gate requires complete zero-offset unpadded columns and k equal original base width. Original retirement/drop/finally behavior clears all five slots.',
 exactness='Return a paid independent copy of the identical original depth5 B transform on hits. All products, leaves, unwind, read arithmetic and source operands unchanged. All16-byte parity required at N2 and N8; this remains empirical until checked.',
 prior_source_closeout=pin(CLOSE),prior_was_no_build_no_model=True,
 explicit_new_root_admission='Root prospectively reopened the full-base transpose source-only lead and authorized internal cache cap640000000 bytes and max5 entries instead of512000000/max3. Earlier source closure and forecast remain unchanged. This is not the recent-join .001% lead.',
 resources=dict(internal_cache_bytes=640000000,cache_slots=5,process_rss_limit_bytes=7500000000,added_swap_bytes=0,setup_seconds_max=4.1,full_serving_limits_unchanged=True,candidate_only_assessor_revision=True,control_cache_bytes=512000000),
 forecast=forecast,forecast_qualification='Predecessor rank35 observed fullD5 transpose routes only, not current measured gain or rank100 hit evidence. Includes paid copy cost; extra221--232MB expected. Actual RSS/runtime/bill and all16 parity decide.',
 gates='N2 vs retained BD-delta: gain>=.02%,2wins,maxreg0,ALL16 bytes exact,allfinite/resources. Own boundary before N8 gain>=.02%,8wins,CIupper<0,additional6negative,ALL16 bytes exact. Broad totalS34>=.8%; final VALID40 bank>=1% plus unchanged fullserving.',
 parity_layers=list(range(16)),baseline_retained=pin(BASE.parent/'retained_component.json'),baseline_native8=pin(BASE.parent/'native8_assessment.json'),no_cache_grid_or_layer_followons=True,model_executed=False,serving_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(BASE),builder=pin(__file__),protocol=pin(D/'protocol.json'),source_inverse=pin(D/'source_inverse.json'),source_diff=pin(D/'source_diff.patch'),fragment=pin(D/'fragment.py'),engine=pin(D/'screen.py'),engine_inverse=pin(D/'engine_inverse.json'),source_inverse_exact=True,model_executed=False))
original=json.loads(T.read_text());old_name=T.parent.name
def rename(x):
 if isinstance(x,dict):return {k:rename(v) for k,v in x.items()}
 if isinstance(x,list):return [rename(v) for v in x]
 if isinstance(x,str) and x.startswith('/'):
  parts=[D.name if v==old_name else D.name+v[len(old_name):] if v.startswith(old_name+'_') else v for v in Path(x).parts];return str(Path(*parts))
 if isinstance(x,str) and x.startswith(old_name+'_'):return D.name+x[len(old_name):]
 return x
c=rename(copy.deepcopy(original));c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(16));c['engine']=pin(D/'screen.py')
bdc=json.loads((BASE.parent/'native_screen_continuation.json').read_text());base=copy.deepcopy(bdc['arms']['candidate'])
for size in ('2','6'):
 block=base['blocks'][size];folder=H/block['tag'];block['artifacts']=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(r):pin(folder/f'pred_{r}.npy') for r in block['ranks']})
c['arms']['base']=copy.deepcopy(base);c['arms']['parity_parent']=copy.deepcopy(base)
c['stages']['2']['gates'].update(adjusted_pct_at_most=-.02,wins_min=2,max_per_network_regression_pct=0.0)
c['stages']['8']['gates'].update(adjusted_pct_at_most=-.02,wins_min=8,max_per_network_regression_pct=0.0)
c['provenance']=[pin(p) for p in (BASE,T,ENGINE,CLOSE,D/'build.py',D/'build.json',D/'protocol.json',D/'source_inverse.json',D/'source_diff.patch',D/'fragment.py',D/'screen.py',D/'engine_inverse.json',D/'engine_diff.patch',BASE.parent/'retained_component.json',BASE.parent/'native_screen_continuation.json')]
c['interpretation']=protocol['gates']
c.pop('additional_root_review',None)
assert c['prospective_scope']=='native2_only'
once(D/'native_screen.json',c)
s34=copy.deepcopy(c);s34['arms']['base']=json.loads((T.parent/'native_screen_s34_report.json').read_text())['arms']['base']
for size in ('2','8'):
 s34['stages'][size]['result']=str(D/f'native{size}_s34_report.json');s34['stages'][size]['assessment_guard']=str(D/f'native{size}_s34_report_guard.json')
s34['interpretation']='Descriptive S34 report only; incremental gates use retained BD parent in native_screen.json.'
once(D/'native_screen_s34_report.json',s34)
native=P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python';guard=P/'codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py';light=R/'.venv/bin/python'
def wrap(record,seconds,rss,cmd):return [str(light),'-B',str(guard),'--record',str(record),'--seconds',str(seconds),'--rss-limit',str(rss),*cmd]
b=c['arms']['candidate']['blocks']['2'];commands=dict(native2=wrap(b['guard'],b['limits']['seconds'],b['limits']['rss_bytes'],[str(native),'-B',b['driver']['path'],str(D/'candidate.py'),','.join(map(str,b['ranks'])),b['tag']]))
for name,file in [('assess_native2','native_screen.json'),('report_s34','native_screen_s34_report.json')]:
 cfg=json.loads((D/file).read_text());stage=cfg['stages']['2'];commands[name]=wrap(stage['assessment_guard'],stage['assessment_limits']['seconds'],stage['assessment_limits']['rss_bytes'],[str(native),'-B',cfg['engine']['path'],str(D/file),pin(D/file)['sha256'],'2'])
once(D/'commands.json',commands)
with (D/'commands.txt').open('x') as f:
 for name,command in commands.items():f.write('# '+name+'\n'+shlex.join(command)+'\n\n')
print(json.dumps({name:pin(D/name) for name in ('candidate.py','native_screen.json','build.json','protocol.json','commands.json')}))
