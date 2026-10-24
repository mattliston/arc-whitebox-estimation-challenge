"""Stdlib-only retention and prospective binding; no arrays or model execution."""
import copy, hashlib, json
from pathlib import Path
E=Path(__file__).resolve().parent; H=E.parent; P=H.parent; R=P.parents[3]
D=H/'hill_climb_exact_plane_reuse_v1'; C=H/'hill_climb_cov_cast_star_reuse_v1'
LIGHT=str(R/'.venv/bin/python'); NATIVE=str(P/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
GUARD=str(P/'codex_resume_20260919_v1/current_p_native_lifecycle_v1/guard.py')
def read(p): return json.loads(Path(p).read_text())
def pin(p):
 p=Path(p).resolve(); b=p.read_bytes(); return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def check(p): assert pin(p['path'])==p; return p
def save(p,x):
 assert not p.exists(), str(p)
 p.write_text(json.dumps(x,indent=2)+'\n'); return pin(p)
def guard(p,seconds,command=None):
 g=read(p)
 assert g['completed'] and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] and not g['remaining_rows']
 assert g['max_swap_growth_bytes']==0 and g['rss_trigger_bytes']==7500000000 and g['max_rss_bytes']<7500000000 and g['elapsed_s']<=seconds
 if command is not None: assert g['command']==command
 return pin(p)
nc=read(D/'native_screen_continuation.json'); a=read(D/'native8_assessment.json'); b=read(D/'boundary_assessment.json')
assert a['config']==pin(D/'native_screen_continuation.json') and a['engine']==check(nc['engine'])
assert a['candidate']==check(nc['arms']['candidate']['source']) and a['comparator']==check(nc['arms']['base']['source'])
assert a['passed_integrity'] and a['admitted'] and a['n']==8 and all(a['gates'].values())
assert a['raw_pct']==0 and a['adjusted_wins']==8 and a['all16_predictions_finite'] and a['all16_parent_bytes_exact'] and a['parity_layers']==list(range(16))
assert a['boundary']==pin(D/'boundary_assessment.json') and b['passed'] and b['receipt_checks_passed'] and all(b['gates'].values()) and b['candidate']==a['candidate']
for p in a['evidence']: check(p)
parentarm=copy.deepcopy(nc['arms']['candidate']); parity=[]; guards=[]
for block in ['2','6']:
 spec=parentarm['blocks'][block]; folder=H/spec['tag']; check(spec['driver'])
 cmd=[NATIVE,'-B',spec['driver']['path'],a['candidate']['path'],','.join(map(str,spec['ranks'])),spec['tag']]
 g=guard(spec['guard'],spec['limits']['seconds'],cmd); guards.append(g)
 artifacts=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=g,predictions={})
 for rank in spec['ranks']:
  cp=pin(folder/f'pred_{rank}.npy'); pp=check(nc['arms']['base']['blocks'][block]['artifacts']['predictions'][str(rank)])
  assert cp['sha256']==pp['sha256'] and cp['bytes']==pp['bytes']
  artifacts['predictions'][str(rank)]=cp; parity.append(dict(rank=rank,candidate=cp,parent=pp,all16_file_bytes_exact=True))
 spec['artifacts']=artifacts
for stage in ['2','8']:
 spec=nc['stages'][stage]
 guards.append(guard(spec['assessment_guard'],spec['assessment_limits']['seconds'],[NATIVE,'-B',nc['engine']['path'],str(D/'native_screen_continuation.json'),pin(D/'native_screen_continuation.json')['sha256'],stage]))
bs=nc['boundary_before8']; guards.append(guard(bs['assessment_guard'],bs['assessment_limits']['seconds'],bs['assessment_command']))
guards.append(guard(D/'boundary_guard.json',240))
cr=read(C/'retained_component.json')
ret=dict(status='RETAINED_COMPONENT',candidate=a['candidate'],comparator=a['comparator'],native8=pin(D/'native8_assessment.json'),config=a['config'],engine=a['engine'],boundary=a['boundary'],parent_retained=pin(C/'retained_component.json'),incremental_adjusted_pct=a['adjusted_pct'],total_vs_S34_adjusted_pct=100*((1+cr['total_vs_S34_adjusted_pct']/100)*(1+a['adjusted_pct']/100)-1),total_comparison_basis='same-eight adjusted mean ratios composed through retained parent',raw_pct=a['raw_pct'],wins=a['adjusted_wins'],confidence_interval95_pct=a['adjusted_ci95_pct'],all16_parent_bytes_exact=True,whole8_byte_parity=parity,completed_guards=guards,added_swap_bytes=0,numerical_jobs='all completed and retired',preserved_parent_failed40=pin(C/'broad_assessment.json'),decision='Retain exact plane reuse as research component. No broad or serving qualification claimed; final bank still requires VALID40 gain >=1% and full serving.',release_qualified=False,submitted=False,recorder=pin(__file__))
rp=save(D/'retained_component.json',ret)
draft=read(E/'native_screen_draft.json'); assert pin(E/'native_screen_draft.json')['sha256']=='b8be6962b64092387b0516ca127478b11bac765cc0716eff8a438e60a075fc8b'
check(draft['engine']); check(draft['arms']['candidate']['source']); cfg=copy.deepcopy(draft)
for role in ['base','parity_parent']:
 old=copy.deepcopy(cfg['arms'][role]); new=copy.deepcopy(parentarm)
 for block in ['2','6']:
  oldart=old['blocks'][block].pop('artifacts',None); newart=new['blocks'][block].pop('artifacts')
  if oldart is not None: assert oldart==newart
 assert old==new
 cfg['arms'][role]=copy.deepcopy(parentarm)
cfg.update(prospective_authorization=True,prospective_scope='native2_only',parent2_pinned=True,parent6_pinned=True,parent_retained=rp)
cfg['provenance'] += [pin(__file__),pin(E/'native_screen_draft.json'),rp,pin(D/'native8_assessment.json'),pin(D/'native_screen_continuation.json'),pin(E/'independent_source_review.json')]
assert cfg['stages']==draft['stages'] and cfg['parity_layers']==list(range(8)) and cfg['arms']['candidate']==draft['arms']['candidate']
cp=save(E/'native_screen.json',cfg)
s34=read(E/'native_screen_s34_draft.json'); s34.update(prospective_authorization=True,prospective_scope='report_only',parent2_pinned=True,parent6_pinned=True,parent_retained=rp)
s34['arms']['parity_parent']=copy.deepcopy(parentarm); s34['provenance'] += [pin(__file__),cp,rp,pin(E/'native_screen_s34_draft.json')]
sp=save(E/'native_screen_s34.json',s34)
def wrapped(record,seconds,cmd): return [LIGHT,'-B',GUARD,'--record',str(record),'--seconds',str(seconds),'--rss-limit','7500000000',*cmd]
ca=cfg['arms']['candidate']; spec=ca['blocks']['2']; stage=cfg['stages']['2']; ss=s34['stages']['2']
commands=dict(native2=wrapped(spec['guard'],spec['limits']['seconds'],[NATIVE,'-B',spec['driver']['path'],ca['source']['path'],','.join(map(str,spec['ranks'])),spec['tag']]),assess_native2=wrapped(stage['assessment_guard'],stage['assessment_limits']['seconds'],[NATIVE,'-B',cfg['engine']['path'],cp['path'],cp['sha256'],'2']),report_s34=wrapped(ss['assessment_guard'],ss['assessment_limits']['seconds'],[NATIVE,'-B',cfg['engine']['path'],sp['path'],sp['sha256'],'2']))
com=save(E/'commands.json',commands)
print(json.dumps(dict(retained=rp,config=cp,report=sp,commands=com),indent=2))
