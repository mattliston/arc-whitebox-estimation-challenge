"""Prospective narrow VALID40/serving adapters. Stdlib only; never launches jobs."""
from pathlib import Path
import copy,difflib,hashlib,json,math,sys
D=Path(__file__).resolve().parent; H=D.parent; T=H/'hill_climb_e1_ynt_psdb_d5_v1'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def pin(p):
 p=Path(p).resolve(); b=p.read_bytes(); return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def read(p): return json.loads(Path(p).read_text())
def once(p,x):
 with Path(p).open('x') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def check(record):
 assert pin(record['path'])==record,record['path'];return read(record['path'])
def guard(p,command,seconds):
 g=read(p);assert g['command']==command
 assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None and not g['remaining_rows']
 assert g['max_swap_growth_bytes']==0 and 0<g['max_rss_bytes']<7500000000 and g['rss_trigger_bytes']==7500000000 and 0<g['elapsed_s']<=seconds

def admit_broad():
 cp=pin(D/'broad_config.json'); c=check(cp); nc=check(c['native_config']); st=nc['stages']['8']
 tr=pin(st['result']); t=check(tr)
 assert t['passed_integrity'] and t['admitted'] and t['n']==8 and all(t['gates'].values()) and t['all16_predictions_finite']
 assert t['config']==c['native_config'] and t['candidate']==c['candidate']['source']==pin(D/'candidate.py')
 assert t['all16_parent_bytes_exact'] and t['comparator']==c['incremental_reference']
 reference=check(c['native_s34_controls']);assert reference['arms']['base']['source']==c['score_reference']['source']
 rows=[]; inputs=[]
 for size in ('2','6'):
  rec=reference['arms']['base']['blocks'][size]['artifacts']['rows'];assert pin(rec['path'])==rec;inputs.append(rec)
  rows.extend(json.loads(line) for line in Path(rec['path']).read_text().splitlines())
 assert sorted(r['rank'] for r in rows)==sorted(t['ranks']) and len(rows)==8
 total=100*(t['candidate_mean_adjusted']/(math.fsum(r['adjusted'] for r in rows)/8)-1)
 assert math.isfinite(total) and total<=-.8
 guard(st['assessment_guard'],[str(NATIVE),'-B',nc['engine']['path'],c['native_config']['path'],c['native_config']['sha256'],'8'],120)
 bp=pin(nc['boundary_before8']['receipt']); b=check(bp)
 assert t['boundary']==bp and b['passed'] and b['receipt_checks_passed'] and all(b['gates'].values())
 assert b['candidate']==t['candidate'] and b['residual']['projected_s']<.4 and b['runtime']['projected_worst_wall_s']<=105
 guard(nc['boundary_before8']['assessment_guard'],nc['boundary_before8']['assessment_command'],120)
 once(D/'broad_admission.json',dict(passed=True,root_admitted=True,config=cp,candidate=t['candidate'],training=tr,
  training_guard=pin(st['assessment_guard']),boundary=bp,boundary_guard=pin(nc['boundary_before8']['assessment_guard']),
  bank_min_gain_pct=1.0,total_vs_S34_native8_pct=total,headroom_required_gain_pct=.8,
  headroom_reference=c['native_s34_controls'],headroom_rows=inputs,own_fixed40_and_serving_still_required=True))
 print(json.dumps(pin(D/'broad_admission.json')))

def enable_serving():
 bcp=pin(D/'broad_config.json'); bc=check(bcp); brp=pin(D/'broad_assessment.json'); br=check(brp)
 assert br['passed'] and br['receipt_checks_passed'] and br['validation_score_passed'] and all(br['gates'].values())
 assert br['config']==bcp and br['candidate']==bc['candidate']['source']==pin(D/'candidate.py')
 m=br['metrics'];assert m['n']==40 and m['adjusted_pct']<=-1 and m['adjusted_wins']>=28 and m['adjusted_ci95_pct'][1]<0 and m['max_per_network_adjusted_regression_pct']<=10
 guard(D/'broad_assess_guard.json',[str(NATIVE),'-B',bc['engine']['path'],bcp['path'],bcp['sha256'],br['admission']['sha256']],120)
 s=read(D/'serving_config_draft.json');assert s['prospective_authorization'] is False and s['broad_config']==bcp
 s['prospective_authorization']=True;once(D/'serving_config.json',s);cp=pin(D/'serving_config.json')
 once(D/'validation_admission.json',dict(passed=True,root_admitted=True,candidate=s['candidate'],config=cp,
  policy=s['policy'],runner=s['runner'],assessor=s['assessor'],broad_assessment=brp,broad_assess_guard=pin(D/'broad_assess_guard.json')))
 print(json.dumps(dict(config=cp,admission=pin(D/'validation_admission.json'))))

def adapt(source,target,replacements):
 old=source.read_text(); new=old
 for before,after,count in replacements:
  assert new.count(before)==count,(source,before,new.count(before));new=new.replace(before,after,count)
 compile(new,str(target),'exec')
 with target.open('x') as f:f.write(new)
 with target.with_suffix('.diff').open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(source),tofile=str(target))))
 return pin(target)

HEADROOM_CHECK = """    # Direct same-eight comparison with saved S34 controls, distinct from incremental parent.
    sp, sc = q.control_engine.configure(Path(c['native_s34_controls']['path']), c['native_s34_controls']['sha256'])
    assert sp == c['native_s34_controls'] and sc['arms']['base']['source'] == c['score_reference']['source']
    s34_rows = []; s34_inputs = []
    for size in (2, 6):
        sa = sc['arms']['base']
        records = {k: sa['blocks'][str(size)]['artifacts'][k] for k in ('manifest', 'rows', 'guard')}
        rows = native_cpu_rows(q, sa, size, records)
        for row in rows:
            assert math.isfinite(row['adjusted']) and row['adjusted'] > 0
            assert row['util'] == row['flops']/2**41
            assert math.isclose(row['adjusted'], row['mse']*max(.1,row['util']), rel_tol=1e-14, abs_tol=0)
            assert row['ubc']['retired'] and row['ubc']['retained_cache_bytes'] == row['ubc']['live_cache_bytes'] == 0
            assert 0 <= row['ubc']['max_resident_bytes'] <= 512000000
        s34_rows.extend(rows); s34_inputs.append(records['rows'])
    for record in q.control_engine.EVIDENCE.values(): q.checked(record)
    assert len(s34_rows) == 8 and sorted(row['rank'] for row in s34_rows) == sorted(training['ranks'])
    total = 100*(training['candidate_mean_adjusted']/(math.fsum(row['adjusted'] for row in s34_rows)/8)-1)
    assert math.isfinite(total) and total <= -.8
    assert a['headroom_required_gain_pct'] == .8 and a['headroom_reference'] == sp and a['headroom_rows'] == s34_inputs
    assert math.isclose(a['total_vs_S34_native8_pct'], total, rel_tol=0, abs_tol=1e-12)
"""

def main():
 if sys.argv[1:]==['--admit-broad']:admit_broad();return
 if sys.argv[1:]==['--enable-serving']:enable_serving();return
 assert not sys.argv[1:]
 nc=read(D/'native_screen_continuation.json');old=read(T/'broad_config.json');bank=read(H/'official_bank_S34_v1.json')
 assert bank['submission_id']==333050 and bank['candidate']==pin(T/'candidate.py')
 assert nc['arms']['candidate']['source']==pin(D/'candidate.py')
 assert nc['arms']['base']['source']['sha256']=='d255a7bc32cf72073c3e944b21730f86b25226324341fbab35df8abc82716c3b'
 assert nc['engine']==pin(D/'screen.py') and nc['engine']['sha256']=='fb54adcf66dd3c736dc960d73755e1742d95c80f08863225fe3c77c8283132dc'
 boundary_config=read(D/'boundary_config.json')
 assert bank['local_validation_reference']==pin(T/'broad_assessment.json')
 E=D/'valid40_engine_v1';E.mkdir();V=H/'valid40_engine_v2'
 assessor=adapt(V/'assess.py',E/'assess.py',[
  ('H2 = D.parent\n','H2 = D.parent.parent\n',1),
  ("SCREEN = H2/'native_screen_engine_v1/screen.py'", "SCREEN = D.parent/'screen.py'\nCONTROL_SCREEN = H2/'native_screen_engine_v1/screen.py'\nCONTROL_SCREEN_SHA = '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'",1),
  ("\nSCREEN_SHA = '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'", "\nSCREEN_SHA = "+repr(nc['engine']['sha256']),1),
  ("    q.ROSTERS.update(ROSTERS)  # panel() accepts these fixed process-order keys.\n", "    q.ROSTERS.update(ROSTERS)  # panel() accepts these fixed process-order keys.\n    assert hashlib.sha256(CONTROL_SCREEN.read_bytes()).hexdigest() == CONTROL_SCREEN_SHA\n    spec = importlib.util.spec_from_file_location('valid40_original_control_native', CONTROL_SCREEN)\n    control = importlib.util.module_from_spec(spec); spec.loader.exec_module(control)\n    q.control_engine = control\n",1),
  ("assert c['native_engine'] == q.pin(SCREEN) and c['score_gates'] == GATES", "assert c['native_engine'] == q.pin(SCREEN) and c['control_native_engine'] == q.pin(CONTROL_SCREEN) and c['score_gates'] == GATES",1),
  ("assert nc['arms']['base']['source'] == c['base']['source']", "assert nc['arms']['base']['source'] == c['incremental_reference']",1),
  ("training['comparator'] == c['base']['source']", "training['comparator'] == c['incremental_reference']",1),
  ("    assert training['all16_predictions_finite'] is True\n", "    assert training['all16_predictions_finite'] is True and training['all16_parent_bytes_exact'] is True\n",1),
  ("_, rc = q.configure(Path(reference['native_config']['path']),reference['native_config']['sha256'])", "_, rc = q.control_engine.configure(Path(reference['native_config']['path']),reference['native_config']['sha256'])\n    for record in q.control_engine.EVIDENCE.values(): q.checked(record)",1),
  ("score_comparator=c['base']['source'],score_comparator_native8_cpu_ratio=component_ratio", "score_comparator=c['incremental_reference'],score_comparator_native8_cpu_ratio=component_ratio",1),
  ("    component_ratio = math.fsum(cpu['candidate'])/math.fsum(cpu['base'])\n", HEADROOM_CHECK+"    component_ratio = math.fsum(cpu['candidate'])/math.fsum(cpu['base'])\n",1),
  ("S33_SHA = '88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'","S34_SHA = '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'",1),
  ("BOUNDARY_SHA = '86b15231a6f5582e5152f56a1595aa73919cb5120833213adb612f57a579dd0e'","BOUNDARY_SHA = "+repr(nc['boundary_before8']['assessor']['sha256']),1),
  ("reference['submission_id'] == 332838 and reference['source']['sha256'] == S33_SHA","reference['submission_id'] == 333050 and reference['source']['sha256'] == S34_SHA",1),
  ("bank['source_sha256'] == S33_SHA","bank['source_sha256'] == S34_SHA",1),
  ("assert residual['gate_s'] == .35 and residual['projected_s'] <= .35","assert residual['gate_s'] == .4 and residual['strict_upper_bound'] is True and residual['projected_s'] < .4",1)])
 runner=adapt(V/'run.py',E/'run.py',[])
 # No metrics, scoring bootstrap, fixed roster, worker schedule, or runtime formula changes.
 reference=dict(submission_id=333050,source=bank['candidate'],bank_receipt=pin(H/'official_bank_S34_v1.json'))
 base=copy.deepcopy(old['candidate']); previous=read(T/'broad_assessment.json')
 assert previous['passed'] and previous['validation_score_passed'] and previous['candidate']==base['source']
 for block in previous['blocks']:
  base['blocks'][block['block']]['artifacts']=copy.deepcopy(block['arms']['candidate'])
 controls=dict(schema='saved-score-controls-v2',score_reference=reference,assessment=pin(T/'broad_assessment.json'),arm=base,
  provenance=[pin(T/'broad_config.json'),pin(T/'broad_assessment.json'),pin(T/'broad_assess_guard.json'),reference['bank_receipt'],pin(__file__)])
 once(D/'score_controls_s34.json',controls)
 c=copy.deepcopy(old);c.pop('research_extension')
 c.update(engine=assessor,runner=runner,native_engine=nc['engine'],control_native_engine=pin(H/'native_screen_engine_v1/screen.py'),
  native_config=pin(D/'native_screen_continuation.json'),controls=pin(D/'score_controls_s34.json'),score_reference=reference,
  incremental_reference=nc['arms']['base']['source'],native_s34_controls=boundary_config['s34_native_config'])
 c['candidate']=copy.deepcopy(base);c['candidate']['source']=nc['arms']['candidate']['source'];c['candidate']['setup']=copy.deepcopy(nc['arms']['candidate']['setup'])
 for key,b in c['candidate']['blocks'].items():
  b.pop('artifacts');b['tag']=D.name+'_valid40_'+key+'_v1';b['guard']=str(D/f'broad_{key}_guard.json')
 c['provenance']=[pin(__file__),pin(D/'build.json'),pin(D/'protocol.json'),pin(D/'boundary_config.json'),pin(D/'boundary_assessment.json'),
  pin(T/'broad_config.json'),pin(V/'assess.py'),pin(V/'run.py'),pin(E/'assess.diff'),*controls['provenance'],c['controls']]
 c['provenance'] += [c['native_engine'],c['control_native_engine'],c['native_s34_controls'],c['incremental_reference']]
 c['interpretation']='Fixed40 compares actual saved S34 controls. Incremental N8 compares retained cachecast only; own exact N8/boundary and total native8 S34 gain>=.8% required. Original bank>=1%/28wins/negativeCI/maxreg10 and S32 runtime; candidate/exactcachecast640M, older controls512M. Serving only after full40 PASS.'
 once(D/'broad_config.json',c);cp=pin(D/'broad_config.json')
 S=D/'serving_engine_v1';S.mkdir();SV=H/'serving_engine_v2'
 common=adapt(SV/'common.py',S/'common.py',[
  ('H2 = D.parent\n','H2 = D.parent.parent\n',1),
  ("BROAD = H2/'valid40_engine_v2/assess.py'","BROAD = D.parent/'valid40_engine_v1/assess.py'",1),
  ("BROAD_SHA = '3f833cc04ab41f5516562590a546b5113386e77d2cda4fc42f7f4566777d4f21'","BROAD_SHA = "+repr(assessor['sha256']),1),
  ('source_bound_residual_s_max=.35','source_bound_residual_s_max=.4',1),
  ("ctx.boundary['residual']['projected_s']<=.35","ctx.boundary['residual']['projected_s']<.4",1)])
 sr=adapt(SV/'run.py',S/'run.py',[])
 sa=adapt(SV/'assess.py',S/'assess.py',[("boundary['residual']['projected_s'] <= .35","boundary['residual']['projected_s'] < .4",1)])
 s=read(H/'hill_climb_precision_g3_symdiag128_v1/serving_config.json')
 s.update(prospective_authorization=False,common=common,runner=sr,assessor=sa,broad_engine=assessor,broad_config=cp,
  candidate=c['candidate']['source'],policy=c['policy'],score_reference=reference,runtime_reference=c['runtime_reference'],
  release_tag=D.name+'_serving_v2',setup_counts=c['candidate']['setup']['counts'])
 s['resource_gates']['source_bound_residual_s_max']=.4
 s['provenance']=[pin(__file__),cp,*c['provenance'],*[pin(SV/n) for n in ('common.py','run.py','assess.py')],pin(S/'common.diff'),pin(S/'assess.diff')]
 s['interpretation']='Inactive worker20/setup3/contract qualification. Requires own complete fixed40 PASS at >=1%; existing boundary and S32 runtime controls retained; strict residual<0.4; no automatic retry.'
 once(D/'serving_config_draft.json',s)
 commands=dict(admit_broad=[str(H.parents[4]/'.venv/bin/python'),'-B',str(Path(__file__).resolve()),'--admit-broad'],
  broad_run=[str(NATIVE),'-B',runner['path'],cp['path'],cp['sha256'],'BROAD_ADMISSION_SHA'],
  broad_assess=[str(NATIVE),'-B',assessor['path'],cp['path'],cp['sha256'],'BROAD_ADMISSION_SHA'],
  enable_serving=[str(H.parents[4]/'.venv/bin/python'),'-B',str(Path(__file__).resolve()),'--enable-serving'])
 once(D/'qualification_commands.json',dict(commands=commands,broad_config=cp,serving_draft=pin(D/'serving_config_draft.json'),
  launch_held_until_root_decision=True,broad_admission_requires_own_native8_pass=True,serving_requires_broad_pass=True,models_executed=False))
 print(json.dumps(dict(broad_config=cp,broad_engine=assessor,serving_draft=pin(D/'serving_config_draft.json'),commands=pin(D/'qualification_commands.json'))))

if __name__=='__main__':main()
