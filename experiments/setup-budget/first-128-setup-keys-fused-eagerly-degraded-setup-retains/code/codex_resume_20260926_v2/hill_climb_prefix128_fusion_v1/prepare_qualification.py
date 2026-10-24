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
 g=read(p);actual=g['command'][1:] if g['command'][:1]==['--'] else g['command'];assert actual==command
 assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None and not g['remaining_rows']
 assert g['max_swap_growth_bytes']==0 and 0<g['max_rss_bytes']<7500000000 and g['rss_trigger_bytes']==7500000000 and 0<g['elapsed_s']<=seconds

def admit_broad():
 cp=pin(D/'broad_config.json'); c=check(cp); nc=check(c['native_config']); st=nc['stages']['8']
 tr=pin(st['result']); t=check(tr)
 assert t['passed_integrity'] and t['admitted'] and t['n']==8 and all(t['gates'].values()) and t['all16_predictions_finite']
 assert t['config']==c['native_config'] and t['candidate']==c['candidate']['source']==pin(D/'candidate.py')
 assert t['all16_parent_bytes_exact'] and t['comparator']==c['incremental_reference']
 assert t['per_network_flops_equal'] and t['raw_pct']==t['adjusted_pct']==t['bill_pct']==0.0
 cheap=check(c['cheap_gate']);assert cheap['passed'] and cheap['candidate']==t['candidate']
 assert [r['rank'] for r in cheap['rows']]==[35,100] and all(r['all16_bytes_equal'] and r['flops_equal'] and r['requests_saved']>0 for r in cheap['rows'])
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


def main():
 if sys.argv[1:]==['--admit-broad']:admit_broad();return
 if sys.argv[1:]==['--enable-serving']:enable_serving();return
 assert not sys.argv[1:]
 F=H/'hill_climb_scatter_forward_gather_v1'
 nc=read(D/'native_screen_continuation.json');bc=read(D/'boundary_config.json');br=read(D/'boundary_assessment.json')
 assert nc['arms']['candidate']['source']==pin(D/'candidate.py')
 assert nc['arms']['candidate']['source']['sha256']=='f9f4790b3c65f730404425e8923c767dc2f895fdcc1e14f14cc649286ad3b69f'
 assert nc['arms']['base']['source']==pin(F/'candidate.py') and nc['engine']==pin(D/'screen_v2.py')
 assert br['passed'] and br['receipt_checks_passed'] and br['candidate']==nc['arms']['candidate']['source']
 assert br['assessor']==nc['boundary_before8']['assessor']
 cheap=pin(D/'cheap_mechanism_result_v1.json')
 assert cheap['sha256']=='7b20f949b6f34d759985975e77b553395247afdb808e063c45309a6f6e14ad71'
 cr=check(cheap);assert cr['passed'] and cr['candidate']==nc['arms']['candidate']['source']
 for n in ('2','8'):
  assert nc['stages'][n]['gates']==dict(adjusted_pct_at_most=0.0,strict_adjusted=False,wins_min=0,ci_upper_below=None,max_per_network_regression_pct=0.0,additional6_gain_required=False)
 assert nc['parity_layers']==list(range(16))
 assert nc['arms']['candidate']['setup']['counts']==dict(bound=687,lvfuse=521,psdb_fused=9,pair_block_indices=1,read7_fused=4)
 old=read(F/'broad_config.json');bank=read(H/'official_bank_S34_v1.json')
 assert bank['submission_id']==333050 and old['score_reference']['source']==bank['candidate']
 E=D/'valid40_engine_v1';E.mkdir()
 extra="""    assert training['per_network_flops_equal'] is True
    assert training['raw_pct'] == training['adjusted_pct'] == training['bill_pct'] == 0.0
    cheap = load(q, c['cheap_gate'])
    assert cheap['passed'] is True and cheap['candidate'] == c['candidate']['source']
    assert [r['rank'] for r in cheap['rows']] == [35,100]
    assert all(r['all16_bytes_equal'] and r['flops_equal'] and r['requests_saved'] > 0 for r in cheap['rows'])
"""
 assessor=adapt(F/'valid40_engine_v1/assess.py',E/'assess.py',[
  ("SCREEN = D.parent/'screen.py'","SCREEN = D.parent/'screen_v2.py'",1),
  ("\nSCREEN_SHA = 'ffbd26aa01839d6017fb3b2482d1667901864b5da4f02d37d7b99974626d6b8a'","\nSCREEN_SHA = "+repr(nc['engine']['sha256']),1),
  ("BOUNDARY_SHA = '0e1c5c61be3724ecb241d2b6d581e151c010e3bccf287fc90d1846fc483c8f11'","BOUNDARY_SHA = "+repr(nc['boundary_before8']['assessor']['sha256']),1),
  ("    assert training['all16_predictions_finite'] is True and training['all16_parent_bytes_exact'] is True\n","    assert training['all16_predictions_finite'] is True and training['all16_parent_bytes_exact'] is True\n"+extra,1)])
 runner=adapt(F/'valid40_engine_v1/run.py',E/'run.py',[])
 c=copy.deepcopy(old)
 c.update(engine=assessor,runner=runner,native_engine=nc['engine'],native_config=pin(D/'native_screen_continuation.json'),
  incremental_reference=nc['arms']['base']['source'],native_s34_controls=bc['s34_native_config'],cheap_gate=cheap)
 c['candidate']['source']=nc['arms']['candidate']['source'];c['candidate']['setup']=copy.deepcopy(nc['arms']['candidate']['setup'])
 for key,b in c['candidate']['blocks'].items():
  assert 'artifacts' not in b
  b['tag']=D.name+'_valid40_'+key+'_v1';b['guard']=str(D/f'broad_{key}_guard.json')
 c['provenance']=[pin(__file__),pin(D/'build.json'),pin(D/'protocol.json'),pin(D/'native_resource_protocol_v2.json'),
  pin(D/'boundary_config.json'),pin(D/'boundary_assessment.json'),cheap,pin(F/'broad_config.json'),
  pin(F/'valid40_engine_v1/assess.py'),pin(F/'valid40_engine_v1/run.py'),pin(E/'assess.diff'),
  c['controls'],c['native_engine'],c['control_native_engine'],c['native_s34_controls'],c['incremental_reference'],pin(H/'official_bank_S34_v1.json')]
 c['interpretation']='Own fixed40 against original saved official S34 controls; FG is only native equality comparator and its official failure remains. Prefix128 source has deferred521+eager128, total649. Requires cheap strict mechanism, own normal N8 all16/FLOP equality and own normal boundary. New full40 bank>=1%/28wins/negativeCI/maxreg10 unchanged; serving20 only after PASS.'
 c['setup_fusion_semantics']=dict(deferred=521,eager=128,total=649)
 once(D/'broad_config.json',c);cp=pin(D/'broad_config.json')
 S=D/'serving_engine_v1';S.mkdir()
 common=adapt(F/'serving_engine_v1/common.py',S/'common.py',[
  ("BROAD_SHA = '0d660c686cf9fbe80ccd5ee3bb44a8c55685212a1b63efdcbfde74f7714cde78'","BROAD_SHA = "+repr(assessor['sha256']),1)])
 sr=adapt(F/'serving_engine_v1/run.py',S/'run.py',[])
 sa=adapt(F/'serving_engine_v1/assess.py',S/'assess.py',[])
 s=read(F/'serving_config_draft.json')
 assert s['schema']=='serving-qualification-v2' and s['receipt_prefix']=='serving_v2_' and s['prospective_authorization'] is False
 s.update(common=common,runner=sr,assessor=sa,broad_engine=assessor,broad_config=cp,candidate=c['candidate']['source'],
  release_tag=D.name+'_serving_v2',setup_counts=copy.deepcopy(c['candidate']['setup']['counts']))
 s['provenance']=[pin(__file__),cp,*c['provenance'],*[pin(F/'serving_engine_v1'/n) for n in ('common.py','run.py','assess.py')],pin(S/'common.diff')]
 s['interpretation']='Inactive own worker20/setup3/contract qualification after own new fixed40PASS>=1% vsS34; no reuse of FG serving pass or official result. Honest deferred521/eager128 setup. All original limits unchanged.'
 once(D/'serving_config_draft.json',s)
 light=H.parents[4]/'.venv/bin/python'
 commands=dict(admit_broad=[str(light),'-B',str(Path(__file__).resolve()),'--admit-broad'],
  broad_run=[str(NATIVE),'-B',runner['path'],cp['path'],cp['sha256'],'BROAD_ADMISSION_SHA'],
  broad_assess=[str(NATIVE),'-B',assessor['path'],cp['path'],cp['sha256'],'BROAD_ADMISSION_SHA'],
  enable_serving=[str(light),'-B',str(Path(__file__).resolve()),'--enable-serving'])
 once(D/'qualification_commands.json',dict(commands=commands,broad_config=cp,serving_draft=pin(D/'serving_config_draft.json'),
  launch_held_until_root_decision=True,own_native8_boundary_and_cheap_gate_required=True,serving_requires_new_full40_pass=True,models_executed=False))
 print(json.dumps(dict(broad_config=cp,broad_engine=assessor,serving_draft=pin(D/'serving_config_draft.json'),commands=pin(D/'qualification_commands.json'))))

if __name__=='__main__':main()
