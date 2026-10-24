"""Reuse READ7 v3 boundary recipe for GN pair reuse; saved N2 outputs, no model execution."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, sys
D=Path(__file__).resolve().parent
H=D.parent
T=H/'hill_climb_s34_read7_v1'
LIGHT=H.parents[4]/'.venv/bin/python'
NATIVE=H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def pin(path):
    path=Path(path).resolve(); raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))

def once(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n')

def admit():
    cp=pin(D/'boundary_config.json'); c=json.loads(Path(cp['path']).read_text())
    np=pin(D/'native_screen_continuation.json'); nc=json.loads(Path(np['path']).read_text())
    rp=pin(nc['stages']['2']['result']); r=json.loads(Path(rp['path']).read_text())
    gp=pin(nc['stages']['2']['assessment_guard']); g=json.loads(Path(gp['path']).read_text())
    assert r['config']==np and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==c['candidate']
    assert g['command']==[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None
    assert g['max_swap_growth_bytes']==0
    once(D/'boundary_admission.json',dict(root_admitted=True,source_changes=False,serving_qualified=False,
        config=cp,assessor=c['engine'],candidate=c['candidate'],native_config=np,native2=rp,
        native2_guard=gp,source_proof=c['build'],additional6_requires_boundary_pass=True,
        original_pilot_preserved=True,no_model_rerun_for_reassessment=True,
        official_residual_limit_s=.4,strict_residual_upper_bound=True,prospective_s34_reference=True))
    print(json.dumps(pin(D/'boundary_admission.json')))

def main():
    if sys.argv[1:]==['--admit-boundary']:
        admit(); return
    assert not sys.argv[1:]
    old_assessor=T/'boundary_assess_s34_v3.py'; old_config=T/'boundary_config_v3.json'
    assert pin(old_assessor)['sha256']=='d01488921900d486301a83526ef174eda9a03c3578a43bf7a753a79d6d3ccaf2'
    assert pin(old_config)['sha256']=='b70f857100b0f498cf8831921fc9fa8878274a9b2249347a1877a1c8783bbe40'
    pilot=pin(D/'native_screen.json')
    assert pilot['sha256']=='b73e2dd828aefd7581efcc9266173aa3e7e490a732d90b88658ad1db08f2356f'
    nc=json.loads(Path(pilot['path']).read_text()); original=copy.deepcopy(nc)
    r=json.loads((D/'native2_assessment.json').read_text())
    assert r['config']==pilot and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==pin(D/'candidate.py') and r['parity_layers']==list(range(16))
    assert nc['parity_layers'] == list(range(16))
    for size, wins in (('2', 2), ('8', 8)):
        gates=nc['stages'][size]['gates']
        assert gates['adjusted_pct_at_most'] == -.02 and gates['wins_min'] == wins
        assert gates['max_per_network_regression_pct'] == 0 and gates['strict_adjusted'] is False
    assert r['raw_pct'] == 0 and r['configured_parent_bytes_exact'] is True
    build=json.loads((D/'build.json').read_text())
    old=old_assessor.read_text(); new=old
    replacements=[
        ('Prospective READ7 boundary2','Prospective GN pair reuse boundary2',1),
        ("'read7-s34-boundary-native2-v1'","'gn-pair-reuse-s34-boundary-native2-v1'",1),
        ('# Appended READ7 implementation: exact S34 prefix and actual N2 layer-0 parity.',
         '# Appended GN pair reuse assignments: exact S34 prefix and actual N2 all16 parity.',1),
        ("assert build['fragment']['sha256'] == 'af467decdcac5ac8e005248984cd10220977df5ac75c27f660f5ec173932ca1e'",
         "assert build['fragment']['sha256'] == "+repr(build['fragment']['sha256']),1),
        ("assert nc['parity_layers'] == r['parity_layers'] == [0]",
         "assert nc['parity_layers'] == r['parity_layers'] == list(range(16))",1),
        ("'boundary_admission_read7_v3.json'","'boundary_admission.json'",1),
        ("'boundary_assessment_read7_v3.json'","'boundary_assessment.json'",2),
        ("'boundary_assess_guard_read7_v3.json'","'boundary_assess_guard.json'",1),
        ("setup=setup, setup_recovery=dict(contract=recovery, launcher=setup_launch_pin, guard=setup_gp, original_setup_stdout_missing=True, prediction_rerun=False),",
         'setup=setup,',1),
        (", setup_guard['max_rss_bytes']",'',1)]
    for before,after,count in replacements:
        assert new.count(before)==count,(before,new.count(before))
        new=new.replace(before,after,count)
    start=new.index("    recovery = c['setup_recovery']")
    end=new.index("    assert nc['arms']['candidate']['setup']['counts']",start)
    new=new[:start]+"    setup = m.setup_record(q, d/'boundary.log', c['setup_counts']['bound'], c['setup_counts']['lvfuse'])\n"+new[end:]
    assert 'setup_recovery' not in new and 'setup_guard' not in new
    for name in ('paired_boundary_assessment.json','paired_boundary_assess_guard.json'):
        assert new.count(repr(name))==old.count(repr(name))==1
    ast.parse(new)
    assessor=D/'boundary_assess_s34.py'
    with assessor.open('x') as f: f.write(new)
    diff=D/'boundary_adapter.diff'
    with diff.open('x') as f:
        f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(old_assessor),tofile=str(assessor))))
    c=json.loads(old_config.read_text()); c.pop('setup_recovery')
    c.update(schema='gn-pair-reuse-s34-boundary-native2-v1',engine=pin(assessor),candidate=pin(D/'candidate.py'),
        build=pin(D/'build.json'),source_protocol=pin(D/'protocol.json'),
        native_config_path=str(D/'native_screen_continuation.json'),tag=D.name+'_boundary_feasibility_v1',
        setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']),native2_gates=copy.deepcopy(nc['stages']['2']['gates']))
    assert c['setup_counts']==dict(bound=682,lvfuse=649,psdb_fused=9)
    c['interpretation']='Own GN pair reuse boundary, unchanged S34 residual and S32 runtime recipe; complete boundary.log, no setup recovery or primitive requirement.'
    bp=D/'boundary_config.json'; once(bp,c); bp=pin(bp)
    nc['stages']['2']['result']=str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard']=str(D/'native2_continuation_assess_guard.json')
    req=nc['boundary_before8']
    req.update(receipt=str(D/'boundary_assessment.json'),assessor=pin(assessor),
        assessment_guard=str(D/'boundary_assess_guard.json'),
        assessment_command=[str(LIGHT),'-B',str(assessor),bp['path'],bp['sha256']])
    nc['continuation']=dict(pilot_config=pilot,pilot_result=pin(D/'native2_assessment.json'),
        pilot_assessment_guard=pin(D/'native2_assess_guard.json'),no_model_rerun=True,
        same_saved_native2_outputs=True,no_score_gate_changes=True)
    nc['provenance']+=[pin(__file__),pin(assessor),pin(diff),bp,pin(old_assessor),pin(old_config),
        *[c[k] for k in ('parent_boundary','s34_bank','s34_evaluation')],
        *[nc['continuation'][k] for k in ('pilot_config','pilot_result','pilot_assessment_guard')]]
    nc['interpretation']='GN pair reuse continuation: same saved N2, own boundary, then fixed six under original exact-prediction N8 gates; no broad/bank authority.'
    nc['prospective_scope']='saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert all(nc[k]==original[k] for k in ('engine','arms','parity_layers','future_only','raw_error_veto'))
    assert all(nc['stages'][s][k]==original['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
    np=D/'native_screen_continuation.json'; once(np,nc); np=pin(np)
    b6=nc['arms']['candidate']['blocks']['6']
    commands=dict(native2_reassessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2'],
        admission=[str(LIGHT),'-B',str(Path(__file__).resolve()),'--admit-boundary'],boundary_assessment=req['assessment_command'],
        native6=[str(NATIVE),'-B',b6['driver']['path'],c['candidate']['path'],','.join(map(str,b6['ranks'])),b6['tag']],
        native8_assessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'8'])
    once(D/'continuation_commands.json',dict(commands=commands,config=np,boundary_config=bp,assessor=pin(assessor),
        native2_guard=nc['stages']['2']['assessment_guard'],boundary_guard=str(D/'boundary_guard.json'),
        boundary_log=str(D/'boundary.log'),boundary_assessment_guard=req['assessment_guard'],
        native6_guard=b6['guard'],native8_guard=nc['stages']['8']['assessment_guard']))
    print(json.dumps(dict(config=np,boundary=bp,assessor=pin(assessor),commands=pin(D/'continuation_commands.json'))))

if __name__=='__main__': main()
