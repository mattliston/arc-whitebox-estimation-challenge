"""Reuse retained delta boundary recipe for old-carried polarization; saved N2 outputs, no model execution."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, sys
D=Path(__file__).resolve().parent
H=D.parent
T=H/'hill_climb_combo_bd_linear_delta_v1'
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
    old_assessor=T/'boundary_assess_s34.py'; old_config=T/'boundary_config.json'
    assert pin(old_assessor)['sha256']=='5e4de6c04f0dc8f782a37b2d98d00755e576890b02cb524cc422402cacf6d44a'
    assert pin(old_config)['sha256']=='77ac67250e3ffad61957eb065f7ee8d281231f5ab3d244fa7c6ba37bf1414006'
    pilot=pin(D/'native_screen.json')
    assert pilot['sha256']=='49c1fa3bd3ab8884c48f84f883710b1c7058c369c7411830bf4d365d4b792170'
    nc=json.loads(Path(pilot['path']).read_text()); original=copy.deepcopy(nc)
    r=json.loads((D/'native2_assessment.json').read_text())
    assert r['config']==pilot and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==pin(D/'candidate.py') and r['parity_layers']==list(range(10))
    assert nc['parity_layers']==list(range(10)) and r['configured_parent_bytes_exact'] is True
    for size,wins in (('2',0),('8',6)):
        gates=nc['stages'][size]['gates']
        assert gates['adjusted_pct_at_most']==(1.0 if size=='2' else -.1) and gates['wins_min']==wins
        assert gates['max_per_network_regression_pct']==(2.0 if size=='2' else None) and gates['strict_adjusted'] is False
    build=json.loads((D/'build.json').read_text())
    assert build['candidate']==pin(D/'candidate.py') and build['source_inverse_exact']
    source=(D/'candidate.py').read_text(); parent=Path(build['parent']['path']).read_text()
    inv=json.loads((D/'source_inverse.json').read_text())
    suffix='\n'+Path(inv['fragment']['path']).read_text();assert source.endswith(suffix); restored=source[:-len(suffix)]
    for edit in reversed(inv['edits']):
        assert restored.count(edit['after'])==1;restored=restored.replace(edit['after'],edit['before'],1)
    assert restored==parent
    assert build['comparator']==nc['arms']['base']['source']==nc['arms']['parity_parent']['source']
    old=old_assessor.read_text(); new=old
    def replace_one(before,after):
        nonlocal new
        assert new.count(before)==1, before
        new=new.replace(before,after,1)
    replace_one('Prospective BD linear plus delta boundary2','Prospective old-carried polarization boundary2')
    replace_one("'combo-bd-linear-delta-s34-boundary-native2-v1'","'old-polar-s34-boundary-native2-v1'")
    start=new.index("    repair = json.loads(q.checked(c['metadata_repair']))")
    end=new.index("    assert c['native8_runtime_recheck_required']",start)
    new=new[:start]+new[end:]
    replace_one("original['engine'] == c['native_engine']", "original['engine'] == c['control_native_engine']")
    replace_one("s34_cfg['engine'] == c['native_engine']", "s34_cfg['engine'] == c['control_native_engine']")
    start=new.index('    # Two exact source inverses: delta -> fixed linear fusion -> retained BD1078.')
    end=new.index("    assert nc['parity_layers']",start)
    proof="""    # Old-polar source inverse and narrowly changed stage2 config assertion.
    assert a['source_proof'] == c['build']
    assert build['parent'] == build['comparator'] == c['native_comparator'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']
    assert build['parent']['sha256'] == '6f104e9db5ca32dc5e7724d64e03fd969bfff9d6cb48ce2b55695a998eecd1bb'
    source = q.checked(c['candidate']).decode()
    inv = json.loads(q.checked(build['source_inverse']))
    assert inv['parent'] == build['parent'] and inv['exact_inverse'] and inv['fragment'] == build['fragment']
    suffix = '\\n' + q.checked(inv['fragment']).decode()
    assert source.endswith(suffix); source = source[:-len(suffix)]
    for edit in reversed(inv['edits']):
        assert source.count(edit['after']) == 1
        source = source.replace(edit['after'], edit['before'], 1)
    assert source == q.checked(build['parent']).decode()
    ei = json.loads(q.checked(build['engine_inverse']))
    assert c['control_native_engine'] == ei['parent']
    assert c['control_native_engine']['sha256'] == '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
    assert ei['engine'] == c['native_engine'] == build['engine'] and ei['exact_inverse']
    custom = q.checked(ei['engine']).decode()
    assert custom.count(ei['after']) == 1 and custom.replace(ei['after'], ei['before'], 1) == q.checked(ei['parent']).decode()
    retained = json.loads(q.checked(c['score_parent_retained']))
    assert retained['status'] == 'RETAINED_COMPONENT' and retained['candidate'] == build['parent']
    parent_n8 = json.loads(q.checked(retained['native8']))
    parent_boundary = json.loads(q.checked(retained['boundary']))
    parent_raw = json.loads(q.checked(retained['native8_raw_review']))
    assert parent_n8['candidate'] == build['parent'] and parent_n8['admitted'] and parent_n8['passed_integrity'] and all(parent_n8['gates'].values())
    assert parent_boundary['candidate'] == build['parent'] and parent_boundary['passed'] and parent_boundary['receipt_checks_passed']
    assert parent_raw['passed'] and parent_raw['native8'] == retained['native8']
"""
    new=new[:start]+proof+new[end:]
    replace_one("    assert nc['parity_layers'] == r['parity_layers'] == [0]", "    assert nc['parity_layers'] == r['parity_layers'] == list(range(10))")
    for name in ('paired_boundary_assessment.json','paired_boundary_assess_guard.json'):
        assert new.count(repr(name))==old.count(repr(name))==1
    compile(new,str(D/'boundary_assess_s34.py'),'exec')
    assessor=D/'boundary_assess_s34.py'
    with assessor.open('x') as f:f.write(new)
    diff=D/'boundary_adapter.diff'
    with diff.open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(old_assessor),tofile=str(assessor))))
    c=json.loads(old_config.read_text())
    for key in ('metadata_repair','linear_build','linear_native2','root_increment_review'):c.pop(key)
    c.update(schema='old-polar-s34-boundary-native2-v1',engine=pin(assessor),candidate=pin(D/'candidate.py'),
        build=pin(D/'build.json'),source_protocol=pin(D/'protocol.json'),native_comparator=build['comparator'],
        native_config_path=str(D/'native_screen_continuation.json'),tag=D.name+'_boundary_feasibility_v1',
        setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']),native2_gates=copy.deepcopy(nc['stages']['2']['gates']),
        score_parent_retained=pin(T/'retained_component.json'),native_engine=nc['engine'],control_native_engine=pin(H/'native_screen_engine_v1/screen.py'))
    assert c['setup_counts']==dict(bound=687,lvfuse=649,psdb_fused=9,pair_block_indices=1,read7_fused=4)
    assert c['parent']['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    c['interpretation']='Own old-carried polarization boundary. Score comparator retained delta; original S34 residual and S32 runtime controls use their separately pinned original engine. Prospective N2 falsification and strict N8 unchanged; no new raw veto.'
    bp=D/'boundary_config.json';once(bp,c);bp=pin(bp)
    nc['stages']['2']['result']=str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard']=str(D/'native2_continuation_assess_guard.json')
    req=nc['boundary_before8'];req.update(receipt=str(D/'boundary_assessment.json'),assessor=pin(assessor),
        assessment_guard=str(D/'boundary_assess_guard.json'),assessment_command=[str(LIGHT),'-B',str(assessor),bp['path'],bp['sha256']])
    nc['continuation']=dict(pilot_config=pilot,pilot_result=pin(D/'native2_assessment.json'),pilot_assessment_guard=pin(D/'native2_assess_guard.json'),
        no_model_rerun=True,same_saved_native2_outputs=True,no_score_gate_changes=True)
    nc['provenance']+=[pin(__file__),pin(assessor),pin(diff),bp,pin(old_assessor),pin(old_config),
        *[c[k] for k in ('score_parent_retained','s34_native_config','parent_boundary','s34_bank','s34_evaluation','control_native_engine')],
        *[nc['continuation'][k] for k in ('pilot_config','pilot_result','pilot_assessment_guard')]]
    nc['interpretation']='Old-carried polarization: same saved N2, own boundary, then fixed six under original strict N8 gate. No broad/bank qualification.'
    nc['prospective_scope']='saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert all(nc[k]==original[k] for k in ('engine','arms','parity_layers','future_only','raw_error_veto'))
    assert all(nc['stages'][s][k]==original['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
    np=D/'native_screen_continuation.json';once(np,nc);np=pin(np)
    b6=nc['arms']['candidate']['blocks']['6']
    commands=dict(native2_reassessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2'],
        admission=[str(LIGHT),'-B',str(Path(__file__).resolve()),'--admit-boundary'],
        boundary_census=[str(LIGHT),'-B',c['driver']['path'],c['candidate']['path'],'35,100',c['tag']],
        boundary_assessment=req['assessment_command'],
        native6=[str(NATIVE),'-B',b6['driver']['path'],c['candidate']['path'],','.join(map(str,b6['ranks'])),b6['tag']],
        native8_assessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'8'])
    once(D/'continuation_commands.json',dict(commands=commands,config=np,boundary_config=bp,assessor=pin(assessor),
        native2_guard=nc['stages']['2']['assessment_guard'],boundary_guard=str(D/'boundary_guard.json'),
        boundary_log=str(D/'boundary.log'),boundary_assessment_guard=req['assessment_guard'],native6_guard=b6['guard'],
        native8_guard=nc['stages']['8']['assessment_guard']))
    print(json.dumps(dict(config=np,boundary=bp,assessor=pin(assessor),commands=pin(D/'continuation_commands.json'))))

if __name__=='__main__':main()
