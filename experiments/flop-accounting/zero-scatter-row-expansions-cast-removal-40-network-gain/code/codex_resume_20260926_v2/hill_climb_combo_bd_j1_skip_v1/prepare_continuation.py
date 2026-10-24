"""Reuse combo boundary recipe for exact BD j1; saved N2 outputs, no model execution."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, sys
D=Path(__file__).resolve().parent
H=D.parent
T=H/'hill_climb_s34_pair_read7_scan_v1'
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
    assert pin(old_assessor)['sha256']=='39d918d902805d022cc7a2301d633fce6639998e0b7792215ea50c040a1898c7'
    assert pin(old_config)['sha256']=='9ed746d6b44d341461896cd403fda5e0757d4bd5727a4e1c736fa4186a16ae81'
    pilot=pin(D/'native_screen.json')
    assert pilot['sha256']=='1f1bb28f549a93a6adbe6f055be3b3f4e86e433d50f6761a721e69627daf5a6f'
    nc=json.loads(Path(pilot['path']).read_text()); original=copy.deepcopy(nc)
    r=json.loads((D/'native2_assessment.json').read_text())
    assert r['config']==pilot and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==pin(D/'candidate.py') and r['parity_layers']==list(range(16))
    assert nc['parity_layers']==list(range(16)) and r['configured_parent_bytes_exact'] is True
    for size,wins in (('2',2),('8',8)):
        gates=nc['stages'][size]['gates']
        assert gates['adjusted_pct_at_most']==-.02 and gates['wins_min']==wins
        assert gates['max_per_network_regression_pct']==0.0 and gates['strict_adjusted'] is False
    build=json.loads((D/'build.json').read_text())
    assert build['candidate']==pin(D/'candidate.py') and build['source_inverse_exact']
    source=(D/'candidate.py').read_text(); parent=Path(build['parent']['path']).read_text()
    assert source.count(build['after'])==1 and source.replace(build['after'],build['before'],1)==parent
    assert build['parent']==nc['arms']['base']['source']==nc['arms']['parity_parent']['source']
    old=old_assessor.read_text(); new=old
    def replace_one(before,after):
        nonlocal new
        assert new.count(before)==1, before
        new=new.replace(before,after,1)
    replace_one('Prospective PP+READ7+scan boundary2','Prospective exact BD j1 boundary2')
    replace_one("'pair-read7-scan-s34-boundary-native2-v1'","'combo-bd-j1-s34-boundary-native2-v1'")
    start=new.index('    # Exact PP source + retained READ7 and scan fragments; GN reuse is excluded.')
    end=new.index("    assert nc['parity_layers']",start)
    proof="""    # Exact single BD edit over retained combo; score and official references differ.
    assert a['source_proof'] == c['build']
    parent = q.checked(build['parent']).decode()
    assert build['parent'] == c['native_comparator'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']
    assert build['parent']['sha256'] == '2212448b123d4be2dadf9c4c8642b49ad1cb3e073a1cc49684e4fc083a19c14f'
    source = q.checked(c['candidate']).decode()
    assert source.count(build['after']) == 1
    assert source.replace(build['after'], build['before'], 1) == parent
    assert build['changed_parent_functions'] == ['_bd_update']
    retained = json.loads(q.checked(c['score_parent_retained']))
    assert retained['status'] == 'retained_for_combination' and retained['candidate'] == build['parent']
    assert retained['native8_runtime_recheck']['passed'] and all(retained['score_gates'].values())
"""
    new=new[:start]+proof+new[end:]
    replace_one("    assert nc['parity_layers'] == r['parity_layers'] == [0]", "    assert nc['parity_layers'] == r['parity_layers'] == list(range(16))")
    replace_one("    assert bank['candidate'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']", "    assert nc['arms']['base']['source'] == nc['arms']['parity_parent']['source'] == c['native_comparator']")
    replace_one("    native_parent_rows = native['base']", """    s34_cfg = json.loads(q.checked(c['s34_native_config']))
    assert s34_cfg['engine'] == c['native_engine'] and s34_cfg['arms']['base']['source'] == c['parent']
    s34_arm = s34_cfg['arms']['base']
    s34_records = {key: s34_arm['blocks']['2']['artifacts'][key] for key in ('manifest', 'rows', 'guard')}
    native_parent_rows = native_arm(q, s34_arm, 2, s34_records)""")
    for name in ('paired_boundary_assessment.json','paired_boundary_assess_guard.json'):
        assert new.count(repr(name))==old.count(repr(name))==1
    compile(new,str(D/'boundary_assess_s34.py'),'exec')
    assessor=D/'boundary_assess_s34.py'
    with assessor.open('x') as f:f.write(new)
    diff=D/'boundary_adapter.diff'
    with diff.open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(old_assessor),tofile=str(assessor))))
    c=json.loads(old_config.read_text())
    for key in ('primitive','packed_build','packed_retained','read7_retained','read7_source'):c.pop(key)
    c.update(schema='combo-bd-j1-s34-boundary-native2-v1',engine=pin(assessor),candidate=pin(D/'candidate.py'),
        build=pin(D/'build.json'),source_protocol=pin(D/'protocol.json'),native_comparator=build['parent'],
        native_config_path=str(D/'native_screen_continuation.json'),tag=D.name+'_boundary_feasibility_v1',
        setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']),native2_gates=copy.deepcopy(nc['stages']['2']['gates']),
        score_parent_retained=pin(T/'retained_component.json'),s34_native_config=pin(T/'native_screen.json'))
    assert c['setup_counts']==dict(bound=687,lvfuse=649,psdb_fused=9,pair_block_indices=1,read7_fused=4)
    assert c['parent']['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    c['interpretation']='Own exact BD j1 boundary. Score comparator retained combo; original S34 residual and S32 runtime controls remain distinct. Original exact-component gates unchanged.'
    bp=D/'boundary_config.json';once(bp,c);bp=pin(bp)
    nc['stages']['2']['result']=str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard']=str(D/'native2_continuation_assess_guard.json')
    req=nc['boundary_before8'];req.update(receipt=str(D/'boundary_assessment.json'),assessor=pin(assessor),
        assessment_guard=str(D/'boundary_assess_guard.json'),assessment_command=[str(LIGHT),'-B',str(assessor),bp['path'],bp['sha256']])
    nc['continuation']=dict(pilot_config=pilot,pilot_result=pin(D/'native2_assessment.json'),pilot_assessment_guard=pin(D/'native2_assess_guard.json'),
        no_model_rerun=True,same_saved_native2_outputs=True,no_score_gate_changes=True)
    nc['provenance']+=[pin(__file__),pin(assessor),pin(diff),bp,pin(old_assessor),pin(old_config),
        *[c[k] for k in ('score_parent_retained','s34_native_config','parent_boundary','s34_bank','s34_evaluation')],
        *[nc['continuation'][k] for k in ('pilot_config','pilot_result','pilot_assessment_guard')]]
    nc['interpretation']='Exact BD j1 continuation: unchanged saved N2, own boundary, fixed six and original exact N8 gates. No broad/bank qualification.'
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
