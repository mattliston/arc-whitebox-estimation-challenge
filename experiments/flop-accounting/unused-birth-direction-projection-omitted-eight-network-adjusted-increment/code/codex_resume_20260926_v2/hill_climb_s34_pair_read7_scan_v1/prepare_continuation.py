"""Reuse packed-pair boundary recipe for PP+READ7+scan; saved N2 outputs, no model execution."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, sys
D=Path(__file__).resolve().parent
H=D.parent
T=H/'hill_climb_s34_pair_blocksym_v1'
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
    assert pin(old_assessor)['sha256']=='7f5bdfab67f9e32282f42982999437f5c7ace6851b12e7b90c2dc79dac64fc67'
    assert pin(old_config)['sha256']=='9f4c1bde1c4adb50cc4b2b36a57f366659ebfd59c40b11e71fef68e58aab8df4'
    pilot=pin(D/'native_screen.json')
    assert pilot['sha256']=='6ee9978d70b59bd2090c05e7b4ae5575030ae1ff0e804aceece9778a4e243605'
    nc=json.loads(Path(pilot['path']).read_text()); original=copy.deepcopy(nc)
    r=json.loads((D/'native2_assessment.json').read_text())
    assert r['config']==pilot and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==pin(D/'candidate.py') and r['parity_layers']==[0]
    assert nc['parity_layers']==[0] and r['configured_parent_bytes_exact'] is True
    for size,wins in (('2',1),('8',6)):
        gates=nc['stages'][size]['gates']
        assert gates['adjusted_pct_at_most']==-.1 and gates['wins_min']==wins
        assert gates['max_per_network_regression_pct']==(1.0 if size=='2' else None) and gates['strict_adjusted'] is False
    build=json.loads((D/'build.json').read_text())
    assert build['candidate']==pin(D/'candidate.py') and build['source_inverse_exact']
    old=old_assessor.read_text(); new=old
    for before,after in [('Prospective packed-pair boundary2','Prospective PP+READ7+scan boundary2'),
                         ("'packed-pair-s34-boundary-native2-v1'","'pair-read7-scan-s34-boundary-native2-v1'")]:
        assert new.count(before)==1; new=new.replace(before,after,1)
    start=new.index('    # Packed pair-series source inverse')
    end=new.index("    assert nc['parity_layers']",start)
    proof="""    # Exact PP source + retained READ7 and scan fragments; GN reuse is excluded.
    assert a['source_proof'] == c['build']
    parent = q.checked(build['parent']).decode()
    assert build['parent'] == c['parent'] == nc['arms']['base']['source']
    assert build['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    packed = q.checked(build['packed_source']).decode()
    appended = [q.checked(p).decode() for p in build['appended_fragments']]
    assert [p['sha256'] for p in build['appended_fragments']] == [
        'af467decdcac5ac8e005248984cd10220977df5ac75c27f660f5ec173932ca1e',
        '1fc78706a4f0fe0a6c59c25e9087e87ce7c2df1abda828359d1089e012aa6b1c']
    assert q.checked(c['candidate']).decode() == packed + ''.join('\\n'+x for x in appended)
    assert build['gn_reuse_absent'] and build['nystrom_absent']
    assert '_GNPR_PARENT_STATE' not in packed + ''.join(appended)
    assert '_ng_selected_gram' not in packed + ''.join(appended)
    pb = json.loads(q.checked(c['packed_build']))
    assert pb['candidate'] == build['packed_source'] and pb['parent'] == build['parent']
    inv = json.loads(q.checked(build['packed_inverse']))
    assert inv['parent'] == build['parent'] and inv['candidate'] == build['packed_source'] and inv['fragment'] == pb['fragment']
    fragment = q.checked(inv['fragment']).decode(); suffix = inv['suffix_separator'] + fragment
    assert packed.endswith(suffix); text = packed[:-len(suffix)]
    assert len(inv['edits']) == 2 and inv['changed_functions'] == ['_nonlinear_pair_batch', '_predict_binary']
    for edit in reversed(inv['edits']):
        assert text.count(edit['after']) == 1
        text = text.replace(edit['after'], edit['before'], 1)
    assert text == parent
    for key, source in (('packed_retained',build['packed_source']), ('read7_retained',c['read7_source'])):
        retained = json.loads(q.checked(c[key]))
        assert retained['status'] == 'retained_for_combination' and retained['candidate'] == source
        assert retained['native8_runtime_recheck']['passed'] and all(retained['score_gates'].values())
    assert q.checked(c['read7_source']).decode() == parent + '\\n' + appended[0]
    # The source-bound primitive belongs to the unchanged packed component.
    pr = json.loads(q.checked(c['primitive']['result']))
    pg = json.loads(q.checked(c['primitive']['guard']))
    pa = json.loads(q.checked(c['primitive']['admission']))
    assert pr['passed'] is True and pr['candidate'] == build['packed_source'] and pr['build'] == c['packed_build']
    assert pr['admission'] == c['primitive']['admission'] and pr['probe'] == c['primitive']['probe']
    assert pa['candidate'] == build['packed_source'] and pa['build'] == c['packed_build'] and pa['probe'] == c['primitive']['probe']
    assert pg['command'] == [str(q.NATIVE), '-B', c['primitive']['probe']['path'], c['primitive']['admission']['sha256'], c['primitive']['result']['path']]
    assert pg['completed'] and pg['whole_group_retired'] and pg['exit_code'] == 0 and pg['error'] is None
    assert pg['elapsed_s'] <= 60 and pg['rss_trigger_bytes'] == 1000000000 and pg['max_rss_bytes'] <= 1000000000
    assert pg['max_swap_growth_bytes'] == 0 and not pg['remaining_rows']
"""
    new=new[:start]+proof+new[end:]
    for name in ('paired_boundary_assessment.json','paired_boundary_assess_guard.json'):
        assert new.count(repr(name))==old.count(repr(name))==1
    compile(new,str(D/'boundary_assess_s34.py'),'exec')
    assessor=D/'boundary_assess_s34.py'
    with assessor.open('x') as f: f.write(new)
    diff=D/'boundary_adapter.diff'
    with diff.open('x') as f:
        f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(old_assessor),tofile=str(assessor))))
    c=json.loads(old_config.read_text()); c.pop('source_inverse')
    c.update(schema='pair-read7-scan-s34-boundary-native2-v1',engine=pin(assessor),candidate=pin(D/'candidate.py'),
        build=pin(D/'build.json'),source_protocol=pin(D/'protocol.json'),
        native_config_path=str(D/'native_screen_continuation.json'),tag=D.name+'_boundary_feasibility_v1',
        setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']),native2_gates=copy.deepcopy(nc['stages']['2']['gates']))
    assert c['setup_counts']==dict(bound=687,lvfuse=649,psdb_fused=9,pair_block_indices=1,read7_fused=4)
    c.update(packed_build=pin(T/'build.json'),packed_retained=pin(T/'retained_component.json'),
        read7_retained=pin(H/'hill_climb_s34_read7_v1/retained_component.json'),
        read7_source=pin(H/'hill_climb_s34_read7_v1/candidate.py'))
    c['interpretation']='Own PP+READ7+scan boundary; unchanged S34 residual and S32 runtime recipe; retained components and packed primitive are provenance, not a substitute for own gates.'
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
        *[c[k] for k in ('packed_build','packed_retained','read7_retained','read7_source','parent_boundary','s34_bank','s34_evaluation')],
        *c['primitive'].values(),*[nc['continuation'][k] for k in ('pilot_config','pilot_result','pilot_assessment_guard')]]
    nc['interpretation']='PP+READ7+scan continuation: same saved N2, own boundary, fixed six under original N8 gates; no broad/bank qualification.'
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
