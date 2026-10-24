"""Narrow prefix128 resource-only continuation; saved N2 and own boundary, no model execution."""
from pathlib import Path
import copy, difflib, hashlib, json, sys
D = Path(__file__).resolve().parent
H = D.parent
T = H/'hill_climb_scatter_forward_gather_v1'
BASE = T
LIGHT = H.parents[4]/'.venv/bin/python'
NATIVE = H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

def once(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write(chr(10))

def admit():
    cp = pin(D/'boundary_config.json'); c = json.loads(Path(cp['path']).read_text())
    np = pin(D/'native_screen_continuation.json'); nc = json.loads(Path(np['path']).read_text())
    rp = pin(nc['stages']['2']['result']); r = json.loads(Path(rp['path']).read_text())
    gp = pin(nc['stages']['2']['assessment_guard']); g = json.loads(Path(gp['path']).read_text())
    assert r['config'] == np and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate'] == c['candidate'] and r['all16_parent_bytes_exact'] and r['per_network_flops_equal']
    assert g['command'] == [str(NATIVE), '-B', nc['engine']['path'], np['path'], np['sha256'], '2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code'] == 0 and g['error'] is None
    assert g['max_swap_growth_bytes'] == 0 and not g['remaining_rows']
    once(D/'boundary_admission.json', dict(root_admitted=True, source_changes=False, serving_qualified=False,
        config=cp, assessor=c['engine'], candidate=c['candidate'], native_config=np, native2=rp,
        native2_guard=gp, source_proof=c['build'], additional6_requires_boundary_pass=True,
        original_pilot_preserved=True, no_model_rerun_for_reassessment=True, formal_receipt_created_after_census=True, prior_root_authorization_and_original_native2=nc['continuation']['pilot_result'], cheap_mechanism=c['root_cheap_gate'],
        official_residual_limit_s=.4, strict_residual_upper_bound=True, prospective_s34_reference=True))
    print(json.dumps(pin(D/'boundary_admission.json')))

PROOF = "    # Exact prefix128 setup schedule; locally qualified FG control, fixed S34/S32 resources.\n    assert a['source_proof'] == c['build']\n    assert c['native_comparator'] == build['source_inverse']['parent'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']\n    assert c['native_comparator']['sha256'] == '1056a1b79acedd1ab75bd711c4880b0507318db15448db48638e618baa7dcf4d'\n    source = q.checked(c['candidate']).decode()\n    inv = build['source_inverse']\n    assert inv['parent'] == c['native_comparator'] and inv['exact_inverse']\n    assert len(inv['edits']) == 1 and build['only_one_condition_changed'] and build['bind_order_unchanged']\n    for edit in reversed(inv['edits']):\n        assert source.count(edit['after']) == 1\n        source = source.replace(edit['after'], edit['before'], 1)\n    assert source == q.checked(c['native_comparator']).decode()\n    ei = json.loads(q.checked(c['engine_inverse']))\n    assert ei['engine'] == c['native_engine'] == nc['engine']\n    assert ei['parent'] == c['control_native_engine'] and ei['exact_inverse']\n    engine_source = q.checked(ei['engine']).decode()\n    assert len(ei['edits']) == 3\n    for edit in reversed(ei['edits']):\n        assert engine_source.count(edit['after']) == 1\n        engine_source = engine_source.replace(edit['after'], edit['before'], 1)\n    assert engine_source == q.checked(ei['parent']).decode()\n    assert ei['parent']['sha256'] == '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'\n    rp = json.loads(q.checked(c['resource_protocol']))\n    assert rp['candidate'] == c['candidate'] and rp['source_build'] == c['build'] and rp['engine_inverse'] == c['engine_inverse']\n    assert rp['resource_only'] and rp['native2_and8_gates'] == c['native2_gates']\n    assert nc['exact_flops_required'] and r['per_network_flops_equal']\n    assert r['adjusted_pct'] == r['raw_pct'] == 0\n    verify_cheap_evidence(q, c)\n    assert c['candidate_cache_limit_bytes'] == c['control_cache_limit_bytes'] == c['score_parent_cache_limit_bytes'] == 512000000\n    parent_n8 = json.loads(q.checked(c['score_parent_native8']))\n    assert parent_n8['candidate'] == c['native_comparator'] and parent_n8['admitted'] and parent_n8['passed_integrity'] and all(parent_n8['gates'].values())\n    assert parent_n8['all16_parent_bytes_exact']\n    parent_cfg = json.loads(q.checked(parent_n8['config']))\n    parent_ng = c['score_parent_native8_guard']\n    q.guard(parent_ng, [str(q.NATIVE), '-B', parent_cfg['engine']['path'], parent_n8['config']['path'], parent_n8['config']['sha256'], '8'], 120)\n    serving = json.loads(q.checked(c['score_parent_serving']))\n    assert serving['passed'] and serving['receipt_checks_passed'] and serving['worker_qualified'] and serving['release_qualified']\n    assert serving['candidate'] == c['native_comparator'] and serving['native8'] == c['score_parent_native8']\n    assert serving['long_worker']['networks'] == 20 and serving['full_estimator_contract']\n    assert serving['resources']['maximum_group_rss_bytes'] < 7500000000 and serving['resources']['maximum_swap_growth_bytes'] == 0\n    q.guard(c['score_parent_serving_guard'], [str(q.NATIVE), '-B', serving['assessor']['path'], serving['config']['path'], serving['config']['sha256'], serving['validation_admission']['sha256']], 120)\n    for record in (serving['assessor'], serving['config'], serving['validation_admission'], *serving['resources']['guards']):\n        q.checked(record)\n    for size in (2, 6):\n        parent_records = next(block for block in parent_n8['blocks'] if block['size'] == size)['arms']['candidate']\n        for role in ('base', 'parity_parent'):\n            native_arm(q, nc['arms'][role], size, parent_records, 512000000)\n            for entry in nc['arms'][role]['blocks'][str(size)]['artifacts']['predictions'].values():\n                q.checked(entry)\n    assert r['all16_parent_bytes_exact'] is True\n"

def main():
    if sys.argv[1:] == ['--admit-boundary']:
        admit(); return
    assert not sys.argv[1:]
    old_assessor = T/'boundary_assess_s34.py'; old_config = T/'boundary_config.json'
    assert pin(old_assessor)['sha256'] == '0e1c5c61be3724ecb241d2b6d581e151c010e3bccf287fc90d1846fc483c8f11'
    assert pin(old_config)['sha256'] == 'b9bd0af7576464861c18f14fde780f69c37af0751db77309409cdf7447aace16'
    pilot = pin(D/'native_screen_v2.json'); assert pilot['sha256'] == '58f2f8ec7fb81947d96adcf1bf6d66e03f894b57df515c06aa7eb8a57baf5406'
    nc = json.loads(Path(pilot['path']).read_text()); original = copy.deepcopy(nc)
    r = json.loads((D/'native2_assessment_v2.json').read_text())
    assert r['config'] == pilot and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate'] == pin(D/'candidate.py') and r['parity_layers'] == list(range(16))
    assert nc['parity_layers'] == list(range(16)) and r['configured_parent_bytes_exact'] and r['all16_parent_bytes_exact']
    for size,wins in (('2',0),('8',0)):
        assert nc['stages'][size]['gates'] == dict(adjusted_pct_at_most=0.,strict_adjusted=False,wins_min=wins,ci_upper_below=None,max_per_network_regression_pct=0.,additional6_gain_required=False)
    build=json.loads((D/'build.json').read_text())
    assert build['candidate']==pin(D/'candidate.py') and build['source_inverse']['parent']==pin(BASE/'candidate.py') and build['source_inverse']['exact_inverse']
    parent_cfg=json.loads((BASE/'native_screen_continuation.json').read_text())
    assert nc['arms']['base']['source']==nc['arms']['parity_parent']['source']==parent_cfg['arms']['candidate']['source']==build['source_inverse']['parent']
    for size in ('2','6'):
        block=parent_cfg['arms']['candidate']['blocks'][size]; folder=H/block['tag']
        artifacts=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(rank):pin(folder/f'pred_{rank}.npy') for rank in block['ranks']})
        for role in ('base','parity_parent'):
            actual=nc['arms'][role]['blocks'][size]
            assert {k:v for k,v in actual.items() if k!='artifacts'}=={k:v for k,v in block.items() if k!='artifacts'}
            assert actual['artifacts']==artifacts
    old=old_assessor.read_text(); new=old
    def replace_one(before,after):
        nonlocal new
        assert new.count(before)==1,before
        new=new.replace(before,after,1)
    replace_one('Prospective forward-cache plus unique-gather boundary2','Prospective prefix128 resource-only boundary2')
    replace_one("'forward-cache-unique-gather-s34-boundary-native2-v1'","'prefix128-resource-s34-boundary-native2-v1'")
    replace_one("640000000 if cfg['arms'][role]['source']['sha256'] == '0f7b93c4c45c447d76f04f428c1aeec066c5eb30027671653565b7c5fe3d93c3' else 512000000","512000000")
    start=new.index('    # Exact forward-cache restore then gather inverse; SC score control, S34 official reference.')
    end=new.index("    assert nc['parity_layers']",start)
    new=new[:start]+PROOF+new[end:]
    replace_one("assert build['candidate'] == c['candidate'] and build['source_inverse_exact'] is True", "assert build['candidate'] == c['candidate'] and build['source_inverse']['exact_inverse'] is True")
    replace_one("    q.guard(records['guard'], command, seconds)", "    guard_normalized(q, records['guard'], command, seconds)")
    replace_one("assert original_config['prospective_scope'] == 'native2_only'", "assert original_config['prospective_scope'] == 'native2_only_until_own_boundary_continuation'")
    replace_one("assert continuation['pilot_assessment_guard'] == pin(original_config['stages']['2']['assessment_guard'])", "assert continuation['pilot_assessment_guard'] == pin(d/'native2_assess_guard.json')  # Actual preserved guard; v2 config anticipated a different filename.")
    helper_text = (D/'boundary_evidence_helpers.py').read_text()
    replace_one('def require_preserved_failure(old):', helper_text+'\n\ndef require_preserved_failure(old):')
    for name in ('paired_boundary_assessment.json','paired_boundary_assess_guard.json'):
        assert new.count(repr(name))==old.count(repr(name))==1
    assessor=D/'boundary_assess_s34.py'; compile(new,str(assessor),'exec')
    with assessor.open('x') as f:f.write(new)
    diff=D/'boundary_adapter.diff'
    with diff.open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(old_assessor),tofile=str(assessor))))
    c=json.loads(old_config.read_text())
    for key in ('composition_parent_build','composition_parent_inverse','score_parent_build','parent_serving_failure','score_parent_retained','cache_build'):c.pop(key,None)
    c.update(schema='prefix128-resource-s34-boundary-native2-v1',engine=pin(assessor),candidate=pin(D/'candidate.py'),build=pin(D/'build.json'),source_protocol=pin(D/'protocol.json'),native_comparator=build['source_inverse']['parent'],native_config_path=str(D/'native_screen_continuation.json'),tag=D.name+'_boundary_feasibility_v1',setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']),native2_gates=copy.deepcopy(nc['stages']['2']['gates']),native_engine=nc['engine'],control_native_engine=pin(H/'native_screen_engine_v1/screen.py'),candidate_cache_limit_bytes=512000000,score_parent_cache_limit_bytes=512000000,control_cache_limit_bytes=512000000,score_parent_native8=pin(BASE/'native8_assessment.json'),score_parent_native8_guard=pin(BASE/'native8_assess_guard.json'),score_parent_serving=pin(BASE/'serving_v2_qualification_assessment.json'),score_parent_serving_guard=pin(BASE/'serving_v2_assess_guard.json'),engine_inverse=pin(D/'engine_inverse_v2.json'),resource_protocol=pin(D/'native_resource_protocol_v2.json'),cheap_gate=pin(D/'cheap_resource_gate_v1.json'),root_cheap_gate=pin(D/'cheap_mechanism_result_v1.json'),official_parent_failure=pin(BASE/'official_failure_333099_v1.json'),setup_evidence_helpers=pin(D/'boundary_evidence_helpers.py'))
    assert c['setup_counts']==dict(bound=687,lvfuse=521,psdb_fused=9,pair_block_indices=1,read7_fused=4)
    assert c['parent']['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    c['interpretation']='Prefix128 scheduling only: actual FG2+6 score controls, exact all16 and per-rank FLOP equality. Normal setup defers521 and eagerly fuses128, total649. Official S34 residual and S32 normal wall references unchanged. Degraded CPU is excluded from projections. FG local qualification is not an official bank; its failed333099 receipt is preserved. No broad, serving or release authority.'
    bp=D/'boundary_config.json';once(bp,c);bp=pin(bp)
    nc['stages']['2']['result']=str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard']=str(D/'native2_continuation_assess_guard.json')
    req=nc['boundary_before8'];req.update(receipt=str(D/'boundary_assessment.json'),assessor=pin(assessor),assessment_guard=str(D/'boundary_assess_guard.json'),assessment_command=[str(LIGHT),'-B',str(assessor),bp['path'],bp['sha256']])
    nc['continuation']=dict(pilot_config=pilot,pilot_result=pin(D/'native2_assessment_v2.json'),pilot_assessment_guard=pin(D/'native2_assess_guard.json'),no_model_rerun=True,same_saved_native2_outputs=True,no_score_gate_changes=True,parent2_and6_artifacts_unchanged=True)
    keys=('score_parent_native8','score_parent_native8_guard','score_parent_serving','score_parent_serving_guard','engine_inverse','resource_protocol','cheap_gate','root_cheap_gate','official_parent_failure','setup_evidence_helpers','s34_native_config','parent_boundary','s34_bank','s34_evaluation','control_native_engine')
    nc['provenance'] += [pin(__file__),pin(assessor),pin(diff),bp,pin(old_assessor),pin(old_config),*[c[k] for k in keys],*[nc['continuation'][k] for k in ('pilot_config','pilot_result','pilot_assessment_guard')]]
    nc['interpretation']=c['interpretation']
    nc['prospective_scope']='saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert all(nc[k]==original[k] for k in ('engine','parity_layers','future_only','raw_error_veto')) and nc['arms']==original['arms']
    assert all(nc['stages'][s][k]==original['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
    np=D/'native_screen_continuation.json';once(np,nc);np=pin(np);b6=nc['arms']['candidate']['blocks']['6']
    commands=dict(native2_reassessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2'],admission=[str(LIGHT),'-B',str(Path(__file__).resolve()),'--admit-boundary'],boundary_census=[str(LIGHT),'-B',c['driver']['path'],c['candidate']['path'],'35,100',c['tag']],boundary_assessment=req['assessment_command'],native6=[str(NATIVE),'-B',b6['driver']['path'],c['candidate']['path'],','.join(map(str,b6['ranks'])),b6['tag']],native8_assessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'8'])
    once(D/'continuation_commands.json',dict(commands=commands,config=np,boundary_config=bp,assessor=pin(assessor),native2_guard=nc['stages']['2']['assessment_guard'],boundary_guard=str(D/'boundary_guard.json'),boundary_log=str(D/'boundary.log'),boundary_assessment_guard=req['assessment_guard'],native6_guard=b6['guard'],native8_guard=nc['stages']['8']['assessment_guard']))
    print(json.dumps(dict(config=np,boundary=bp,assessor=pin(assessor),commands=pin(D/'continuation_commands.json'))))

if __name__=='__main__':main()
