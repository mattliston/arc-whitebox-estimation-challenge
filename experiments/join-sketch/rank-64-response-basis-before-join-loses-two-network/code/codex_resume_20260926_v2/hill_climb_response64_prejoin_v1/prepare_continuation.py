"""Narrow response64 saved-output continuation; source/stdlib only."""
from pathlib import Path
import copy, difflib, hashlib, json, sys
D = Path(__file__).resolve().parent
H = D.parent
BASE = H/'hill_climb_prefix128_fusion_v1'
LIGHT = H.parents[4]/'.venv/bin/python'
NATIVE = H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def pin(path):
    path=Path(path).resolve(); raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
def read(path): return json.loads(Path(path).read_text())
def once(path,value):
    with Path(path).open('x') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,indent=2,allow_nan=False)+'\n')
def checked(record):
    assert pin(record['path'])==record
    return read(record['path'])

def admit():
    cp=pin(D/'boundary_config.json');c=checked(cp)
    np=pin(D/'native_screen_continuation.json');nc=checked(np)
    rp=pin(nc['stages']['2']['result']);r=checked(rp)
    gp=pin(nc['stages']['2']['assessment_guard']);g=checked(gp)
    assert r['config']==np and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==c['candidate'] and r['parity_layers']==[0] and r['configured_parent_bytes_exact'] and r['all16_predictions_finite']
    assert g['command']==[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None
    assert g['max_swap_growth_bytes']==0 and not g['remaining_rows']
    assert 0<g['max_rss_bytes']<7500000000 and 0<g['elapsed_s']<=120
    assert not (D/'boundary_guard.json').exists(), 'Admit prospectively before own census'
    once(D/'boundary_admission.json',dict(root_admitted=True,source_changes=False,serving_qualified=False,
        config=cp,assessor=c['engine'],candidate=c['candidate'],native_config=np,native2=rp,native2_guard=gp,
        source_proof=c['build'],additional6_requires_boundary_pass=True,original_pilot_preserved=True,
        no_model_rerun_for_reassessment=True,formal_receipt_created_before_census=True,
        official_residual_limit_s=.4,strict_residual_upper_bound=True,retention_only=True))
    print(json.dumps(pin(D/'boundary_admission.json')))

def main():
    if sys.argv[1:]==['--admit-boundary']: admit();return
    assert not sys.argv[1:]
    old_assessor=BASE/'boundary_assess_s34.py';old_config=BASE/'boundary_config.json'
    assert pin(old_assessor)['sha256']=='5a1ae6608c57b76d150f3f7225ad1adc288afeee1ae74626b1c3c52cbf64a3a5'
    assert pin(old_config)['sha256']=='dfc41a22ef8058d1855489e1ba9c04a47f86683535d243ca1b3d748531ae8eeb'
    pilot=pin(D/'native_screen.json');assert pilot['sha256']=='8eca090fe4a3709eada9ffd9df314147d9b6f5fbb1ff97427ff4769324602240'
    nc=checked(pilot);original=copy.deepcopy(nc);r=read(D/'native2_assessment.json')
    assert r['config']==pilot and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate']==pin(D/'candidate.py') and r['parity_layers']==nc['parity_layers']==[0]
    assert r['configured_parent_bytes_exact'] and r['all16_predictions_finite']
    for size,wins,ci,add in (('2',1,None,False),('8',6,0,True)):
        assert nc['stages'][size]['gates']==dict(adjusted_pct_at_most=-.1,strict_adjusted=False,wins_min=wins,ci_upper_below=ci,max_per_network_regression_pct=1.,additional6_gain_required=add)
    build=read(D/'build.json');assert build['candidate']==pin(D/'candidate.py') and build['source_inverse']['parent']==pin(BASE/'candidate.py')
    parent_cfg=read(BASE/'native_screen_continuation.json')
    for size in ('2','6'):
        block=parent_cfg['arms']['candidate']['blocks'][size];folder=H/block['tag']
        artifacts=dict(manifest=pin(folder/'manifest.json'),rows=pin(folder/'rows.jsonl'),guard=pin(block['guard']),predictions={str(rank):pin(folder/f'pred_{rank}.npy') for rank in block['ranks']})
        for role in ('base','parity_parent'):
            actual=nc['arms'][role]['blocks'][size]
            assert {k:v for k,v in actual.items() if k!='artifacts'}=={k:v for k,v in block.items() if k!='artifacts'}
            if 'artifacts' in actual: assert actual['artifacts']==artifacts
            else: assert size=='6';actual['artifacts']=artifacts
    old=old_assessor.read_text();new=old
    def replace_one(before,after):
        nonlocal new
        assert new.count(before)==1,before
        new=new.replace(before,after,1)
    replace_one('Prospective prefix128 resource-only boundary2','Prospective response64 prejoin boundary2')
    replace_one("'prefix128-resource-s34-boundary-native2-v1'","'response64-prejoin-s34-boundary-native2-v1'")
    # Retain guard normalization; remove the unrelated ancestor fault-injection verifier.
    start=new.index('def verify_cheap_evidence(');end=new.index('def require_preserved_failure(',start)
    new=new[:start]+new[end:]
    start=new.index('    # Exact prefix128 setup schedule;');end=new.index("    assert c['candidate_cache_limit_bytes']",start)
    proof='''    # One response call plus helper suffix; P128 controls and original resource anchors.
    assert a['source_proof'] == c['build']
    assert c['native_comparator'] == build['source_inverse']['parent'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']
    assert c['native_comparator']['sha256'] == 'f9f4790b3c65f730404425e8923c767dc2f895fdcc1e14f14cc649286ad3b69f'
    source = q.checked(c['candidate']).decode()
    inv = build['source_inverse']
    assert inv['exact_inverse'] and len(inv['edits']) == 1 and source.endswith(inv['suffix'])
    source = source[:-len(inv['suffix'])]
    for edit in reversed(inv['edits']):
        assert source.count(edit['after']) == 1
        source = source.replace(edit['after'], edit['before'], 1)
    assert source == q.checked(c['native_comparator']).decode()
    ei = json.loads(q.checked(c['engine_inverse']))
    assert ei['engine'] == c['native_engine'] == nc['engine']
    assert ei['parent'] == c['control_native_engine'] and ei['exact_inverse']
    engine_source = q.checked(ei['engine']).decode()
    assert len(ei['edits']) == 1
    for edit in reversed(ei['edits']):
        assert engine_source.count(edit['after']) == 1
        engine_source = engine_source.replace(edit['after'], edit['before'], 1)
    assert engine_source == q.checked(ei['parent']).decode()
    assert ei['parent']['sha256'] == '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
    protocol = json.loads(q.checked(c['source_protocol']))
    assert protocol['candidate'] == c['candidate'] and protocol['parent'] == c['native_comparator']
    assert protocol['native2_gates'] == c['native2_gates'] and protocol['parity_layers'] == [0]
'''
    new=new[:start]+proof+new[end:]
    replace_one("    assert r['all16_parent_bytes_exact'] is True\n    assert nc['parity_layers'] == r['parity_layers'] == list(range(16))", "    assert nc['parity_layers'] == r['parity_layers'] == [0]")
    replace_one("assert original_config['prospective_scope'] == 'native2_only_until_own_boundary_continuation'", "assert original_config['prospective_scope'] == 'NATIVE2_ONLY'")
    replace_one("    assert nc['arms'] == original_config['arms']", "    restored_arms = json.loads(json.dumps(nc['arms']))\n    for role in ('base', 'parity_parent'):\n        assert 'artifacts' not in original_config['arms'][role]['blocks']['6']\n        del restored_arms[role]['blocks']['6']['artifacts']\n    assert restored_arms == original_config['arms']")
    replace_one("assert continuation['pilot_assessment_guard'] == pin(d/'native2_assess_guard.json')  # Actual preserved guard; v2 config anticipated a different filename.", "assert continuation['pilot_assessment_guard'] == pin(original_config['stages']['2']['assessment_guard'])")
    for name in ('paired_boundary_assessment.json','paired_boundary_assess_guard.json'):
        assert new.count(repr(name))==old.count(repr(name))==1
    # The official normal runtime/residual methodology is byte-identical.
    assert new[new.index('    control_pin = q.fixed'):]==old[old.index('    control_pin = q.fixed'):]
    assessor=D/'boundary_assess_s34.py';compile(new,str(assessor),'exec');once(assessor,new)
    diff=D/'boundary_adapter.diff';once(diff,''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(old_assessor),tofile=str(assessor))))
    c=read(old_config)
    for key in ('resource_protocol','cheap_gate','root_cheap_gate','official_parent_failure','setup_evidence_helpers'):c.pop(key,None)
    c.update(schema='response64-prejoin-s34-boundary-native2-v1',engine=pin(assessor),candidate=pin(D/'candidate.py'),build=pin(D/'build.json'),source_protocol=pin(D/'protocol.json'),native_comparator=pin(BASE/'candidate.py'),native_config_path=str(D/'native_screen_continuation.json'),tag=D.name+'_boundary_feasibility_v1',setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']),native2_gates=copy.deepcopy(nc['stages']['2']['gates']),native_engine=nc['engine'],control_native_engine=pin(H/'native_screen_engine_v1/screen.py'),score_parent_native8=pin(parent_cfg['stages']['8']['result']),score_parent_native8_guard=pin(parent_cfg['stages']['8']['assessment_guard']),score_parent_serving=pin(BASE/'serving_v2_qualification_assessment.json'),score_parent_serving_guard=pin(BASE/'serving_v2_assess_guard.json'),engine_inverse=pin(H/'hill_climb_prefix128_edge_bias_hoist_v1/engine_inverse.json'))
    c['interpretation']='Response64 retention only: actual P128 score controls; L0 parity and standard approximate N2/N8 gates. Own normal census; S34 official residual and S32 normal wall/CPU formula unchanged. P128 accepted grading status is not assumed successful. No broad, serving or release authority.'
    bp=D/'boundary_config.json';once(bp,c);bp=pin(bp)
    nc['stages']['2']['result']=str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard']=str(D/'native2_continuation_assess_guard.json')
    req=nc['boundary_before8'];req.update(receipt=str(D/'boundary_assessment.json'),assessor=pin(assessor),assessment_guard=str(D/'boundary_assess_guard.json'),assessment_command=[str(LIGHT),'-B',str(assessor),bp['path'],bp['sha256']])
    nc['continuation']=dict(pilot_config=pilot,pilot_result=pin(D/'native2_assessment.json'),pilot_assessment_guard=pin(D/'native2_assess_guard.json'),no_model_rerun=True,same_saved_native2_outputs=True,no_score_gate_changes=True,only_parent6_artifacts_added=True)
    keys=('score_parent_native8','score_parent_native8_guard','score_parent_serving','score_parent_serving_guard','engine_inverse','s34_native_config','parent_boundary','s34_bank','s34_evaluation','control_native_engine')
    nc['provenance'] += [pin(__file__),pin(assessor),pin(diff),bp,pin(old_assessor),pin(old_config),*[c[k] for k in keys],*[nc['continuation'][k] for k in ('pilot_config','pilot_result','pilot_assessment_guard')]]
    nc['interpretation']=c['interpretation'];nc['prospective_scope']='retention_only_saved_native2_own_boundary_fixed6_native8';nc['no_further_stage_authority']=False
    assert all(nc[k]==original[k] for k in ('engine','parity_layers','future_only','raw_error_veto'))
    assert all(nc['stages'][s][k]==original['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
    np=D/'native_screen_continuation.json';once(np,nc);np=pin(np);b6=nc['arms']['candidate']['blocks']['6']
    commands=dict(native2_reassessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2'],admission=[str(LIGHT),'-B',str(Path(__file__).resolve()),'--admit-boundary'],boundary_census=[str(LIGHT),'-B',c['driver']['path'],c['candidate']['path'],'35,100',c['tag']],boundary_assessment=req['assessment_command'],native6=[str(NATIVE),'-B',b6['driver']['path'],c['candidate']['path'],','.join(map(str,b6['ranks'])),b6['tag']],native8_assessment=[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'8'])
    once(D/'continuation_commands.json',dict(commands=commands,config=np,boundary_config=bp,assessor=pin(assessor),native2_guard=nc['stages']['2']['assessment_guard'],boundary_guard=str(D/'boundary_guard.json'),boundary_log=str(D/'boundary.log'),boundary_assessment_guard=req['assessment_guard'],native6_guard=b6['guard'],native8_guard=nc['stages']['8']['assessment_guard'],limits=dict(native2_reassessment=120,boundary_census=240,boundary_assessment=120,native6=300,native8_assessment=120,rss_bytes=7500000000,zero_added_swap=True),environment=dict.fromkeys(('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'),'1'),retention_only=True))
    print(json.dumps(dict(config=np,boundary=bp,assessor=pin(assessor),commands=pin(D/'continuation_commands.json'))))

if __name__=='__main__':main()
