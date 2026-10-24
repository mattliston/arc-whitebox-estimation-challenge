"""Minimal saved-N2 continuation: READ7 score, S34 residual, original S32 runtime."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json
import sys

D = Path(__file__).resolve().parent
H = D.parent
P = H/'hill_climb_s34_read7_v1'
LIGHT = H.parents[4]/'.venv/bin/python'
NATIVE = H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def pin(p):
    p = Path(p).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))

def once(p, x):
    with p.open('x') as f: json.dump(x, f, indent=2, allow_nan=False); f.write('\n')

def replace(s, before, after, count=1):
    assert s.count(before) == count, (before, s.count(before))
    return s.replace(before, after)

def source():
    old = (P/'boundary_assess_s34_v3.py').read_text()
    assert pin(P/'boundary_assess_s34_v3.py')['sha256'] == 'd01488921900d486301a83526ef174eda9a03c3578a43bf7a753a79d6d3ccaf2'
    new = old
    for before, after, count in (
        ("'read7-s34-boundary-native2-v1'", "'read7-kpad-boundary-native2-v1'", 1),
        ("'boundary_admission_read7_v3.json'", "'boundary_admission.json'", 1),
        ("'boundary_assessment_read7_v3.json'", "'boundary_assessment.json'", 2),
        ("'boundary_assess_guard_read7_v3.json'", "'boundary_assess_guard.json'", 1)):
        new = replace(new, before, after, count)
    new = replace(new, '# Appended READ7 implementation: exact S34 prefix and actual N2 layer-0 parity.',
        '# Appended K-padding extension: exact retained-READ7 prefix and actual N2 layer-0 parity.')
    new = replace(new, "assert build['parent'] == c['parent'] == nc['arms']['base']['source']",
        "assert build['parent'] == c['source_parent'] == nc['arms']['base']['source']")
    new = replace(new, "assert build['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'",
        "assert build['parent']['sha256'] == '9ea3a975561d6fae2ae8806b9812d81fe7cadc23845f86cce14b80696d0bf7d6'")
    new = replace(new, "assert build['fragment']['sha256'] == 'af467decdcac5ac8e005248984cd10220977df5ac75c27f660f5ec173932ca1e'",
        "assert build['fragment']['sha256'] == '"+pin(D/'fragment.py')['sha256']+"'")
    start = new.index("    recovery = c['setup_recovery']\n")
    end = new.index("    assert nc['arms']['candidate']['setup']['counts']", start)
    new = new[:start] + "    setup = m.setup_record(q, d/'boundary.log', c['setup_counts']['bound'], c['setup_counts']['lvfuse'])\n" + new[end:]
    new = replace(new,
        "setup_recovery=dict(contract=recovery, launcher=setup_launch_pin, guard=setup_gp, original_setup_stdout_missing=True, prediction_rerun=False), ", '')
    new = replace(new, ", setup_guard['max_rss_bytes']", '')
    new = replace(new, "assert bank['candidate'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']",
        "assert bank['candidate'] == c['parent']\n    assert c['source_parent'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']")
    new = replace(new, "    native_parent_rows = native['base']\n", '''    # Score comparator is retained READ7; residual controls remain official S34.
    residual_native_config = json.loads(q.checked(c['residual_native_config']))
    assert residual_native_config['engine'] == c['native_engine']
    residual_arm = residual_native_config['arms']['base']
    assert residual_arm['source'] == c['parent']
    residual_records = {key: residual_arm['blocks']['2']['artifacts'][key] for key in ('manifest', 'rows', 'guard')}
    native_parent_rows = native_arm(q, residual_arm, 2, residual_records)
''')
    new = replace(new, "parent_boundary=c['parent_boundary'], parent_census=parent_receipt['current_census'], parent_guard=parent_receipt['guard'],",
        "parent_boundary=c['parent_boundary'], parent_census=parent_receipt['current_census'], parent_guard=parent_receipt['guard'],\n            native_config=c['residual_native_config'], native_control_rows=residual_records['rows'],")
    ast.parse(new); compile(new, str(D/'boundary_assess_s34.py'), 'exec')
    def paired_literals(s):
        return sorted(n.value for n in ast.walk(ast.parse(s)) if isinstance(n, ast.Constant)
            and isinstance(n.value, str) and 'paired_' in n.value)
    assert paired_literals(old) == paired_literals(new)
    assert "q.fixed(V1/'paired_boundary_assessment.json', CONTROL_SHA)" in new
    assert "q.fixed(V1/'paired_boundary_assess_guard.json', CONTROL_GUARD_SHA)" in new
    assert 'setup_recovery' not in new and 'setup_guard' not in new
    assert "['s34_telemetry']" not in new
    return old, new

def admit():
    cp=pin(D/'boundary_config.json'); c=json.loads(Path(cp['path']).read_text())
    np=pin(D/'native_screen_continuation.json'); nc=json.loads(Path(np['path']).read_text())
    rp=pin(nc['stages']['2']['result']); r=json.loads(Path(rp['path']).read_text())
    assert r['config']==np and r['candidate']==c['candidate'] and r['passed_integrity'] and r['admitted'] and all(r['gates'].values())
    gp=pin(nc['stages']['2']['assessment_guard']); g=json.loads(Path(gp['path']).read_text())
    assert g['command']==[str(NATIVE),'-B',nc['engine']['path'],np['path'],np['sha256'],'2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code']==0 and g['error'] is None and g['max_swap_growth_bytes']==0
    once(D/'boundary_admission.json',dict(root_admitted=True,source_changes=False,serving_qualified=False,
        config=cp,assessor=c['engine'],candidate=c['candidate'],native_config=np,native2=rp,native2_guard=gp,
        source_proof=c['build'],original_pilot_preserved=True,no_model_rerun_for_reassessment=True,
        score_comparator=c['source_parent'],residual_reference=c['parent'],runtime_reference=c['runtime_reference'],
        additional6_requires_boundary_pass=True))
    print(json.dumps(pin(D/'boundary_admission.json')))

def main():
    if sys.argv[1:]==['--admit-boundary']: admit();return
    assert not sys.argv[1:]
    pilot=pin(D/'native_screen.json');assert pilot['sha256']=='b9636a5396372eacdda1c38c42b42de666da5f217dfc5d3ff4548f8e58f71c93'
    nc=json.loads(Path(pilot['path']).read_text());original=copy.deepcopy(nc)
    r=json.loads((D/'native2_assessment.json').read_text())
    assert r['config']==pilot and r['admitted'] and r['passed_integrity'] and r['parity_layers']==[0]
    assert r['candidate']==pin(D/'candidate.py') and r['comparator']==pin(P/'candidate.py')
    old,new=source();engine=D/'boundary_assess_s34.py'
    with engine.open('x') as f:f.write(new)
    diff=D/'boundary_adapter.diff'
    with diff.open('x') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(P/'boundary_assess_s34_v3.py'),tofile=str(engine))))
    c=json.loads((P/'boundary_config_v3.json').read_text());c.pop('setup_recovery')
    c.update(schema='read7-kpad-boundary-native2-v1',prospective_authorization=True,engine=pin(engine),
        candidate=pin(D/'candidate.py'),build=pin(D/'build.json'),source_protocol=pin(D/'protocol.json'),
        native_config_path=str(D/'native_screen_continuation.json'),native_comparator=pin(P/'candidate.py'),
        source_parent=pin(P/'candidate.py'),residual_native_config=pin(P/'native_screen.json'),
        tag=D.name+'_boundary_feasibility_v1',setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']),
        native2_gates=copy.deepcopy(nc['stages']['2']['gates']))
    assert c['setup_counts']==dict(bound=698,lvfuse=649,psdb_fused=9,read7_fused=4,read7_extended_fused=12)
    c['interpretation']='Kpad own boundary after saved-N2 reassessment: retained READ7 score comparator; official S34 residual plus4x candidate-minus-S34 first-census delta; unchanged original S32 wall/CPU controls. Strict residual<.4,wall105,setup4.1,7.5GB,zeroSwap. No inherited boundary/serving pass.'
    boundary=D/'boundary_config.json';once(boundary,c);bp=pin(boundary)
    nc['stages']['2']['result']=str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard']=str(D/'native2_continuation_assess_guard.json')
    nc['boundary_before8'].update(receipt=str(D/'boundary_assessment.json'),assessor=pin(engine),
        assessment_guard=str(D/'boundary_assess_guard.json'),
        assessment_command=[str(LIGHT),'-B',str(engine),str(boundary),bp['sha256']])
    nc['continuation']=dict(pilot_config=pilot,pilot_result=pin(D/'native2_assessment.json'),
        pilot_assessment_guard=pin(D/'native2_assess_guard.json'),no_model_rerun=True,same_saved_native2_outputs=True,no_score_gate_changes=True)
    nc['provenance'] += [pin(__file__),pin(engine),pin(diff),bp,pin(P/'boundary_assess_s34_v3.py'),
        *[c[k] for k in ('source_parent','residual_native_config','parent_boundary','s34_bank','s34_evaluation')],
        *[nc['continuation'][k] for k in ('pilot_config','pilot_result','pilot_assessment_guard')]]
    nc['interpretation']='Prospective Kpad continuation: same saved N2, own boundary, then fixed six and original N8 gates; score comparator retained READ7, S34 residual controls, original S32 runtime controls. No broad/bank authority.'
    nc['prospective_scope']='saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert all(nc[k]==original[k] for k in ('engine','arms','parity_layers','future_only','raw_error_veto'))
    assert all(nc['stages'][s][k]==original['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
    output=D/'native_screen_continuation.json';once(output,nc);np=pin(output)
    commands=dict(native2_reassessment=[str(NATIVE),'-B',nc['engine']['path'],str(output),np['sha256'],'2'],
        admission=[str(LIGHT),'-B',str(Path(__file__).resolve()),'--admit-boundary'],
        boundary=[str(LIGHT),'-B',c['driver']['path'],c['candidate']['path'],'35,100',c['tag']],
        boundary_assessment=nc['boundary_before8']['assessment_command'],
        native8_assessment=[str(NATIVE),'-B',nc['engine']['path'],str(output),np['sha256'],'8'])
    once(D/'continuation_commands.json',dict(commands=commands,config=np,boundary_config=bp,assessor=pin(engine),
        native2_guard=nc['stages']['2']['assessment_guard'],boundary_guard=str(D/'boundary_guard.json'),
        boundary_log=str(D/'boundary.log'),boundary_assessment_guard=nc['boundary_before8']['assessment_guard']))
    assert pin(D/'native_screen.json')==pilot
    print(json.dumps(dict(config=np,boundary=bp,assessor=pin(engine),builder=pin(__file__))))

if __name__=='__main__':main()
