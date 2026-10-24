"""Technical-only primitive guard repair; reuse every saved numerical artifact."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, sys

D = Path(__file__).resolve().parent
H = D.parent
LIGHT = H.parents[4] / '.venv/bin/python'
NATIVE = H.parent / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')

def admit():
    cp = pin(D/'boundary_config_v2.json'); c = json.loads(Path(cp['path']).read_text())
    np = pin(D/'native_screen_continuation_v2.json'); nc = json.loads(Path(np['path']).read_text())
    rp = pin(nc['stages']['2']['result']); r = json.loads(Path(rp['path']).read_text())
    gp = pin(nc['stages']['2']['assessment_guard']); g = json.loads(Path(gp['path']).read_text())
    assert r['config'] == np and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate'] == c['candidate']
    assert g['command'] == [str(NATIVE), '-B', nc['engine']['path'], np['path'], np['sha256'], '2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code'] == 0
    assert g['error'] is None and g['max_swap_growth_bytes'] == 0
    once(D/'boundary_admission_v2.json', dict(root_admitted=True, source_changes=False, serving_qualified=False,
        config=cp, assessor=c['engine'], candidate=c['candidate'], native_config=np, native2=rp,
        native2_guard=gp, source_proof=c['build'], additional6_requires_boundary_pass=True,
        original_pilot_preserved=True, no_model_rerun_for_reassessment=True,
        official_residual_limit_s=.4, strict_residual_upper_bound=True, prospective_s34_reference=True))
    print(json.dumps(pin(D/'boundary_admission_v2.json')))

def main():
    if sys.argv[1:] == ['--admit-boundary']:
        admit(); return
    assert not sys.argv[1:]
    old_path = D/'boundary_assess_s34.py'; old = old_path.read_text()
    assert pin(old_path)['sha256'] == '7130ffcbefa5ab1382b1427024bdc28625c9684c9c019efbd54c80b2d0f1b2d3'
    before = """    primitive_guard = q.guard(c['primitive']['guard'], [str(q.NATIVE), '-B', build['primitive']['path'],
        c['build']['sha256'], c['primitive']['result']['path']], 60)
    assert primitive_guard['max_rss_bytes'] < 1000000000
"""
    after = """    # Primitive actually ran under its exact 1 GB guard, not the model's 7.5 GB guard.
    primitive_guard = json.loads(q.checked(c['primitive']['guard']))
    assert primitive_guard['command'] == [str(q.NATIVE), '-B', build['primitive']['path'],
        c['build']['sha256'], c['primitive']['result']['path']]
    assert primitive_guard['completed'] and primitive_guard['exit_code'] == 0 and primitive_guard['error'] is None
    assert primitive_guard['whole_group_retired'] and not primitive_guard['remaining_rows']
    assert primitive_guard['rss_trigger_bytes'] == 1000000000 and 0 < primitive_guard['max_rss_bytes'] < 1000000000
    assert primitive_guard['max_swap_growth_bytes'] == 0 and 0 < primitive_guard['elapsed_s'] <= 60
"""
    assert old.count(before) == 1
    new = old.replace(before, after, 1)
    for stem in ('boundary_admission', 'boundary_assessment', 'boundary_assess_guard'):
        new = new.replace(stem+'.json', stem+'_v2.json')
    ast.parse(new)
    assessor = D/'boundary_assess_s34_v2.py'
    with assessor.open('x') as stream: stream.write(new)
    diff = D/'boundary_adapter_v2.diff'
    with diff.open('x') as stream:
        stream.write(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
            fromfile=str(old_path), tofile=str(assessor))))
    old_c = pin(D/'boundary_config.json'); c = json.loads(Path(old_c['path']).read_text())
    old_n = pin(D/'native_screen_continuation.json'); nc = json.loads(Path(old_n['path']).read_text())
    original = copy.deepcopy(nc)
    preserved = [pin(D/name) for name in ('boundary_assessment.json', 'boundary_assess_guard.json',
        'boundary_admission.json', 'native2_continuation_assessment.json', 'native2_continuation_assess_guard.json')]
    c.update(engine=pin(assessor), native_config_path=str(D/'native_screen_continuation_v2.json'),
        technical_repair=dict(reason='Primitive guard was 1 GB; shared model guard incorrectly required 7.5 GB.',
            original_assessor=pin(old_path), original_config=old_c, preserved_artifacts=preserved,
            primitive_rss_limit_bytes=1000000000, no_numerical_reruns=True))
    bp = D/'boundary_config_v2.json'; once(bp, c); bp = pin(bp)
    nc['stages']['2']['result'] = str(D/'native2_continuation_v2_assessment.json')
    nc['stages']['2']['assessment_guard'] = str(D/'native2_continuation_v2_assess_guard.json')
    req = nc['boundary_before8']
    req.update(receipt=str(D/'boundary_assessment_v2.json'), assessor=pin(assessor),
        assessment_guard=str(D/'boundary_assess_guard_v2.json'),
        assessment_command=[str(LIGHT), '-B', str(assessor), bp['path'], bp['sha256']])
    nc['provenance'] += [pin(__file__), pin(assessor), pin(diff), bp, old_c, old_n, *preserved]
    assert all(nc[k] == original[k] for k in ('engine', 'arms', 'parity_layers', 'future_only', 'raw_error_veto', 'continuation'))
    assert all(nc['stages'][s][k] == original['stages'][s][k] for s in ('2', '8') for k in ('gates', 'assessment_limits'))
    np = D/'native_screen_continuation_v2.json'; once(np, nc); np = pin(np)
    commands = json.loads((D/'continuation_commands.json').read_text())
    commands['commands'].pop('boundary')
    commands['commands'].update(native2_reassessment=[str(NATIVE), '-B', nc['engine']['path'], np['path'], np['sha256'], '2'],
        admission=[str(LIGHT), '-B', str(Path(__file__).resolve()), '--admit-boundary'],
        boundary_assessment=req['assessment_command'],
        native8_assessment=[str(NATIVE), '-B', nc['engine']['path'], np['path'], np['sha256'], '8'])
    commands.update(config=np, boundary_config=bp, assessor=pin(assessor),
        native2_guard=nc['stages']['2']['assessment_guard'], boundary_assessment_guard=req['assessment_guard'])
    once(D/'continuation_commands_v2.json', commands)
    print(json.dumps(dict(config=np, boundary=bp, assessor=pin(assessor), commands=pin(D/'continuation_commands_v2.json'))))

if __name__ == '__main__': main()
