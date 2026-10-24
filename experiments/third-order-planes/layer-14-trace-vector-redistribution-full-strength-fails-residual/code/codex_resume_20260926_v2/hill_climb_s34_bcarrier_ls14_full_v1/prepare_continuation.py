"""Reuse the reviewed READ7/S34 boundary recipe; source and saved receipts only."""
from pathlib import Path
import ast, copy, difflib, hashlib, json, sys

D = Path(__file__).resolve().parent
H = D.parent
T = H / 'hill_climb_s34_read7_v1'
LIGHT = H.parents[4] / '.venv/bin/python'
NATIVE = H.parent / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'

def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))

def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')

def admit():
    cp = pin(D / 'boundary_config.json'); c = json.loads(Path(cp['path']).read_text())
    np = pin(D / 'native_screen_continuation.json'); nc = json.loads(Path(np['path']).read_text())
    rp = pin(nc['stages']['2']['result']); r = json.loads(Path(rp['path']).read_text())
    gp = pin(nc['stages']['2']['assessment_guard']); g = json.loads(Path(gp['path']).read_text())
    assert r['config'] == np and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate'] == c['candidate']
    assert g['command'] == [str(NATIVE), '-B', nc['engine']['path'], np['path'], np['sha256'], '2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code'] == 0
    assert g['error'] is None and g['max_swap_growth_bytes'] == 0
    once(D / 'boundary_admission.json', dict(root_admitted=True, source_changes=False, serving_qualified=False,
        config=cp, assessor=c['engine'], candidate=c['candidate'], native_config=np, native2=rp,
        native2_guard=gp, source_proof=c['build'], additional6_requires_boundary_pass=True,
        original_pilot_preserved=True, no_model_rerun_for_reassessment=True,
        official_residual_limit_s=.4, strict_residual_upper_bound=True, prospective_s34_reference=True))
    print(json.dumps(pin(D / 'boundary_admission.json')))

def main():
    if sys.argv[1:] == ['--admit-boundary']:
        admit(); return
    assert not sys.argv[1:]
    original_assessor = T / 'boundary_assess_s34_v3.py'
    original_config = T / 'boundary_config_v3.json'
    assert pin(original_assessor)['sha256'] == 'd01488921900d486301a83526ef174eda9a03c3578a43bf7a753a79d6d3ccaf2'
    assert pin(original_config)['sha256'] == 'b70f857100b0f498cf8831921fc9fa8878274a9b2249347a1877a1c8783bbe40'
    pilot = pin(D / 'native_screen.json')
    assert pilot['sha256'] == 'f582275ceacac33f94e48e56a2e15b8e67264a160d612e6af8f2d0df79f3efad'
    nc = json.loads(Path(pilot['path']).read_text()); original = copy.deepcopy(nc)
    old_result = json.loads((D / 'native2_assessment.json').read_text())
    assert old_result['config'] == pilot and old_result['admitted'] and old_result['passed_integrity']
    assert old_result['candidate'] == pin(D / 'candidate.py') and old_result['parity_layers'] == list(range(15))
    build = json.loads((D / 'build.json').read_text())
    primitive = dict(result=pin(D / 'primitive_result.json'), guard=pin(D / 'primitive_guard.json'))
    pr = json.loads(Path(primitive['result']['path']).read_text())
    assert pr['passed'] and pr['candidate'] == build['candidate'] and pr['build'] == pin(D / 'build.json')
    old = original_assessor.read_text(); new = old.replace('READ7', 'BCLS14 full-step')
    new = new.replace('read7-s34-boundary-native2-v1', 'bcls14-full-s34-boundary-native2-v1')
    new = new.replace("build['exact_parent_prefix'] is True", "build['source_inverse_exact'] is True")
    start = new.index('    # Appended BCLS14 full-step implementation:')
    end = new.index('    # The continuation only re-assesses', start)
    proof = '''    # Exact source inverse and actual paid primitive; no inherited candidate pass.
    assert a['source_proof'] == c['build']
    parent = q.checked(build['parent']).decode()
    fragment = q.checked(build['fragment']).decode()
    assert build['parent'] == c['parent'] == nc['arms']['base']['source']
    assert build['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    source = q.checked(c['candidate']).decode()
    assert source.endswith('\\n' + fragment)
    inverse = source[:-len('\\n' + fragment)]
    for edit in reversed(build['edits']):
        assert inverse.count(edit['after']) == 1
        inverse = inverse.replace(edit['after'], edit['before'], 1)
    assert inverse == parent
    assert nc['parity_layers'] == r['parity_layers'] == list(range(15))
    assert r['configured_parent_bytes_exact'] is True
    primitive = json.loads(q.checked(c['primitive']['result']))
    assert primitive['passed'] is True and primitive['candidate'] == c['candidate']
    assert primitive['build'] == c['build'] and primitive['probe'] == build['primitive']
    assert primitive['all_inputs_unchanged'] is True and primitive['model_executed'] is False
    assert len(primitive['cases']) == 4 and all(x['passed'] and x['flops'] > 0 for x in primitive['cases'])
    q.checked(build['primitive'])
    primitive_guard = q.guard(c['primitive']['guard'], [str(q.NATIVE), '-B', build['primitive']['path'],
        c['build']['sha256'], c['primitive']['result']['path']], 60)
    assert primitive_guard['max_rss_bytes'] < 1000000000
'''
    new = new[:start] + proof + new[end:]
    start = new.index("    recovery = c['setup_recovery']")
    end = new.index("    assert nc['arms']['candidate']['setup']['counts']", start)
    new = new[:start] + "    setup = m.setup_record(q, d/'boundary.log', c['setup_counts']['bound'], c['setup_counts']['lvfuse'])\n" + new[end:]
    new = new.replace("setup=setup, setup_recovery=dict(contract=recovery, launcher=setup_launch_pin, guard=setup_gp, original_setup_stdout_missing=True, prediction_rerun=False),", "setup=setup,")
    new = new.replace(", setup_guard['max_rss_bytes']", "")
    new = new.replace('boundary_admission_read7_v3.json', 'boundary_admission.json')
    new = new.replace('boundary_assessment_read7_v3.json', 'boundary_assessment.json')
    new = new.replace('boundary_assess_guard_read7_v3.json', 'boundary_assess_guard.json')
    ast.parse(new)
    assessor = D / 'boundary_assess_s34.py'
    with assessor.open('x') as stream: stream.write(new)
    diff = D / 'boundary_adapter.diff'
    with diff.open('x') as stream:
        stream.write(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
            fromfile=str(original_assessor), tofile=str(assessor))))
    c = json.loads(original_config.read_text()); c.pop('setup_recovery')
    c.update(schema='bcls14-full-s34-boundary-native2-v1', engine=pin(assessor),
        candidate=pin(D / 'candidate.py'), build=pin(D / 'build.json'), source_protocol=pin(D / 'protocol.json'),
        native_config_path=str(D / 'native_screen_continuation.json'), tag=D.name+'_boundary_feasibility_v1',
        setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']), primitive=primitive,
        native2_gates=copy.deepcopy(nc['stages']['2']['gates']))
    assert c['setup_counts'] == dict(bound=682, lvfuse=649, psdb_fused=9)
    c['preserved_failures'].append(pin(H / 'hill_climb_s34_bcarrier_ls14_v1/native2_assessment.json'))
    c['interpretation'] = ('Own BCLS14 full boundary; same S34 residual anchor and four-times delta, strict0.4; '
        'unchanged S32 runtime controls,105s wall,4.1s setup,7.5GB/zeroSwap. Original complete boundary.log; no setup recovery/retry.')
    boundary = D / 'boundary_config.json'; once(boundary, c); bp = pin(boundary)
    nc['stages']['2']['result'] = str(D / 'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard'] = str(D / 'native2_continuation_assess_guard.json')
    req = nc['boundary_before8']
    req.update(receipt=str(D / 'boundary_assessment.json'), assessor=pin(assessor),
        assessment_guard=str(D / 'boundary_assess_guard.json'),
        assessment_command=[str(LIGHT), '-B', str(assessor), str(boundary), bp['sha256']])
    nc['continuation'] = dict(pilot_config=pilot, pilot_result=pin(D / 'native2_assessment.json'),
        pilot_assessment_guard=pin(D / 'native2_assess_guard.json'), no_model_rerun=True,
        same_saved_native2_outputs=True, no_score_gate_changes=True)
    nc['provenance'] += [pin(__file__), pin(assessor), pin(diff), bp, pin(original_assessor), pin(original_config),
        *primitive.values(), *[c[k] for k in ('parent_boundary', 's34_bank', 's34_evaluation')],
        *[nc['continuation'][k] for k in ('pilot_config', 'pilot_result', 'pilot_assessment_guard')]]
    nc['interpretation'] = ('BCLS14 full continuation: same saved N2 outputs, own boundary, then fixed six and original N8 gates. '
        'Paid primitive/source inverse bound; original pilot preserved. No broad/bank authority.')
    nc['prospective_scope'] = 'saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert all(nc[k] == original[k] for k in ('engine', 'arms', 'parity_layers', 'future_only', 'raw_error_veto'))
    assert all(nc['stages'][s][k] == original['stages'][s][k] for s in ('2', '8') for k in ('gates', 'assessment_limits'))
    output = D / 'native_screen_continuation.json'; once(output, nc); np = pin(output)
    b6 = nc['arms']['candidate']['blocks']['6']
    commands = dict(native2_reassessment=[str(NATIVE), '-B', nc['engine']['path'], str(output), np['sha256'], '2'],
        admission=[str(LIGHT), '-B', str(Path(__file__).resolve()), '--admit-boundary'],
        boundary=[str(LIGHT), '-B', c['driver']['path'], c['candidate']['path'], '35,100', c['tag']],
        boundary_assessment=req['assessment_command'],
        native6=[str(NATIVE), '-B', b6['driver']['path'], c['candidate']['path'], ','.join(map(str, b6['ranks'])), b6['tag']],
        native8_assessment=[str(NATIVE), '-B', nc['engine']['path'], str(output), np['sha256'], '8'])
    once(D / 'continuation_commands.json', dict(commands=commands, config=np, boundary_config=bp, assessor=pin(assessor),
        native2_guard=nc['stages']['2']['assessment_guard'], boundary_guard=str(D / 'boundary_guard.json'),
        boundary_log=str(D / 'boundary.log'), boundary_assessment_guard=req['assessment_guard'],
        native6_guard=b6['guard'], native8_guard=nc['stages']['8']['assessment_guard']))
    assert pin(D / 'native_screen.json') == pilot
    print(json.dumps(dict(config=np, boundary=bp, assessor=pin(assessor), commands=pin(D / 'continuation_commands.json'))))

if __name__ == '__main__': main()
