"""Prepare one prospective continuation; --enable is for root after review."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json
import sys

D = Path(__file__).resolve().parent
H = D.parent
P = H/'hill_climb_e1_ynt_psdb_d5_v1'
V3 = H/'boundary_feasibility_engine_v3/assess.py'


def pin(p):
    p = Path(p).resolve(); raw = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def once(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


def source():
    assert pin(V3)['sha256'] == '0c4076d5bb693d72a2f9020a73a496330dff68989e918e92c59f22fb4dac31b2'
    old = V3.read_text(); new = old
    changes = [
        ('setup_s_max=4.1, residual_s_max=.35, projected_wall_s_max=105,', 'setup_s_max=4.1, residual_s_max=.4, projected_wall_s_max=105,'),
        ("c['schema'] == 'early-boundary-native2-v3'", "c['schema'] == 'early-boundary-native2-official-limit-v1'"),
        ("'source_protocol', 'native_engine', 'probe', 'driver', 'harness'", "'source_protocol', 'native_engine', 'driver', 'harness'"),
        ("('native_config', 'native2', 'native2_guard', 'primitive', 'primitive_guard')", "('native_config', 'native2', 'native2_guard', 'source_proof')"),
        ('residual_projection=residual <= .35,', 'residual_projection=residual < .4,'),
        ("native_config=a['native_config'], primitive=a['primitive'],", "native_config=a['native_config'], source_proof=a['source_proof'],"),
        ('projected_s=residual, gate_s=.35)', 'projected_s=residual, gate_s=.4, strict_upper_bound=True)'),
    ]
    for before, after in changes:
        assert new.count(before) == 1, before
        new = new.replace(before, after, 1)
    start = new.index("    assert a['primitive'] == pin(d/'primitive_result.json')")
    end = new.index("    for failure in c['preserved_failures']:", start)
    replacement = '''    # A two-literal policy change: prove its source inverse and actual N2 parity.
    assert a['source_proof'] == c['build']
    parent = q.checked(build['parent']).decode()
    old_fragment = q.checked(build['original_fragment']).decode()
    fragment = q.checked(build['fragment']).decode()
    assert build['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    assert build['original_fragment']['sha256'] == '85e75784afb7f3d79854cd5599f4e7f77a15bdef117f6edccb7f23fa85ec70c0'
    edits = (("negative = (ctx['a'] < -2.5) & (~ctx['live'])", "negative = (ctx['a'] < -2.75) & (~ctx['live'])"),
             ("score = fnp.where(negative, ctx['a'] + 2.5, 0.0)", "score = fnp.where(negative, ctx['a'] + 2.75, 0.0)"))
    expected_fragment = old_fragment
    for before, after in edits:
        assert expected_fragment.count(before) == 1 and after not in expected_fragment
        expected_fragment = expected_fragment.replace(before, after, 1)
    assert fragment == expected_fragment and parent.count(old_fragment) == 1
    candidate_source = q.checked(c['candidate']).decode()
    assert candidate_source == parent.replace(old_fragment, fragment, 1)
    assert candidate_source.replace(fragment, old_fragment, 1) == parent
    assert nc['parity_layers'] == list(range(7)) and r['parity_layers'] == list(range(7))
    assert r['configured_parent_bytes_exact'] is True
    # The continuation only re-assesses the same immutable pilot outputs.
    continuation = nc['continuation']
    original_config = json.loads(q.checked(continuation['pilot_config']))
    original_result = json.loads(q.checked(continuation['pilot_result']))
    assert original_config['prospective_scope'] == 'native2_only'
    for key in ('schema', 'engine', 'arms', 'parity_layers', 'future_only', 'raw_error_veto'):
        assert nc[key] == original_config[key]
    for stage_id in ('2', '8'):
        for key in ('gates', 'assessment_limits'):
            assert nc['stages'][stage_id][key] == original_config['stages'][stage_id][key]
    assert original_result['config'] == continuation['pilot_config']
    assert continuation['pilot_result'] == pin(original_config['stages']['2']['result'])
    assert continuation['pilot_assessment_guard'] == pin(original_config['stages']['2']['assessment_guard'])
    q.guard(continuation['pilot_assessment_guard'], [str(q.NATIVE), '-B', c['native_engine']['path'],
        continuation['pilot_config']['path'], continuation['pilot_config']['sha256'], '2'], 120)
    excluded = {'config', 'evidence'}
    assert {k: v for k, v in r.items() if k not in excluded} == {k: v for k, v in original_result.items() if k not in excluded}
'''
    new = new[:start] + replacement + new[end:]
    new = new.replace('"""Future-only early boundary2 assessment from pinned JSON/source; no array reads.',
        '"""Prospective YNT2.75 boundary2: v3 with strict official residual cap and source inverse.')
    ast.parse(new)
    return old, new


def main():
    enable = sys.argv[1:] == ['--enable']
    assert sys.argv[1:] in ([], ['--enable'])
    old, new = source()
    assessor = D/'boundary_assess_official_limit.py'
    if enable:
        assert assessor.read_text() == new
    else:
        with assessor.open('x') as f: f.write(new)
        with (D/'boundary_adapter.diff').open('x') as f:
            f.write(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(V3), tofile=str(assessor))))
    pilot_pin = pin(D/'native_screen.json')
    assert pilot_pin['sha256'] == 'ded77255ca6d6a62c4ed4635f8c1ea8653626e1ba6f4c85d97f7df8a9f7d3207'
    nc = json.loads((D/'native_screen.json').read_text())
    original = copy.deepcopy(nc)
    receipt = json.loads((D/'native2_assessment.json').read_text())
    assert receipt['config'] == pilot_pin and receipt['passed_integrity'] and receipt['admitted']
    assert receipt['candidate'] == pin(D/'candidate.py') and receipt['parity_layers'] == list(range(7))
    c = json.loads((P/'boundary_config.json').read_text())
    c['schema'] = 'early-boundary-native2-official-limit-v1'
    c['prospective_authorization'] = enable
    c['engine'] = pin(assessor)
    c['candidate'] = pin(D/'candidate.py')
    c['build'] = pin(D/'build.json')
    c['source_protocol'] = pin(D/'protocol.json')
    c['native_config_path'] = str(D/'native_screen_continuation.json')
    c['native_comparator'] = nc['arms']['base']['source']
    c['tag'] = D.name+'_boundary_feasibility_v1'
    c['limits']['residual_s_max'] = .4
    c['native2_gates'] = copy.deepcopy(nc['stages']['2']['gates'])
    for key in ('probe', 'primitive_contract', 'root_pre_boundary_metadata_requirement', 'additional6_gate'):
        c.pop(key, None)
    c['interpretation'] = 'Prospective own-candidate boundary after saved-N2 reassessment, strict projected residual <0.4; unchanged wall105/setup4.1/7.5GB/zeroSwap and exact S32 runtime anchor. D5 is the score comparator. Exact two-literal source inverse and passed L0-6 parity replace an unnecessary new numerical primitive. Original pilot evidence remains immutable. No boundary or serving pass is inherited.'
    c['official_residual_limit_source'] = 'https://github.com/AIcrowd/whest-starterkit/blob/main/docs/reference/rounds.md'
    boundary_path = D/('boundary_config.json' if enable else 'boundary_config_draft.json')
    once(boundary_path, c)
    boundary_pin = pin(boundary_path)
    nc['prospective_authorization'] = enable
    nc['stages']['2']['result'] = str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard'] = str(D/'native2_continuation_assess_guard.json')
    req = nc['boundary_before8']
    req['receipt'] = str(D/'boundary_assessment_extension_v2.json')
    req['assessor'] = pin(assessor)
    req['assessment_guard'] = str(D/'boundary_assess_guard_extension_v2.json')
    req['assessment_command'] = [str(H.parents[4]/'.venv/bin/python'), '-B', str(assessor), str(D/'boundary_config.json'), boundary_pin['sha256']]
    assert Path(req['assessment_command'][0]).is_file()
    nc['continuation'] = dict(pilot_config=pilot_pin, pilot_result=pin(D/'native2_assessment.json'),
        pilot_assessment_guard=pin(D/'native2_assess_guard.json'), no_model_rerun=True,
        same_saved_native2_outputs=True, no_score_gate_changes=True)
    nc['provenance'] += [pin(__file__), pin(assessor), pin(D/'boundary_adapter.diff'),
        boundary_pin, *[nc['continuation'][k] for k in ('pilot_config', 'pilot_result', 'pilot_assessment_guard')]]
    nc['interpretation'] = 'Prospective continuation of the passed YNT2.75 pilot. Re-assess saved N2 outputs under this boundary-enabled config without rerunning the estimator; original result/config/guard remain pinned and unchanged. Own boundary must pass before fixed six and original native8 assessment. Native8 gates unchanged: >=0.1% gain, >=6 wins, CI upper<0 and additional6 negative. No broad or bank authority.'
    nc['prospective_scope'] = 'saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert nc['arms'] == original['arms'] and nc['parity_layers'] == original['parity_layers']
    assert all(nc['stages'][k]['gates'] == original['stages'][k]['gates'] for k in ('2', '8'))
    output = D/('native_screen_continuation.json' if enable else 'native_screen_continuation_draft.json')
    once(output, nc)
    print(json.dumps(dict(enabled=enable, engine=pin(assessor), config=pin(output), boundary=boundary_pin,
        builder=pin(__file__), original_config=pilot_pin)))


if __name__ == '__main__': main()
