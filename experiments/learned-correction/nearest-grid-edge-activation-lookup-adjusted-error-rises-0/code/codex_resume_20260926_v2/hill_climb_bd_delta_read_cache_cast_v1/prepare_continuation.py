"""Narrow cache+cast continuation; saved N2 and own boundary, no model execution."""
from pathlib import Path
import copy, difflib, hashlib, json, sys
D = Path(__file__).resolve().parent
H = D.parent
T = H/'hill_climb_bd_delta_old_polar_v1'
CACHE = H/'hill_climb_bd_delta_uvp_read_cache_v1'
BASE = H/'hill_climb_combo_bd_linear_delta_v1'
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
    assert r['candidate'] == c['candidate'] and r['all16_parent_bytes_exact']
    assert g['command'] == [str(NATIVE), '-B', nc['engine']['path'], np['path'], np['sha256'], '2']
    assert g['completed'] and g['whole_group_retired'] and g['exit_code'] == 0 and g['error'] is None
    assert g['max_swap_growth_bytes'] == 0 and not g['remaining_rows']
    once(D/'boundary_admission.json', dict(root_admitted=True, source_changes=False, serving_qualified=False,
        config=cp, assessor=c['engine'], candidate=c['candidate'], native_config=np, native2=rp,
        native2_guard=gp, source_proof=c['build'], additional6_requires_boundary_pass=True,
        original_pilot_preserved=True, no_model_rerun_for_reassessment=True,
        official_residual_limit_s=.4, strict_residual_upper_bound=True, prospective_s34_reference=True))
    print(json.dumps(pin(D/'boundary_admission.json')))

PROOF = """    # Exact two-stage inverse: public cast elision, then owned transpose cache.
    assert a['source_proof'] == c['build']
    assert build['comparator'] == c['native_comparator'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']
    assert build['comparator']['sha256'] == '6f104e9db5ca32dc5e7724d64e03fd969bfff9d6cb48ce2b55695a998eecd1bb'
    source = q.checked(c['candidate']).decode()
    inv = json.loads(q.checked(build['source_inverse']))
    assert inv['parent'] == build['parent'] and inv['exact_inverse']
    for edit in reversed(inv['edits']):
        assert source.count(edit['after']) == 1
        source = source.replace(edit['after'], edit['before'], 1)
    assert source == q.checked(build['parent']).decode()
    cb = json.loads(q.checked(c['cache_build']))
    ci = json.loads(q.checked(cb['source_inverse']))
    assert cb['candidate'] == build['parent'] and cb['parent'] == build['comparator']
    assert ci['parent'] == cb['parent'] and ci['exact_inverse'] and ci['fragment'] == cb['fragment']
    suffix = chr(10) + q.checked(ci['fragment']).decode()
    assert source.endswith(suffix); source = source[:-len(suffix)]
    for edit in reversed(ci['edits']):
        assert source.count(edit['after']) == 1
        source = source.replace(edit['after'], edit['before'], 1)
    assert source == q.checked(build['comparator']).decode()
    ei = json.loads(q.checked(cb['engine_inverse']))
    assert c['control_native_engine'] == ei['parent']
    assert c['control_native_engine']['sha256'] == '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
    assert ei['engine'] == c['native_engine'] == build['engine'] == cb['engine'] and ei['exact_inverse']
    assert ei['engine']['sha256'] == '116793c99086668ca2c58f6288fc2f0b05d3970cc221191aae550577ceb39d5b'
    assert ei['before'] == "            assert 0 <= ubc['max_resident_bytes'] <= 512000000"
    assert ei['after'] == "            assert 0 <= ubc['max_resident_bytes'] <= (640000000 if role == 'candidate' else 512000000)"
    custom = q.checked(ei['engine']).decode()
    assert custom.count(ei['after']) == 1 and custom.replace(ei['after'], ei['before'], 1) == q.checked(ei['parent']).decode()
    assert c['candidate_cache_limit_bytes'] == 640000000 and c['control_cache_limit_bytes'] == 512000000
    retained = json.loads(q.checked(c['score_parent_retained']))
    assert retained['status'] == 'RETAINED_COMPONENT' and retained['candidate'] == build['comparator']
    parent_n8 = json.loads(q.checked(retained['native8']))
    parent_boundary = json.loads(q.checked(retained['boundary']))
    parent_raw = json.loads(q.checked(retained['native8_raw_review']))
    assert parent_n8['candidate'] == build['comparator'] and parent_n8['admitted'] and parent_n8['passed_integrity'] and all(parent_n8['gates'].values())
    assert parent_boundary['candidate'] == build['comparator'] and parent_boundary['passed'] and parent_boundary['receipt_checks_passed']
    assert parent_raw['passed'] and parent_raw['native8'] == retained['native8']
    assert r['all16_parent_bytes_exact'] is True
"""

def main():
    if sys.argv[1:] == ['--admit-boundary']:
        admit(); return
    assert not sys.argv[1:]
    old_assessor = T/'boundary_assess_s34.py'; old_config = T/'boundary_config.json'
    assert pin(old_assessor)['sha256'] == 'e27a76001f3d0bb41fbbe9d6a61ca9d038631de521035e06011e30ba4dab99b6'
    assert pin(old_config)['sha256'] == '4327fabb2a9eb778b32d2186e6c8d6f816e508e8fd4ac575c215ff2303cd088a'
    pilot = pin(D/'native_screen.json')
    assert pilot['sha256'] == '123d74598d1a89e859465aef5e2be49c7e05cd1af488fe070f5c476fed211f7c'
    nc = json.loads(Path(pilot['path']).read_text()); original = copy.deepcopy(nc)
    r = json.loads((D/'native2_assessment.json').read_text())
    assert r['config'] == pilot and r['admitted'] and r['passed_integrity'] and all(r['gates'].values())
    assert r['candidate'] == pin(D/'candidate.py') and r['parity_layers'] == list(range(16))
    assert nc['parity_layers'] == list(range(16)) and r['configured_parent_bytes_exact'] and r['all16_parent_bytes_exact']
    for size, wins in (('2', 2), ('8', 8)):
        gates = nc['stages'][size]['gates']
        assert gates['adjusted_pct_at_most'] == -.02 and gates['wins_min'] == wins
        assert gates['max_per_network_regression_pct'] == 0 and gates['strict_adjusted'] is False
    build = json.loads((D/'build.json').read_text())
    assert build['candidate'] == pin(D/'candidate.py') and build['source_inverse_exact']
    assert build['comparator'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']
    old = old_assessor.read_text(); new = old
    def replace_one(before, after):
        nonlocal new
        assert new.count(before) == 1, before
        new = new.replace(before, after, 1)
    replace_one('Prospective old-carried polarization boundary2', 'Prospective full-base cache plus public cast-elision boundary2')
    replace_one("'old-polar-s34-boundary-native2-v1'", "'read-cache-cast-s34-boundary-native2-v1'")
    replace_one('def native_arm(q, arm, size, records):', 'def native_arm(q, arm, size, records, cache_limit=512000000):')
    replace_one("assert 0 <= row['ubc']['max_resident_bytes'] <= 512000000", "assert 0 <= row['ubc']['max_resident_bytes'] <= cache_limit")
    replace_one("native_arm(q, cfg['arms'][role], 2, receipt['blocks'][0]['arms'][role])", "native_arm(q, cfg['arms'][role], 2, receipt['blocks'][0]['arms'][role], 640000000 if role == 'candidate' else 512000000)")
    start = new.index('    # Old-polar source inverse and narrowly changed stage2 config assertion.')
    end = new.index("    assert nc['parity_layers']", start)
    new = new[:start] + PROOF + new[end:]
    replace_one("assert nc['parity_layers'] == r['parity_layers'] == list(range(10))", "assert nc['parity_layers'] == r['parity_layers'] == list(range(16))")
    for name in ('paired_boundary_assessment.json', 'paired_boundary_assess_guard.json'):
        assert new.count(repr(name)) == old.count(repr(name)) == 1
    assessor = D/'boundary_assess_s34.py'; compile(new, str(assessor), 'exec')
    with assessor.open('x') as f: f.write(new)
    diff = D/'boundary_adapter.diff'
    with diff.open('x') as f: f.write(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(old_assessor), tofile=str(assessor))))
    c = json.loads(old_config.read_text())
    c.update(schema='read-cache-cast-s34-boundary-native2-v1', engine=pin(assessor), candidate=pin(D/'candidate.py'),
        build=pin(D/'build.json'), source_protocol=pin(D/'protocol.json'), native_comparator=build['comparator'],
        native_config_path=str(D/'native_screen_continuation.json'), tag=D.name+'_boundary_feasibility_v1',
        setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']), native2_gates=copy.deepcopy(nc['stages']['2']['gates']),
        score_parent_retained=pin(BASE/'retained_component.json'), native_engine=nc['engine'],
        control_native_engine=pin(H/'native_screen_engine_v1/screen.py'), cache_build=pin(CACHE/'build.json'),
        candidate_cache_limit_bytes=640000000, control_cache_limit_bytes=512000000)
    assert c['setup_counts'] == dict(bound=687, lvfuse=649, psdb_fused=9, pair_block_indices=1, read7_fused=4)
    assert c['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    c['preserved_failures'] += [pin(CACHE/'native2_assessment.json')]
    c['interpretation'] = 'Cache+cast own boundary; score comparator retained BD-delta, S34 residual and S32 runtime controls unchanged. Candidate-only cache640M, controls512M. Exact all16 and all-wins gates unchanged.'
    bp = D/'boundary_config.json'; once(bp, c); bp = pin(bp)
    nc['stages']['2']['result'] = str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard'] = str(D/'native2_continuation_assess_guard.json')
    req = nc['boundary_before8']; req.update(receipt=str(D/'boundary_assessment.json'), assessor=pin(assessor),
        assessment_guard=str(D/'boundary_assess_guard.json'), assessment_command=[str(LIGHT), '-B', str(assessor), bp['path'], bp['sha256']])
    nc['continuation'] = dict(pilot_config=pilot, pilot_result=pin(D/'native2_assessment.json'), pilot_assessment_guard=pin(D/'native2_assess_guard.json'),
        no_model_rerun=True, same_saved_native2_outputs=True, no_score_gate_changes=True)
    nc['provenance'] += [pin(__file__), pin(assessor), pin(diff), bp, pin(old_assessor), pin(old_config),
        *[c[k] for k in ('score_parent_retained', 's34_native_config', 'parent_boundary', 's34_bank', 's34_evaluation', 'control_native_engine', 'cache_build')],
        *[nc['continuation'][k] for k in ('pilot_config', 'pilot_result', 'pilot_assessment_guard')]]
    nc['interpretation'] = 'Cache+cast saved N2 reassessment, own boundary, fixed six and strict exact N8 gates; no broad/bank qualification.'
    nc['prospective_scope'] = 'saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert all(nc[k] == original[k] for k in ('engine', 'arms', 'parity_layers', 'future_only', 'raw_error_veto'))
    assert all(nc['stages'][s][k] == original['stages'][s][k] for s in ('2', '8') for k in ('gates', 'assessment_limits'))
    np = D/'native_screen_continuation.json'; once(np, nc); np = pin(np)
    b6 = nc['arms']['candidate']['blocks']['6']
    commands = dict(native2_reassessment=[str(NATIVE), '-B', nc['engine']['path'], np['path'], np['sha256'], '2'],
        admission=[str(LIGHT), '-B', str(Path(__file__).resolve()), '--admit-boundary'],
        boundary_census=[str(LIGHT), '-B', c['driver']['path'], c['candidate']['path'], '35,100', c['tag']],
        boundary_assessment=req['assessment_command'],
        native6=[str(NATIVE), '-B', b6['driver']['path'], c['candidate']['path'], ','.join(map(str, b6['ranks'])), b6['tag']],
        native8_assessment=[str(NATIVE), '-B', nc['engine']['path'], np['path'], np['sha256'], '8'])
    once(D/'continuation_commands.json', dict(commands=commands, config=np, boundary_config=bp, assessor=pin(assessor),
        native2_guard=nc['stages']['2']['assessment_guard'], boundary_guard=str(D/'boundary_guard.json'),
        boundary_log=str(D/'boundary.log'), boundary_assessment_guard=req['assessment_guard'], native6_guard=b6['guard'],
        native8_guard=nc['stages']['8']['assessment_guard']))
    print(json.dumps(dict(config=np, boundary=bp, assessor=pin(assessor), commands=pin(D/'continuation_commands.json'))))

if __name__ == '__main__': main()
