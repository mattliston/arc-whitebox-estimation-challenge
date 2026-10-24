"""Narrow forward-cache plus unique-gather continuation; saved N2 and own boundary, no model execution."""
from pathlib import Path
import copy, difflib, hashlib, json, sys
D = Path(__file__).resolve().parent
H = D.parent
T = H/'hill_climb_exact_scatter_rows_v1'
COMPOSITION = H/'hill_climb_unique_row_gather_v1'
CACHE = H/'hill_climb_bd_delta_uvp_read_cache_v1'
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

PROOF = '    # Exact forward-cache restore then gather inverse; SC score control, S34 official reference.\n    assert a[\'source_proof\'] == c[\'build\']\n    assert c[\'native_comparator\'] == nc[\'arms\'][\'base\'][\'source\'] == nc[\'arms\'][\'parity_parent\'][\'source\']\n    assert c[\'native_comparator\'][\'sha256\'] == \'0f7b93c4c45c447d76f04f428c1aeec066c5eb30027671653565b7c5fe3d93c3\'\n    assert build[\'score_comparator\'] == c[\'native_comparator\']\n    assert build[\'parent\'][\'sha256\'] == \'ead972324718e466d6f6b847d85ca4995d11acff423035bc5ba28458214e8380\'\n    source = q.checked(c[\'candidate\']).decode()\n    inv = json.loads(q.checked(build[\'source_inverse\']))\n    assert inv[\'parent\'] == build[\'parent\'] and inv[\'exact_inverse\']\n    for edit in reversed(inv[\'edits\']):\n        assert source.count(edit[\'after\']) == 1\n        source = source.replace(edit[\'after\'], edit[\'before\'], 1)\n    assert source == q.checked(build[\'parent\']).decode()\n    composed = json.loads(q.checked(c[\'composition_parent_build\']))\n    assert c[\'composition_parent_build\'] == build[\'parent_build\']\n    assert composed[\'candidate\'] == build[\'parent\'] and composed[\'parent\'] == c[\'native_comparator\']\n    assert composed[\'source_inverse\'] == build[\'parent_inverse\'] == c[\'composition_parent_inverse\']\n    ci = json.loads(q.checked(c[\'composition_parent_inverse\']))\n    assert ci[\'exact_inverse\'] and ci[\'parent\'] == c[\'native_comparator\']\n    suffix = ci[\'suffix\']; assert suffix and source.endswith(suffix)\n    source = source[:-len(suffix)]\n    for edit in reversed(ci[\'edits\']):\n        assert source.count(edit[\'after\']) == 1\n        source = source.replace(edit[\'after\'], edit[\'before\'], 1)\n    assert source == q.checked(c[\'native_comparator\']).decode()\n    ei = json.loads(q.checked(build[\'engine_inverse\']))\n    assert ei[\'engine\'] == c[\'native_engine\'] == build[\'engine\'] == nc[\'engine\'] and ei[\'exact_inverse\']\n    assert ei[\'parent\'][\'sha256\'] == \'dac3ffed0495da3431a287b55d89b38d157afafae11809bcd1a4fbb6bf087558\'\n    assert ei[\'before\'] == "(640000000 if role == \'candidate\' or arm[\'source\'][\'sha256\'] == \'cf36eb1f82672a4d1e4396393922f88edc434b3cc07e82bf7ec1ac8b4da6db76\' else 512000000)"\n    assert ei[\'after\'] == "(640000000 if arm[\'source\'][\'sha256\'] == \'0f7b93c4c45c447d76f04f428c1aeec066c5eb30027671653565b7c5fe3d93c3\' else 512000000)"\n    custom = q.checked(ei[\'engine\']).decode()\n    assert custom.count(ei[\'after\']) == 1 and custom.replace(ei[\'after\'], ei[\'before\'], 1) == q.checked(ei[\'parent\']).decode()\n    pb = json.loads(q.checked(c[\'score_parent_build\']))\n    pe = json.loads(q.checked(pb[\'engine_inverse\']))\n    assert pb[\'candidate\'] == c[\'native_comparator\'] and pe[\'engine\'] == ei[\'parent\'] and pe[\'exact_inverse\']\n    assert pe[\'parent\'][\'sha256\'] == \'116793c99086668ca2c58f6288fc2f0b05d3970cc221191aae550577ceb39d5b\'\n    ps = q.checked(pe[\'engine\']).decode()\n    assert pe[\'before\'] == "(640000000 if role == \'candidate\' else 512000000)" and pe[\'after\'] == ei[\'before\']\n    assert ps.count(pe[\'after\']) == 1 and ps.replace(pe[\'after\'], pe[\'before\'], 1) == q.checked(pe[\'parent\']).decode()\n    cb = json.loads(q.checked(c[\'cache_build\']))\n    old_ei = json.loads(q.checked(cb[\'engine_inverse\']))\n    assert old_ei[\'engine\'] == pe[\'parent\'] and old_ei[\'parent\'] == c[\'control_native_engine\'] and old_ei[\'exact_inverse\']\n    assert c[\'control_native_engine\'][\'sha256\'] == \'1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287\'\n    assert old_ei[\'before\'] == "            assert 0 <= ubc[\'max_resident_bytes\'] <= 512000000"\n    assert old_ei[\'after\'] == "            assert 0 <= ubc[\'max_resident_bytes\'] <= (640000000 if role == \'candidate\' else 512000000)"\n    old_engine = q.checked(old_ei[\'engine\']).decode()\n    assert old_engine.count(old_ei[\'after\']) == 1 and old_engine.replace(old_ei[\'after\'], old_ei[\'before\'], 1) == q.checked(old_ei[\'parent\']).decode()\n    assert c[\'candidate_cache_limit_bytes\'] == c[\'control_cache_limit_bytes\'] == 512000000\n    assert c[\'score_parent_cache_limit_bytes\'] == 640000000\n    retained = json.loads(q.checked(c[\'score_parent_retained\']))\n    assert retained[\'status\'] == \'RETAINED_COMPONENT\' and retained[\'candidate\'] == c[\'native_comparator\']\n    parent_n8 = json.loads(q.checked(retained[\'native8\']))\n    parent_boundary = json.loads(q.checked(retained[\'boundary\']))\n    assert parent_n8[\'candidate\'] == c[\'native_comparator\'] and parent_n8[\'admitted\'] and parent_n8[\'passed_integrity\'] and all(parent_n8[\'gates\'].values())\n    assert parent_n8[\'all16_parent_bytes_exact\'] and retained[\'all16_parent_bytes_exact\']\n    assert parent_boundary[\'candidate\'] == c[\'native_comparator\'] and parent_boundary[\'passed\'] and parent_boundary[\'receipt_checks_passed\']\n    failed_worker = json.loads(q.checked(c[\'parent_serving_failure\']))\n    assert failed_worker[\'completed\'] is False and failed_worker[\'error\'] == "RuntimeError(\'RSS trigger\')"\n    assert failed_worker[\'whole_group_retired\'] and not failed_worker[\'remaining_rows\'] and failed_worker[\'max_swap_growth_bytes\'] == 0\n    assert failed_worker[\'max_rss_bytes\'] >= failed_worker[\'rss_trigger_bytes\'] == 7500000000\n    assert c[\'native_comparator\'][\'path\'] in failed_worker[\'command\']\n    parent6 = next(block for block in parent_n8[\'blocks\'] if block[\'size\'] == 6)[\'arms\'][\'candidate\']\n    for role in (\'base\', \'parity_parent\'):\n        native_arm(q, nc[\'arms\'][role], 6, parent6, 640000000)\n        for entry in nc[\'arms\'][role][\'blocks\'][\'6\'][\'artifacts\'][\'predictions\'].values():\n            q.checked(entry)\n    assert r[\'all16_parent_bytes_exact\'] is True\n'

def main():
    if sys.argv[1:] == ['--admit-boundary']:
        admit(); return
    assert not sys.argv[1:]
    old_assessor = T/'boundary_assess_s34.py'; old_config = T/'boundary_config.json'
    assert pin(old_assessor)['sha256'] == '606204b3bde7d3d0b0cf36d68d4fe34b14c18b78c0dce1ec78bbf2342ed19904'
    assert pin(old_config)['sha256'] == '1c76b0a183f44fd5128534527f36abf7b0f5ddf6b79ae204e10543618c48fda2'
    pilot = pin(D/'native_screen.json')
    assert not (D/'native_screen_continuation.json').exists()
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
    assert build['parent'] == pin(COMPOSITION/'candidate.py')
    assert nc['arms']['base']['source'] == nc['arms']['parity_parent']['source'] == pin(BASE/'candidate.py')
    parent_cfg = json.loads((BASE/'native_screen_continuation.json').read_text())
    parent_arm = parent_cfg['arms']['candidate']
    assert parent_arm['source'] == nc['arms']['base']['source']
    for size in ('2', '6'):
        block = parent_arm['blocks'][size]
        folder = H/block['tag']
        artifacts = dict(manifest=pin(folder/'manifest.json'), rows=pin(folder/'rows.jsonl'), guard=pin(block['guard']),
            predictions={str(rank): pin(folder/f'pred_{rank}.npy') for rank in block['ranks']})
        for role in ('base', 'parity_parent'):
            actual = nc['arms'][role]['blocks'][size]
            assert {k:v for k,v in actual.items() if k != 'artifacts'} == {k:v for k,v in block.items() if k != 'artifacts'}
            assert actual['artifacts'] == artifacts
    old = old_assessor.read_text(); new = old
    def replace_one(before, after):
        nonlocal new
        assert new.count(before) == 1, before
        new = new.replace(before, after, 1)
    replace_one('Prospective exact cast plus row scatter boundary2', 'Prospective forward-cache plus unique-gather boundary2')
    replace_one("'exact-cast-scatter-s34-boundary-native2-v1'", "'forward-cache-unique-gather-s34-boundary-native2-v1'")
    replace_one("640000000 if role == 'candidate' or cfg['arms'][role]['source']['sha256'] == 'cf36eb1f82672a4d1e4396393922f88edc434b3cc07e82bf7ec1ac8b4da6db76' else 512000000", "640000000 if cfg['arms'][role]['source']['sha256'] == '0f7b93c4c45c447d76f04f428c1aeec066c5eb30027671653565b7c5fe3d93c3' else 512000000")
    start = new.index('    # Exact scatter then three-cast inverse; retained plane comparator and engine chain.')
    end = new.index("    assert nc['parity_layers']", start)
    new = new[:start] + PROOF + new[end:]
    for name in ('paired_boundary_assessment.json', 'paired_boundary_assess_guard.json'):
        assert new.count(repr(name)) == old.count(repr(name)) == 1
    assessor = D/'boundary_assess_s34.py'; compile(new, str(assessor), 'exec')
    with assessor.open('x') as f: f.write(new)
    diff = D/'boundary_adapter.diff'
    with diff.open('x') as f: f.write(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(old_assessor), tofile=str(assessor))))
    c = json.loads(old_config.read_text())
    for key in ('composition_parent_build','composition_parent_native2','composition_parent_native2_guard','auxiliary_increment'): c.pop(key, None)
    c.update(schema='forward-cache-unique-gather-s34-boundary-native2-v1', engine=pin(assessor), candidate=pin(D/'candidate.py'),
        build=pin(D/'build.json'), source_protocol=pin(D/'protocol.json'), native_comparator=nc['arms']['base']['source'],
        composition_parent_build=pin(COMPOSITION/'build.json'), composition_parent_inverse=pin(COMPOSITION/'source_inverse.json'),
        score_parent_build=pin(BASE/'build.json'), parent_serving_failure=pin(BASE/'serving_v2_worker_guard.json'),
        native_config_path=str(D/'native_screen_continuation.json'), tag=D.name+'_boundary_feasibility_v1',
        setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']), native2_gates=copy.deepcopy(nc['stages']['2']['gates']),
        score_parent_retained=pin(BASE/'retained_component.json'), native_engine=nc['engine'],
        control_native_engine=pin(H/'native_screen_engine_v1/screen.py'), cache_build=pin(CACHE/'build.json'),
        candidate_cache_limit_bytes=512000000, score_parent_cache_limit_bytes=640000000, control_cache_limit_bytes=512000000)
    assert c['setup_counts'] == dict(bound=687, lvfuse=649, psdb_fused=9, pair_block_indices=1, read7_fused=4)
    assert c['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    c['interpretation'] = 'Forward-only cache512 plus unique-gather own boundary; comparator retained SC0f7 with its failed serving receipt preserved. Both source inverses bound; no unexecuted UG result used. S34 residual and S32 runtime controls unchanged, candidate512M, exactSC comparator640M and historical512M. Original all16/allwins gates; no reused broad pass.'
    bp = D/'boundary_config.json'; once(bp, c); bp = pin(bp)
    nc['stages']['2']['result'] = str(D/'native2_continuation_assessment.json')
    nc['stages']['2']['assessment_guard'] = str(D/'native2_continuation_assess_guard.json')
    req = nc['boundary_before8']; req.update(receipt=str(D/'boundary_assessment.json'), assessor=pin(assessor),
        assessment_guard=str(D/'boundary_assess_guard.json'), assessment_command=[str(LIGHT), '-B', str(assessor), bp['path'], bp['sha256']])
    nc['continuation'] = dict(pilot_config=pilot, pilot_result=pin(D/'native2_assessment.json'), pilot_assessment_guard=pin(D/'native2_assess_guard.json'),
        no_model_rerun=True, same_saved_native2_outputs=True, no_score_gate_changes=True, parent2_and6_artifacts_unchanged=True)
    nc['provenance'] += [pin(__file__), pin(assessor), pin(diff), bp, pin(old_assessor), pin(old_config),
        *[c[k] for k in ('score_parent_retained', 's34_native_config', 'parent_boundary', 's34_bank', 's34_evaluation', 'control_native_engine', 'cache_build', 'composition_parent_build', 'composition_parent_inverse', 'score_parent_build', 'parent_serving_failure')],
        *[nc['continuation'][k] for k in ('pilot_config', 'pilot_result', 'pilot_assessment_guard')]]
    nc['interpretation'] = 'Forward-cache plus unique-gather saved N2 reassessment, actual SC2+6 controls, own boundary and fixed6/N8; new full40/serving remain mandatory. Original exact improvement gates unchanged.'
    nc['prospective_scope'] = 'saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert all(nc[k] == original[k] for k in ('engine', 'parity_layers', 'future_only', 'raw_error_veto'))
    assert nc['arms'] == original['arms']
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
