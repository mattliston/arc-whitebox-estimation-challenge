"""Prospective READ7 continuation from the frozen YNT/S34 boundary recipe; stdlib only."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json
import sys

D = Path(__file__).resolve().parent
H = D.parent
Y = H/'hill_climb_d5_ynt275_v1'
LIGHT = H.parents[4]/'.venv/bin/python'
NATIVE = H.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'


def pin(p):
    p = Path(p).resolve(); raw = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def once(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n')


def changed(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after, 1)


def source():
    original = Y/'boundary_assess_official_limit.py'
    anchor = Y/'boundary_assess_s34_reference_v2.py'
    assert pin(original)['sha256'] == 'f8a474c88e61e27179b54b9374d558b319ece544a62ef91429d9a2bbe339fa74'
    assert pin(anchor)['sha256'] == '14990ee6e432abb67bc434420034434242550d4dffad80f4e985d1a2a6a31c8f'
    old = original.read_text(); new = old
    new = changed(new, 'Prospective YNT2.75 boundary2: v3 with strict official residual cap and source inverse.',
        'Prospective READ7 boundary2: current-winner residual reference, unchanged S32 wall/CPU recipe.')
    new = new.replace('early-boundary-native2-official-limit-v1', 'read7-s34-boundary-native2-v1')
    new = changed(new, "build['source_inverse_exact'] is True", "build['exact_parent_prefix'] is True")
    start = new.index('    # A two-literal policy change:')
    end = new.index('    # The continuation only re-assesses', start)
    proof = '''    # Appended READ7 implementation: exact S34 prefix and actual N2 layer-0 parity.
    assert a['source_proof'] == c['build']
    parent = q.checked(build['parent']).decode()
    fragment = q.checked(build['fragment']).decode()
    assert build['parent'] == c['parent'] == nc['arms']['base']['source']
    assert build['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    assert build['fragment']['sha256'] == 'af467decdcac5ac8e005248984cd10220977df5ac75c27f660f5ec173932ca1e'
    assert q.checked(c['candidate']).decode() == parent + '\\n' + fragment
    assert nc['parity_layers'] == r['parity_layers'] == [0]
    assert r['configured_parent_bytes_exact'] is True
'''
    new = new[:start] + proof + new[end:]
    # Reuse the exact authenticated-bank/public-evaluation schema checks from v2.
    anchor_text = anchor.read_text()
    start = anchor_text.index("    bank = json.loads(q.checked(c['s34_bank']))")
    end = anchor_text.index("    candidate_rows, _ = q.boundary", start)
    reference = anchor_text[start:end]
    reference = reference.replace('BANK_SHA', "'8b2cba1ddc5381b4ce70948107511bb888fe86fb3f8766d80abac4417f37d0e2'")
    reference = reference.replace('telemetry', 's34_telemetry').replace('maximum', 's34_maximum')
    reference = reference.replace("['s34_telemetry']", "['telemetry']")
    reference += '''    native_parent_rows = native['base']
    assert [row['rank'] for row in parent_rows] == [row['rank'] for row in current] == [35, 100]
    assert [(row['rank'], row['flops']) for row in parent_rows] == [(row['rank'], row['flops']) for row in native_parent_rows]
    residual_comparisons = [dict(rank=x['rank'], candidate_residual_s=x['residual_s'], parent_residual_s=y['residual_s'],
        residual_delta_s=x['residual_s']-y['residual_s']) for x, y in zip(current, parent_rows)]
'''
    needle = "    residual = maximum+4*max(0., *(x['residual_delta_s'] for x in comparisons))\n"
    new = changed(new, needle, reference + "    residual = s34_maximum+4*max(0., *(x['residual_delta_s'] for x in residual_comparisons))\n")
    new = changed(new,
        "residual=dict(official_s32_max_s=maximum, multiplier=4, projected_s=residual, gate_s=.4, strict_upper_bound=True),",
        "residual=dict(official_s34_max_s=s34_maximum, multiplier=4, projected_s=residual, gate_s=.4, strict_upper_bound=True),\n"
        "        residual_reference=dict(source=c['parent'], official_bank=c['s34_bank'], official_evaluation=c['s34_evaluation'],\n"
        "            parent_boundary=c['parent_boundary'], parent_census=parent_receipt['current_census'], parent_guard=parent_receipt['guard'],\n"
        "            candidate_census=census, comparisons=residual_comparisons),")
    new = changed(new, "maximum_boundary_rss_bytes=max(g['max_rss_bytes'], bg['max_rss_bytes'])",
        "maximum_boundary_rss_bytes=max(g['max_rss_bytes'], bg['max_rss_bytes'], parent_guard['max_rss_bytes'])")
    new = new.replace('boundary_admission_extension_v2.json', 'boundary_admission_read7_v3.json')
    new = new.replace('boundary_assessment_extension_v2.json', 'boundary_assessment_read7_v3.json')
    new = new.replace('boundary_assess_guard_extension_v2.json', 'boundary_assess_guard_read7_v3.json')
    new = changed(new, "    setup = m.setup_record(q, d/'boundary.log', c['setup_counts']['bound'], c['setup_counts']['lvfuse'])\n", "    recovery = c['setup_recovery']\n    for key in ('launcher', 'harness', 'metadata_record'):\n        q.checked(recovery[key])\n    metadata = json.loads(q.checked(recovery['metadata_record']))\n    assert metadata['root_admitted'] is True and metadata['prediction_rerun'] is False\n    assert metadata['original_census'] == census and metadata['original_launcher'] == launcher and metadata['original_guard'] == gp\n    setup_launch_pin = pin(d/'setup_only_launcher.json')\n    setup_launch = json.loads(q.checked(setup_launch_pin))\n    assert setup_launch['candidate'] == c['candidate'] and setup_launch['original_harness'] == c['harness']\n    assert setup_launch['harness'] == recovery['harness'] and setup_launch['launcher'] == recovery['launcher']\n    assert setup_launch['client_exit'] == 0 and type(setup_launch['server_exit']) is int and setup_launch['server_retired'] is True\n    assert setup_launch['prediction_calls'] == 0 and setup_launch['original_census_unchanged'] is True\n    assert setup_launch['setup_log'] == pin(d/'setup_only.log')\n    setup_gp = pin(d/'setup_only_guard.json')\n    setup_guard = q.guard(setup_gp, [str(q.LIGHT), '-B', recovery['launcher']['path'], c['candidate']['path'],\n        '35,100', d.name+'_setup_only_v1'], 120)\n    setup = m.setup_record(q, d/'setup_only.log', c['setup_counts']['bound'], c['setup_counts']['lvfuse'])\n")
    new = changed(new, "parent_guard['max_rss_bytes'])", "parent_guard['max_rss_bytes'], setup_guard['max_rss_bytes'])")
    new = changed(new, "guard=gp, setup=setup, control_setup=base_setup, comparisons=comparisons, gates=gates,", "guard=gp, setup=setup, setup_recovery=dict(contract=recovery, launcher=setup_launch_pin, guard=setup_gp, original_setup_stdout_missing=True, prediction_rerun=False), control_setup=base_setup, comparisons=comparisons, gates=gates,")
    ast.parse(new)
    return old, new


def admit():
    cp = pin(D/'boundary_config_v3.json'); c = json.loads(Path(cp['path']).read_text())
    np = pin(D/'native_screen_continuation_v3.json'); nc = json.loads(Path(np['path']).read_text())
    result_pin = pin(nc['stages']['2']['result'])
    r = json.loads(Path(result_pin['path']).read_text())
    assert r['config'] == np and r['admitted'] and r['passed_integrity']
    assert r['candidate'] == c['candidate'] and all(r['gates'].values())
    gp = pin(nc['stages']['2']['assessment_guard']); guard = json.loads(Path(gp['path']).read_text())
    assert guard['command'] == [str(NATIVE), '-B', nc['engine']['path'], np['path'], np['sha256'], '2']
    assert guard['completed'] and guard['whole_group_retired'] and guard['exit_code'] == 0
    assert guard['error'] is None and guard['max_swap_growth_bytes'] == 0
    once(D/'boundary_admission_read7_v3.json', dict(root_admitted=True, source_changes=False, serving_qualified=False,
        config=cp, assessor=c['engine'], candidate=c['candidate'], native_config=np, native2=result_pin,
        native2_guard=gp, source_proof=c['build'], additional6_requires_boundary_pass=True,
        original_pilot_preserved=True, no_model_rerun_for_reassessment=True, official_residual_limit_s=.4,
        strict_residual_upper_bound=True, prospective_s34_reference=True))
    print(json.dumps(pin(D/'boundary_admission_read7_v3.json')))


def main():
    if sys.argv[1:] == ['--admit-boundary']:
        admit(); return
    assert not sys.argv[1:]
    pilot = pin(D/'native_screen.json')
    assert pilot['sha256'] == '54e3f68b6bd72785dda871de21ba794a43f25fa9c97469db1c93cb8a51f05903'
    nc = json.loads(Path(pilot['path']).read_text()); original = copy.deepcopy(nc)
    receipt = json.loads((D/'native2_assessment.json').read_text())
    assert receipt['config'] == pilot and receipt['passed_integrity'] and receipt['admitted']
    assert receipt['candidate'] == pin(D/'candidate.py') and receipt['parity_layers'] == [0]
    build = json.loads((D/'build.json').read_text())
    assert build['candidate'] == pin(D/'candidate.py') and build['exact_parent_prefix']
    assert Path(build['parent']['path']).read_bytes() + b'\n' + Path(build['fragment']['path']).read_bytes() == (D/'candidate.py').read_bytes()
    old, new = source(); assessor = D/'boundary_assess_s34_v3.py'
    with assessor.open('x') as f: f.write(new)
    diff = D/'boundary_adapter_v3.diff'
    with diff.open('x') as f:
        f.write(''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile=str(Y/'boundary_assess_official_limit.py'), tofile=str(assessor))))
    c = json.loads((Y/'boundary_config.json').read_text())
    reference = json.loads((Y/'boundary_s34_reference_v2_config.json').read_text())
    c.update(schema='read7-s34-boundary-native2-v1', prospective_authorization=True, engine=pin(assessor),
        candidate=pin(D/'candidate.py'), build=pin(D/'build.json'), source_protocol=pin(D/'protocol.json'),
        native_config_path=str(D/'native_screen_continuation_v3.json'), native_comparator=nc['arms']['base']['source'],
        tag=D.name+'_boundary_feasibility_v1', setup_counts=copy.deepcopy(nc['arms']['candidate']['setup']['counts']),
        native2_gates=copy.deepcopy(nc['stages']['2']['gates']))
    for key in ('parent', 'parent_boundary', 's34_bank', 's34_evaluation'): c[key] = reference[key]
    assert c['setup_counts'] == dict(bound=686, lvfuse=649, psdb_fused=9, read7_fused=4)
    c['interpretation'] = 'Prospective own READ7 boundary: pinned official S34 residual + 4 times nonnegative first-census delta versus D5, strict <0.4; original S32 wall/CPU projection, wall105/setup4.1/7.5GB/zeroSwap unchanged. Reassess immutable saved N2; no inherited candidate pass, no timing retry, no broad or bank authority.'
    recovery_path = D/'setup_metadata_recovery.json'
    once(recovery_path, dict(root_admitted=True, reason='Original setup stdout was lost by root filtering; no original setup value is inferred.',
        prediction_rerun=False, measurement='One separately guarded setup-only execution of original harness lines1-77.',
        original_census=pin(H/(c['tag']+'_census.json')), original_launcher=pin(H/(c['tag']+'_launcher.json')),
        original_guard=pin(D/'boundary_guard.json'), original_tail=pin(D/'boundary_tail.log'),
        source=pin(D/'candidate.py'), setup_launcher=pin(D/'run_setup_only.py'), setup_harness=pin(D/'setup_only_harness.py'),
        setup_cap_s=4.1, seconds=120, rss_bytes=7500000000, added_swap_bytes=0,
        prior_adapter_versions_preserved=[pin(D/'boundary_assess_s34.py'), pin(D/'boundary_assess_s34_v2.py')]))
    c['setup_recovery'] = dict(launcher=pin(D/'run_setup_only.py'), harness=pin(D/'setup_only_harness.py'), metadata_record=pin(recovery_path))
    boundary = D/'boundary_config_v3.json'; once(boundary, c); bp = pin(boundary)
    nc['prospective_authorization'] = True
    nc['stages']['2']['result'] = str(D/'native2_continuation_v3_assessment.json')
    nc['stages']['2']['assessment_guard'] = str(D/'native2_continuation_v3_assess_guard.json')
    req = nc['boundary_before8']
    req.update(receipt=str(D/'boundary_assessment_read7_v3.json'), assessor=pin(assessor),
        assessment_guard=str(D/'boundary_assess_guard_read7_v3.json'),
        assessment_command=[str(LIGHT), '-B', str(assessor), str(boundary), bp['sha256']])
    nc['continuation'] = dict(pilot_config=pilot, pilot_result=pin(D/'native2_assessment.json'),
        pilot_assessment_guard=pin(D/'native2_assess_guard.json'), no_model_rerun=True,
        same_saved_native2_outputs=True, no_score_gate_changes=True)
    nc['provenance'] += [pin(__file__), pin(assessor), pin(diff), bp,
        pin(Y/'boundary_assess_official_limit.py'), pin(Y/'boundary_assess_s34_reference_v2.py'),
        pin(Y/'boundary_s34_reference_v2_config.json'),
        *[c[k] for k in ('parent_boundary', 's34_bank', 's34_evaluation')],
        *[nc['continuation'][k] for k in ('pilot_config', 'pilot_result', 'pilot_assessment_guard')]]
    nc['provenance'] += list(c['setup_recovery'].values())
    nc['interpretation'] = 'Prospective READ7 continuation: reassess the same saved N2 outputs, then own boundary and fixed six before original native8 gates. No broad/bank authority; original pilot untouched.'
    nc['prospective_scope'] = 'saved_native2_reassessment_then_own_boundary_then_fixed6_and_native8'
    assert all(nc[k] == original[k] for k in ('engine', 'arms', 'parity_layers', 'future_only', 'raw_error_veto'))
    assert all(nc['stages'][s][k] == original['stages'][s][k] for s in ('2', '8') for k in ('gates', 'assessment_limits'))
    output = D/'native_screen_continuation_v3.json'; once(output, nc); np = pin(output)
    commands = dict(native2_reassessment=[str(NATIVE), '-B', nc['engine']['path'], str(output), np['sha256'], '2'],
        admission=[str(LIGHT), '-B', str(Path(__file__).resolve()), '--admit-boundary'],
        boundary=[str(LIGHT), '-B', c['driver']['path'], c['candidate']['path'], '35,100', c['tag']],
        boundary_assessment=req['assessment_command'],
        native8_assessment=[str(NATIVE), '-B', nc['engine']['path'], str(output), np['sha256'], '8'])
    commands['setup_only'] = [str(LIGHT), '-B', str(D/'run_setup_only.py'), str(D/'candidate.py'), '35,100', D.name+'_setup_only_v1']
    once(D/'continuation_commands_v3.json', dict(commands=commands, config=np, boundary_config=bp, assessor=pin(assessor),
        native2_guard=nc['stages']['2']['assessment_guard'], boundary_guard=str(D/'boundary_guard.json'),
        boundary_log=str(D/'boundary.log'), boundary_assessment_guard=req['assessment_guard'],
        native8_guard=nc['stages']['8']['assessment_guard']))
    assert pin(D/'native_screen.json') == pilot
    print(json.dumps(dict(config=np, boundary=bp, assessor=pin(assessor), builder=pin(__file__), commands=pin(D/'continuation_commands_v3.json'))))


if __name__ == '__main__': main()
