"""Read-only preparation of exact stack quiet commands and mandatory original gates.

Never imports an estimator, launches native work, pauses research, or submits.
The lead must provide isolation and accept completed Linux/fresh evidence first.
"""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
A = HERE.parent
K = A.parent
R = K.parent
O = R.parent.parent
ROOT = next(p for p in HERE.parents if (p / '.venv/bin/python').exists())
PY = ROOT / '.venv/bin/python'
D = R / 'astra_oct20_bill_v1'
V = R / 'astra_oct20_vm_frontier_v1'
SCI = K / 'candidate_astra_o21_edge8_hidden_complement90_v1_sci_v1.py'
REL = K / 'candidate_rel_astra_o21_edge8_hidden_complement90_v1.py'
BANK = K / 'candidate_rel_astra_o21_edge8_local90_per_layer960_v1.py'
SCI_SHA = 'b54f81fff93462607a7129eb7d42ba619fb0b001f3a09b01e71c713a8740214c'
REL_SHA = '4685e1454ebd2d9607fcd88b64cc1d8435050761212efd1902b04d3f83151a46'
BANK_SHA = '660bc2d6daa5950c8614ca505775d305d3e19031a7a57eb0ac815b90ab5b8200'
SCI_RUN = D / 'hidden_complement90_v1/science32_edge8_hidden_complement90_parallel5_v1'
BANK_RUN = D / 'local_innovation90_v1/science32_edge8_local90_perlayer_parallel5_v1'
LAST = O / 'release/runs/astra_edge8_local90_perlayer_660bc2d6_v1/submission.tar.gz'
LAST_SHA = '24216de79706eabd70deb923083a3b9a51b0aede9856d1d424d8c8de6e534c2c'
HELPER = A / 'prepare_runtime_pair_parity_v3.py'
ISOLATION = A / 'qualify_runtime_companion_isolated_v3.py'
GUARD = R / 'astra_crunch_physics_v1/guard_memory50_v1.py'
B = R / 'astra_oct18_accuracy_v1/dual_thread_boundary_v3'
BOUNDARY = B / 'boundary_run_dual_v3.py'
GATE = R / 'fable_oct19_l13rank_v1/release_min_v1.sh'
TAG = 'astra_edge8_hidden_complement90_4685e145_v1'
QUIET = D / 'hidden_complement90_v1/edge8_complement_quiet_v1'
QUAL = D / 'hidden_complement90_v1/qual_edge8_complement_4685e145_v1'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
stamp = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def prepare():
    fixed = {SCI: SCI_SHA, REL: REL_SHA, BANK: BANK_SHA, LAST: LAST_SHA,
             HELPER: 'f0292b0045e5e21f661265e4807199fc437908e025706c5dc792d363a5ffa1b6',
             HERE / 'SECOND_FINAL10_AUTHORIZATION_v1.json': '94a642d19fca55940427432820157c33c531919b09bea4b10c29b5bfaabdb852',
             D / 'hidden_complement90_v1/MAC32_EDGE8_COMPLEMENT_INCREMENT_VS337466_v1.json': '95445fee610f0402460f479b87978bf3a697d995929883b69b2701245d1e8bc4'}
    for p, h in fixed.items():
        assert sha(p) == h, ('pin', str(p))
    helper = module(HELPER, 'edge90_runtime_validation')
    isolation = module(ISOLATION, 'edge90_science_validation')
    selected = {}
    full = {}
    for label, path, source_sha in [('bank', BANK_RUN, '434b3a4730febb062b59a8a29c042d27e1f4bdbf8301cef3a72c90b2e798fcb4'),
                                    ('candidate', SCI_RUN, SCI_SHA)]:
        result, grade, rr, cold, manifests = isolation.read_science(path, helper)
        assert result['candidate_sha256'] == source_sha
        if label == 'candidate':
            paired = read(D / 'hidden_complement90_v1/MAC32_EDGE8_COMPLEMENT_INCREMENT_VS337466_v1.json')
            assert paired['candidate_sha256'] == SCI_SHA
            assert Path(paired['reference']) == BANK_RUN
            assert paired['all32']['n'] == 32 and paired['all32']['adjusted_pct'] < 0
            assert paired['heldout25']['adjusted_pct'] < 0
        part = path / 'candidate_part01'
        assert [x['rank'] for x in helper.rows(part / 'rows.jsonl')] == [35, 150, 200]
        selected[label] = dict(source_sha256=source_sha, science_run=str(path),
            result_sha256=sha(path / 'RESULT32_v1.json'),
            references=[dict(rank=n, prediction=str(part / f'pred_{n}.bin'),
                             prediction_sha256=sha(part / f'pred_{n}.bin'), row=rr[n]) for n in (35, 150, 200)])
        keys = {json.dumps(x, separators=(',', ':')) for row in rr.values() for x in row.get('bind_list', [])}
        full[label] = dict(result=result, no_rescue_calls=sum(x['fast_rescue_stats']['calls'] for x in rr.values()),
            all_compile_missing_zero=all(x.get('compile_stats', {}).get('missing', 0) == 0 for x in rr.values()),
            all_compile_capacity_skipped_zero=all(x.get('compile_stats', {}).get('capacity_skipped', 0) == 0 for x in rr.values()),
            binding_keys=sorted(keys), cold_first_ranks=sorted(cold), manifest_hashes=manifests)
    a, b = ast.parse(SCI.read_text()), ast.parse(REL.read_text())
    sci_cap = ast.dump(ast.parse('_FAST_RESCUE_CAP_S = 1000000000.0').body[0], include_attributes=False)
    dump = lambda tree: [ast.dump(n, include_attributes=False) for n in tree.body]
    aa, bb = dump(a), dump(b)
    assert aa.count(sci_cap) == 1 and sci_cap not in bb
    assert [x for x in aa if x != sci_cap] == bb, 'Unexplained science/release difference'
    off = ast.dump(ast.parse('_FAST_RESCUE_FRAC = {}').body[0], include_attributes=False)
    assert aa[-1] == bb[-1] == off
    assert full['candidate']['no_rescue_calls'] == full['bank']['no_rescue_calls'] == 0
    for p in (QUIET, QUAL, O / 'release/runs' / TAG):
        assert not p.exists(), ('fresh output required', str(p))
    fixed.update({p: sha(p) for p in (GUARD, BOUNDARY, B / 'boundary_client_dual_v3.py',
        B / 'thread_counts_v3.py', O / 'release/common.py', O / 'release/server_light.py',
        R / 'astra_oct18_accuracy_v1/official_setup_five_v2.py', GATE,
        O / 'release/stage_a_package.py', O / 'fire/prefire_check_v1.py', ISOLATION,
        HERE / 'analyze_quiet_v1.py', Path(__file__))})
    commands = {}
    for label, source in [('bank', BANK), ('candidate', REL)]:
        dest = QUIET / label
        commands[label] = [str(PY), '-B', str(GUARD), '--record', str(dest / 'GUARD_v1.json'),
            '--seconds', '125', '--footprint-limit', '7500000000', '--rss-backstop', '7500000000',
            str(PY), '-B', str(BOUNDARY), str(source), '35,150,200', str(dest), '--short-setup']
    commands['analyze'] = [str(PY), '-B', str(HERE / 'analyze_quiet_v1.py'), '--manifest', str(HERE / 'MANIFEST_v1.json')]
    commands['release_min'] = ['bash', str(GATE), str(REL), TAG, str(QUAL), str(LAST)]
    new_keys = sorted(set(full['candidate']['binding_keys']) - set(full['bank']['binding_keys']))
    report = dict(utc=stamp(), status='COMMANDS_PREPARED_NO_NATIVE_LAUNCH',
        sources={'science': str(SCI), 'science_sha256': SCI_SHA, 'release': str(REL), 'release_sha256': REL_SHA,
                 'bank': str(BANK), 'bank_sha256': BANK_SHA},
        best_reference_submission=337466, last_fired_archive=str(LAST), last_fired_archive_sha256=LAST_SHA,
        source_pins={str(p): h for p, h in fixed.items()},
        parity={'full_AST_equal_after_removing_only_science_cap': True, 'last_assignment_permanent_no_governor': True,
                'science_cap_is_dead_with_final_empty_rescue_map': True, 'all32_actual_rescue_calls': 0,
                'no_source_or_prebinding_edits': True},
        actual32=full, actual_paired_337466=paired,
        legacy_result_denominator_note='Original RESULT32 schema retains historical336935 denominator; current gain is separately validated from actual VS_BEST_GRADED reference337466.',
        new_binding_keys_vs_bank=new_keys, selected=selected,
        quiet_output=str(QUIET), qualification=str(QUAL), release_tag=TAG,
        commands=commands, native_launched=False, root_submission_authorized_after_gates=True, agent_submission_authorized=False,
        authorization=read(HERE / 'SECOND_FINAL10_AUTHORIZATION_v1.json'),
        requirements=['Root drains all other laptop work and establishes quiet isolation before running commands.',
            'Root reviews actual VM32 and own fresh4 confirmation. Exact user authorization covers this source after all unchanged original gates.',
            'Create quiet bank/candidate directories once, run bank then candidate, each exact35,150,200 cold/warm sequence.',
            'Run analyzer before release_min; require exact own-science prediction bytes and FLOPs, original clean guards and unchanged source pins.',
            'Runtime ratios remain descriptive and require lead judgment; no hosted zero-failure guarantee.',
            'Run unchanged release_min: five setup starts under original7.5GB/free50 guard, StageA, prefire against exactLASTFIRED337466.',
            'No fire command is included. No queue, guard, reference or existing source mutation.'])
    with (HERE / 'MANIFEST_v1.json').open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')
    lines = ['# PREPARED ONLY. Root schedules isolation; no auto-fire.',
             '# Run only after the current laptop wrappers and required confirmations complete.',
             shlex.join(['mkdir', '-p', str(QUIET / 'bank'), str(QUIET / 'candidate')])]
    for label in ('bank', 'candidate'):
        lines.append(shlex.join(commands[label]) + ' > ' + shlex.quote(str(QUIET / label / 'run.log')) + ' 2>&1')
    lines += [shlex.join(commands['analyze']), '# Proceed only after analyzer PASS and lead runtime review.',
              shlex.join(commands['release_min']) + ' > ' + shlex.quote(str(QUIET / 'release_min.log')) + ' 2>&1']
    (HERE / 'COMMANDS_v1.txt').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(status=report['status'], manifest=str(HERE / 'MANIFEST_v1.json'),
        new_binding_key_count=len(new_keys), compile_missing_zero=full['candidate']['all_compile_missing_zero'],
        compile_capacity_skipped_zero=full['candidate']['all_compile_capacity_skipped_zero'])))


if __name__ == '__main__':
    prepare()
