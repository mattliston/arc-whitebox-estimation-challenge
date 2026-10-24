"""Bind the single exported endpoint to existing RRPI native controls; stdlib only."""
from pathlib import Path
import copy
import hashlib
import json
import sys

D = Path(__file__).resolve().parent
H = D.parent
P = H / 'hill_climb_pair_rr_inverse_v1'
TEMPLATE = H / 'hill_climb_join10_tail_ritz_v1/native_screen.json'
TEMPLATE_SHA = '1fe88b20b8246021a54c8e4828093f0c6c2efa93e2089ce2539b7d1a3a2826bc'
PARENT_SHA = 'a3efb3a1066506e5bc2bc5a5da668056277ba65ea6cdd88c1de3be636fd419f3'


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked(record):
    assert pin(record['path']) == record, record['path']


def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    # Actual completed export only: no placeholder candidate or fabricated pins.
    assert len(sys.argv) == 1, 'Run only after candidate.py and build.json exist.'
    parent = pin(P / 'candidate.py')
    assert parent['sha256'] == PARENT_SHA
    assert pin(TEMPLATE)['sha256'] == TEMPLATE_SHA
    build = json.loads((D / 'build.json').read_text())
    protocol = json.loads((D / 'protocol.json').read_text())
    assert protocol['root_admitted'] is True
    assert protocol['baseline'] == parent and protocol['updates'] == 2 and protocol['batch'] == 4
    assert protocol['training_ranks'] == [38, 40, 41, 43, 44, 46, 47, 49]
    assert not set(protocol['training_ranks']) & {35, 100, 36, 45, 52, 58, 64, 70}
    candidate = pin(D / 'candidate.py')
    assert build['candidate'] == candidate and build['parent'] == parent
    assert build['protocol'] == pin(D / 'protocol.json')
    assert build['source_inverse_exact'] is True and build['frozen_parameter_bytes_exact'] is True
    assert build['only_five_learned_and_six_dial_assignments'] is True
    assert build['optimizer_updates'] == 2
    for key in ('candidate', 'parent', 'exporter', 'protocol', 'training_completion',
                'training_guard', 'endpoint', 'dials'):
        checked(build[key])
    done = json.loads(Path(build['training_completion']['path']).read_text())
    assert done['completed'] is True and done['optimizer_updates'] == 2 and done['checkpoints_saved'] == 1
    assert done['training_ranks'] == protocol['training_ranks']
    guard = json.loads(Path(build['training_guard']['path']).read_text())
    assert guard['completed'] is True and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] is True and guard['remaining_rows'] == []
    assert guard['max_swap_growth_bytes'] == 0
    assert 0 < guard['elapsed_s'] <= 600
    assert 0 < guard['max_rss_bytes'] < guard['rss_trigger_bytes'] <= 7500000000

    cfg = json.loads(TEMPLATE.read_text())
    for role in ('base', 'parity_parent'):
        assert cfg['arms'][role]['source'] == parent
        for block in cfg['arms'][role]['blocks'].values():
            for name, record in block['artifacts'].items():
                if name == 'predictions':
                    for item in record.values(): checked(item)
                else:
                    checked(record)
    cfg['engine'] = pin(D / 'screen_accuracy.py')
    inverse = json.loads((D / 'screen_accuracy_inverse.json').read_text())
    assert inverse['candidate'] == cfg['engine']
    checked(inverse['source_parent'])
    cfg['raw_error_veto'] = True
    cfg['parity_layers'] = [0]
    cfg['arms']['candidate'] = copy.deepcopy(cfg['arms']['base'])
    cfg['arms']['candidate']['source'] = candidate
    for size in ('2', '6'):
        block = cfg['arms']['candidate']['blocks'][size]
        block.pop('artifacts', None)
        block['tag'] = D.name + '_train' + size + '_v1'
        block['guard'] = str(D / ('native' + size + '_guard.json'))
    for size in ('2', '8'):
        stage = cfg['stages'][size]
        stage['result'] = str(D / ('native' + size + '_assessment.json'))
        stage['assessment_guard'] = str(D / ('native' + size + '_assess_guard.json'))
        stage['gates'] = dict(adjusted_pct_at_most=-0.1, strict_adjusted=False,
            wins_min=1 if size == '2' else 6, ci_upper_below=None if size == '2' else 0,
            max_per_network_regression_pct=1.0, additional6_gain_required=size == '8')
    cfg['boundary_before8']['receipt'] = str(D / 'NATIVE8_NOT_AUTHORIZED.json')
    cfg['boundary_before8']['assessment_guard'] = str(D / 'NATIVE8_NOT_AUTHORIZED_guard.json')
    cfg['prospective_scope'] = 'NATIVE2_ONLY'
    cfg['no_further_stage_authority'] = True
    cfg['root_execution_held_until_parent_qualification'] = False
    cfg['interpretation'] = ('One tail-aligned two-update warm endpoint versus actual RRPI controls. '
        'Both stages require raw and adjusted gain at least0.1%; L0 byte parity only. '
        'Own boundary and separately admitted continuation required before six/eight. No checkpoint selection.')
    cfg['provenance'] = [pin(D / n) for n in ('build.json', 'protocol.json',
        'prepare_native_screen.py', 'screen_accuracy_inverse.json', 'independent_training_source_review.json')]
    cfg['provenance'] += [build['training_completion'], build['training_guard'],
        pin(P / 'retained_component.json'), pin(P / 'native8_assessment.json'),
        pin(P / 'native8_assess_guard.json'), pin(P / 'native_screen_continuation.json'),
        pin(H / 'official_bank_S35_v1.json'), pin(TEMPLATE)]
    once(D / 'native_screen.json', cfg)
    config_pin = pin(D / 'native_screen.json')
    native = str(H.parent / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python')
    block = cfg['arms']['candidate']['blocks']['2']
    once(D / 'commands_native2.json', dict(config=config_pin,
        commands=dict(native2=[native, '-B', block['driver']['path'], candidate['path'],
                              '35,100', block['tag']],
            native2_assessment=[native, '-B', cfg['engine']['path'], config_pin['path'],
                                config_pin['sha256'], '2']),
        guards=dict(native2=block['guard'], native2_assessment=cfg['stages']['2']['assessment_guard']),
        limits=dict(native2_seconds=150, assessment_seconds=120, rss_bytes=7500000000, added_swap_bytes=0),
        thread_environment={key: '1' for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                               'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
        native6_roster=[36, 45, 52, 58, 64, 70], native6_requires_own_boundary_and_new_continuation=True))
    print(json.dumps(dict(config=config_pin, commands=pin(D / 'commands_native2.json'))))


if __name__ == '__main__':
    main()
