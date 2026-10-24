"""Stdlib-only inactive S33-score/S32-runtime config preparation; never runs jobs.

python -B build_configs.py STUDY NATIVE_CONFIG_PATH NATIVE_CONFIG_SHA S33_BANK_SHA
Writes exclusive score_controls_s33.json and broad_config_draft.json only.
Only root may promote the drafts and create admissions after all prerequisites.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import sys

D = Path(__file__).resolve().parent
H2 = D.parent
BASE = H2/'hill_climb_precision_g3_symdiag128_v1'
SERVING = H2/'serving_engine_v2'
BANK_SHA = '5a3cb090baaea04ff74e95780aea8aa799a9e99b18d0cadeab73957646ef8657'
SOURCE_SHA = '88683ee1ae144556393ba83e013ba975c63a2b91469fce5c6aee9f8f0c4db5d6'
PINS = {
    'broad_config.json': 'f8095ce35cc3cf88173648cf4a5ca7b05adcf763b93fe4ec840a6b08983c6ba3',
    'broad_assessment.json': '85bb2a5d89812725014f47b005d21268aea8244ca651031fe1f023a1954fa54e',
    'broad_assess_guard.json': 'd99ba3c4d834daaa57c4bd44a9b1b5730f66aca00494a05ebed911f9e78d472c',
    'serving_config.json': '1d5774330731d5ee28c3850c7c4716c16cba8d225e820c8bfcc99f8b48b4b4be',
}


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))


def raw(value):
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()


def write(path,value):
    with path.open('xb') as stream: stream.write(raw(value))
    return pin(path)


def enabled_pin(path,value):
    enabled = copy.deepcopy(value); assert enabled['prospective_authorization'] is False
    enabled['prospective_authorization'] = True; data = raw(enabled)
    return dict(path=str(path),sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))


def main():
    assert len(sys.argv) == 5
    study = Path(sys.argv[1]).resolve(); np = pin(Path(sys.argv[2]).resolve())
    assert study.is_dir() and study.parent == H2 and Path(np['path']).parent == study
    assert np['sha256'] == sys.argv[3]
    bank_pin = pin(H2/'official_bank_S33_v1.json')
    assert bank_pin['sha256'] == sys.argv[4] == BANK_SHA
    bank = json.loads(Path(bank_pin['path']).read_text())
    assert bank['submission_id'] == 332838 and bank['source_sha256'] == SOURCE_SHA
    assert bank['telemetry']['completed'] == 100 and bank['telemetry']['failed'] == 0
    assert bank['authenticated_poll']['grading_status_cd'] == 'graded'
    assert all(math.isfinite(bank['authenticated_poll'][k]) and bank['authenticated_poll'][k] > 0
               for k in ('score','score_secondary'))
    nc = json.loads(Path(np['path']).read_text())
    assert nc['schema'] == 'native-screen-v1' and nc['future_only'] is True
    assert nc['prospective_authorization'] is True and nc['arms']['base']['source'] == bank['candidate']
    assert nc['arms']['candidate']['source'] == pin(study/'candidate.py')
    parent_pins = {name:pin(BASE/name) for name in PINS}
    assert all(parent_pins[k]['sha256'] == digest for k,digest in PINS.items())
    bc = json.loads((BASE/'broad_config.json').read_text())
    br = json.loads((BASE/'broad_assessment.json').read_text())
    bg = json.loads((BASE/'broad_assess_guard.json').read_text())
    assert br['passed'] is True and br['receipt_checks_passed'] is True and br['validation_score_passed'] is True
    assert br['candidate'] == bank['candidate'] == bc['candidate']['source']
    assert bank['local_validation_reference'] == parent_pins['broad_assessment.json']
    assert br['config'] == parent_pins['broad_config.json']
    assert bg['completed'] is True and bg['exit_code'] == 0 and bg['error'] is None
    assert bg['whole_group_retired'] is True and bg['remaining_rows'] == []
    assert bg['rss_trigger_bytes'] == 7500000000 and 0 < bg['max_rss_bytes'] < 7500000000
    assert bg['max_swap_growth_bytes'] == 0 and 0 < bg['elapsed_s'] <= 120 and bg['samples'] > 0
    assert bg['command'] == [str(H2.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'),
        '-B',bc['engine']['path'],br['config']['path'],br['config']['sha256'],br['admission']['sha256']]
    reference = dict(submission_id=332838,source=bank['candidate'],bank_receipt=bank_pin)
    controls = dict(schema='saved-score-controls-v2',score_reference=reference,
        assessment=parent_pins['broad_assessment.json'],arm=copy.deepcopy(bc['candidate']),
        provenance=[*parent_pins.values(),bank_pin])
    assert [b['block'] for b in br['blocks']] == list(controls['arm']['blocks'])
    for block in br['blocks']:
        controls['arm']['blocks'][block['block']]['artifacts'] = block['arms']['candidate']
    output_paths = [study/n for n in ('score_controls_s33.json','broad_config_draft.json')]
    assert all(not p.exists() for p in output_paths)
    controls_pin = write(output_paths[0],controls)
    b = copy.deepcopy(bc)
    b.update(schema='valid40-screen-v2',prospective_authorization=False,
        engine=pin(D/'assess.py'),runner=pin(D/'run.py'),native_config=np,
        controls=controls_pin,score_reference=reference)
    b['runtime_reference'] = dict(official_submission_id=332738,
        source=json.loads(Path(bc['controls']['path']).read_text())['arm']['source'],
        native_config=pin(BASE/'native_screen.json'),native_arm='base',official=bc['official'])
    b['candidate'] = dict(source=nc['arms']['candidate']['source'],setup=nc['arms']['candidate']['setup'],
        ubc_retired_required=True,blocks={})
    for key,old in bc['candidate']['blocks'].items():
        block = copy.deepcopy(old); block['tag'] = study.name+'_valid40_'+key+'_v2'
        block['guard'] = str(study/('broad_'+key+'_guard.json'))
        b['candidate']['blocks'][key] = block
    ep = pin(study/'official_limit_extension_admission.json')
    ea = json.loads(Path(ep['path']).read_text())
    result = pin(study/'native8_official_limit_extension.json')
    training = json.loads(Path(result['path']).read_text())
    assert ea['root_admitted'] is True and ea['candidate'] == b['candidate']['source'] and ea['native_config'] == np
    assert training['passed_integrity'] is True and training['admitted'] is True and all(training['gates'].values())
    assert training['candidate'] == b['candidate']['source'] and training['config'] == np
    assert training['extension_admission'] == ep and training['extension_policy'] == ea['policy']
    assert training['original_boundary_passed'] is False and training['research_only'] is True
    adapter = pin(H2/'official_limit_research_extension_v1/native8.py')
    assert training['engine'] == adapter and training['reused_original_engine'] == nc['engine']
    b['research_extension'] = dict(admission=ep,policy=ea['policy'],engine=adapter,
        assessment=result,assessment_guard=pin(ea['assessment_guard']),original_boundary=ea['original_boundary'])
    b['provenance'] = [pin(__file__),bank_pin,*nc['provenance'],*b['research_extension'].values(),
        pin(H2/'valid40_engine_v2/assess.py'),pin(H2/'valid40_engine_v2/run.py'),pin(H2/'valid40_engine_v2/build_configs.py')]
    b['interpretation'] = 'Inactive VALID40 draft after explicit official-limit research extension. Original boundary stays failed; projected residual must remain below0.4, direct S32 wall projection105 and all other planning margins stay fixed. S33 score controls and full40 gates unchanged. Serving is separate after a score pass.'
    broad_pin = write(output_paths[1],b)
    broad_enabled = enabled_pin(study/'broad_config.json',b)
    print(json.dumps(dict(controls=controls_pin,broad_draft=broad_pin,broad_enabled=broad_enabled,
        research_extension=b['research_extension'],activated=False,serving_prepared=False),indent=2))


if __name__ == '__main__': main()
