"""Root-only precision/core TRAIN2/8 assessment; saved predictions, no model or targets.

Run under the native BLAS1 environment and a 120s/7.5GB/zero-swap guard.
TRAIN8 is conditional on the same source's admitted TRAIN2 receipt.
"""
from pathlib import Path
import hashlib
import io
import json
import math
import os
import sys
import numpy as np

D = Path(__file__).resolve().parent
H2 = D.parent
NATIVE = H2.parent/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
DRIVER = H2/'hill_climb_uvp_base_bcache_probe_v1/native_probe.py'
DRIVER_SHA = '8acc3e338ca5f991f778edaf0376bc8265ff69b330137c5bba4d079766ca8af5'
CANDIDATE_SHA = '83c4d267c754746321eefb2471f0e56881563d13b2320f6ba4f5b1f216d35084'
PARENT_SHA = '1571846b21141a16c2677ca371c4175a774d424d375ee656eb49a0b4e65ac5f3'
S32_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
PROTOCOL_SHA = 'ba9aa8e38c6e5329e52b7f7f0c6afd7535cad48816f0217ae13f9eb23f3ba935'
BLOCKS = ((35, 100), (36, 45, 52, 58, 64, 70))
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
BASE_PINS = {
    'hill_climb_star_projection32_train2_v1/manifest.json': '4327a61916482e1b699de367ca8c9a79ca9ff25c7467f1e37f4b54ce68285324',
    'hill_climb_star_projection32_train2_v1/rows.jsonl': '9dfa3606493dd151cc3c2086dd6b35012912b4a9db2eb043ce7847e8be62e510',
    'hill_climb_star_projection32_v1/train2_guard.json': '3eaaa150076f8f29fee64725dffe5f890e991239fc5d37829d8035eb72a94a04',
    'hill_climb_star_projection32_train6_v1/manifest.json': '728875ac00795f06ef06e5d3a32ca452496fd41f9e0b333e60643cd174d9ee29',
    'hill_climb_star_projection32_train6_v1/rows.jsonl': '2a4fbdd048be56be4fd62e6f7bf5c0e4905fc8a3545c32f0f37ec9b1828356f3',
    'hill_climb_star_projection32_v1/train6_guard.json': 'ab5be94d347c6dd32e9da1854567c84614fc6ae92ddb82116e258138d39df5f0',
}
PARENT_STUDY = H2/'hill_climb_precision_symmetry_bundle_v1'
PARENT_PINS = {
    'hill_climb_precision_symmetry_bundle_v1/native8_extension_assessment.json': 'f06e9cf9fe34da8f90c206f5a871914d53e7cded255d6a02d44f660693cf773a',
    'hill_climb_precision_symmetry_bundle_v1/native8_extension_assess_guard.json': 'cc2fc6afe5b737a0b27ddd0ce779f428417fdd84b6336238d703e17ea8430240',
    'hill_climb_precision_symmetry_bundle_train2_v1/manifest.json': '223c71581e0b8e4820a7afc18eebc4fafb63900c1707c068f258ac84452f4ef1',
    'hill_climb_precision_symmetry_bundle_v1/native2_guard.json': '22551c16137ee5948a4cd33e0c2af00d092ac0ad74838dc822cd9026f7415452',
    'hill_climb_precision_symmetry_bundle_train6_v1/manifest.json': '17b9fbf434db8280d782ace7aed2bd821a1d040dbf4390c33360de9dbf83d8ed',
    'hill_climb_precision_symmetry_bundle_v1/native6_guard.json': 'a6c43c592bf969b92da79b6f95d06518bac8caffd440338775325337f3579309',
}
EVIDENCE = {}


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def checked(record):
    data = Path(record['path']).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], record['path']
    EVIDENCE[record['path']] = record
    return data


def load(path):
    return json.loads(checked(pin(path)))


def guard(path, seconds, command=None):
    value = load(path)
    assert value['completed'] and value['exit_code'] == 0 and value['error'] is None
    assert value['whole_group_retired'] and value['remaining_rows'] == []
    assert value['rss_trigger_bytes'] == 7_500_000_000
    assert 0 <= value['max_rss_bytes'] < 7_500_000_000 and value['max_swap_growth_bytes'] == 0
    assert 0 <= value['elapsed_s'] <= seconds
    if command is not None:
        assert value['command'] == command
    return value


def build_check():
    build = load(D/'build.json')
    for key in ('parent', 'candidate', 'builder', 'protocol', 'fragment', 'parent_result', 'exact_parent_core_build'):
        checked(build[key])
    assert build['candidate'] == pin(D/'candidate.py') and build['candidate']['sha256'] == CANDIDATE_SHA
    assert build['parent']['sha256'] == PARENT_SHA and build['protocol'] == pin(D/'PROTOCOL.md')
    assert build['protocol']['sha256'] == PROTOCOL_SHA
    assert build['builder'] == pin(D/'build.py')
    parent, child = checked(build['parent']), checked(build['candidate'])
    assert build['fragment'] == pin(H2/'hill_climb_core_carry_consistency_v1/fragment.py')
    assert build['fragment']['sha256'] == 'a2defc728c49800f8bf078581db046ca75c40e8225d8db25b70e462c0baa6f4d'
    suffix = b'\n' + checked(build['fragment'])
    assert child == parent + suffix and child[:-len(suffix)] == parent
    assert build['append_only'] is True and build['source_inverse_exact'] is True
    assert build['suffix_bytes'] == len(suffix) == 417
    assert build['unchanged_scales'] == [1.0032287195131746, 0.9932375877606482]
    assert build['no_intervening_core_consumer'] is True
    assert build['nominal_core_update_layers'] == [7, 8, 9, 10, 11, 12]
    assert build['first_affected_core_consumer_layers'] == [10, 13]
    assert build['required_parent_prefix_layers'] == list(range(10))
    assert build['parent_result'] == pin(PARENT_STUDY/'native8_extension_assessment.json')
    assert build['parent_result']['sha256'] == PARENT_PINS[str((PARENT_STUDY/'native8_extension_assessment.json').relative_to(H2))]
    assert build['parent_original_bank_gate_passed'] is False and build['parent_failure_preserved'] is True
    checked(pin(D/'assess.py'))
    assert pin(DRIVER)['sha256'] == DRIVER_SHA
    checked(pin(DRIVER))
    return build['candidate']


def causal_parent_panel(size):
    # Provenance only: this parent's original 1% score-gate failure stays failed.
    folder = H2/f'hill_climb_precision_symmetry_bundle_train{size}_v1'
    result_path = PARENT_STUDY/'native8_extension_assessment.json'
    for path in (result_path, PARENT_STUDY/'native8_extension_assess_guard.json',
                 folder/'manifest.json', PARENT_STUDY/f'native{size}_guard.json'):
        assert pin(path)['sha256'] == PARENT_PINS[str(path.relative_to(H2))]
    receipt = load(result_path)
    assert receipt['passed_integrity'] and receipt['admitted'] is False and receipt['n'] == 8
    parent = pin(PARENT_STUDY/'candidate.py')
    assert receipt['candidate'] == parent and parent['sha256'] == PARENT_SHA
    checked(parent)
    guard(PARENT_STUDY/'native8_extension_assess_guard.json', 120,
          [str(NATIVE), '-B', str(PARENT_STUDY/'assess_extension.py'), '8'])
    ranks = list(BLOCKS[0 if size == 2 else 1])
    manifest = load(folder/'manifest.json')
    assert manifest['ranks'] == ranks and manifest['sha256'] == PARENT_SHA
    assert Path(manifest['candidate']).resolve() == Path(parent['path'])
    assert manifest['runtime'] == str(NATIVE) and manifest['numpy'] == '2.4.6'
    assert set(manifest['thread_env']) == set(THREADS) and all(x == '1' for x in manifest['thread_env'].values())
    guard(PARENT_STUDY/f'native{size}_guard.json', 150 if size == 2 else 300,
          [str(NATIVE), '-B', str(DRIVER), parent['path'], ','.join(map(str, ranks)), folder.name])
    evidence = {record['path']: record for record in receipt['evidence']}
    return {rank: evidence[str(folder/f'pred_{rank}.npy')] for rank in ranks}


def panel(size, is_child, child):
    ranks = list(BLOCKS[0 if size == 2 else 1])
    tag = f'hill_climb_precision_core_bundle_train{size}_v1' if is_child else f'hill_climb_star_projection32_train{size}_v1'
    folder = H2/tag
    gp = D/f'native{size}_guard.json' if is_child else H2/'hill_climb_star_projection32_v1'/f'train{size}_guard.json'
    if not is_child:
        for path in (folder/'manifest.json', folder/'rows.jsonl', gp):
            assert pin(path)['sha256'] == BASE_PINS[str(path.relative_to(H2))]
    manifest = load(folder/'manifest.json')
    assert manifest['ranks'] == ranks and manifest['runtime'] == str(NATIVE) and manifest['numpy'] == '2.4.6'
    assert set(manifest['thread_env']) == set(THREADS) and all(x == '1' for x in manifest['thread_env'].values())
    assert manifest['sha256'] == (CANDIDATE_SHA if is_child else S32_SHA)
    source = pin(manifest['candidate'])
    assert source['sha256'] == manifest['sha256']
    checked(source)
    if is_child:
        assert source == child
        command = [str(NATIVE), '-B', str(DRIVER), child['path'], ','.join(map(str, ranks)), tag]
    else:
        command = None  # Exact historical command is already bound by BASE_PINS.
    g = guard(gp, 150 if size == 2 else 300, command)
    setup = manifest['setup']
    assert math.isfinite(setup['setup_s']) and 0 <= setup['setup_s'] <= 4.1
    bound, fused = (660, 636) if is_child else (624, 600)
    assert [r[1] for r in setup['pb_log'] if r[0] == 'bound'] == [bound]
    assert [r[1] for r in setup['pb_log'] if r[0] == 'lvfuse'] == [fused]
    parent_predictions = causal_parent_panel(size) if is_child else None
    rows = [json.loads(line) for line in checked(pin(folder/'rows.jsonl')).decode().splitlines()]
    assert [r['rank'] for r in rows] == ranks
    for row in rows:
        assert all(math.isfinite(row[k]) and row[k] > 0 for k in ('mse', 'adjusted', 'util', 'wall_s', 'cpu_s'))
        assert type(row['flops']) is int and 0 < row['flops'] <= 2**41
        assert row['util'] == row['flops']/2**41
        assert math.isclose(row['adjusted'], row['mse']*max(.1, row['util']), rel_tol=1e-14, abs_tol=0)
        array = np.load(io.BytesIO(checked(pin(folder/f'pred_{row["rank"]}.npy'))), allow_pickle=False)
        assert array.shape == (16, 1024) and array.dtype == np.dtype('float64') and np.isfinite(array).all()
        if is_child:
            # Core is first consumed by join10; earlier means must be byte-exact.
            base_array = np.load(io.BytesIO(checked(parent_predictions[row['rank']])), allow_pickle=False)
            assert base_array.shape == (16, 1024) and base_array.dtype == np.dtype('float64') and np.isfinite(base_array).all()
            assert array[:10].tobytes() == base_array[:10].tobytes(), ('prefix_L0_9', row['rank'])
            del base_array
        del array
        if is_child:
            ubc = row['ubc']
            assert ubc['retired'] is True and ubc['retained_cache_bytes'] == ubc['live_cache_bytes'] == 0
            assert 0 <= ubc['max_resident_bytes'] <= 512_000_000
    return rows, g['max_rss_bytes']


def main():
    assert len(sys.argv) == 2 and sys.argv[1] in ('2', '8')
    assert Path(sys.executable).absolute() == NATIVE and np.__version__ == '2.4.6'
    assert all(os.environ.get(name) == '1' for name in THREADS)
    count = int(sys.argv[1])
    target = D/f'native{count}_assessment.json'
    assert not target.exists()
    child = build_check()
    if count == 8:
        prior = load(D/'native2_assessment.json')
        assert prior['passed_integrity'] and prior['admitted'] and prior['n'] == 2 and prior['candidate'] == child
        assert prior['protocol'] == pin(D/'PROTOCOL.md') and prior['assessor'] == pin(D/'assess.py')
        for record in prior['evidence']:
            checked(record)
        guard(D/'native2_assess_guard.json', 120, [str(NATIVE), '-B', str(D/'assess.py'), '2'])
    base, current, peaks = [], [], []
    for size in ((2,) if count == 2 else (2, 6)):
        for flag, output in ((False, base), (True, current)):
            rows, peak = panel(size, flag, child)
            output.extend(rows)
            peaks.append(peak)
    b, c = ({row['rank']: row for row in rows} for rows in (base, current))
    ranks = sorted(b)
    assert ranks == sorted(c) and len(ranks) == count
    delta = lambda key: 100*(sum(c[r][key] for r in ranks)/sum(b[r][key] for r in ranks)-1)
    adjusted, raw, bill = delta('adjusted'), delta('mse'), delta('flops')
    wins = sum(c[r]['adjusted'] < b[r]['adjusted'] for r in ranks)
    bb = np.array([b[r]['adjusted'] for r in ranks])
    cc = np.array([c[r]['adjusted'] for r in ranks])
    idx = np.random.default_rng(20260926).integers(0, count, size=(50000, count))
    ci = (100*np.quantile(cc[idx].mean(1)/bb[idx].mean(1)-1, [.025, .975])).tolist()
    gates = dict(adjusted_gain=adjusted <= -1., wins=wins >= (2 if count == 2 else 6), raw_regression=raw <= .25)
    if count == 8:
        gates['ci_upper_negative'] = ci[1] < 0
    result = dict(passed_integrity=True, admitted=all(gates.values()), n=count, ranks=ranks,
                  candidate=child, protocol=pin(D/'PROTOCOL.md'), assessor=pin(D/'assess.py'),
                  adjusted_pct=adjusted, raw_pct=raw, bill_pct=bill, adjusted_wins=wins,
                  adjusted_ci95_pct=ci, bootstrap=dict(resamples=50000, seed=20260926), gates=gates,
                  maximum_guard_rss_bytes=max(peaks), evidence=list(EVIDENCE.values()),
                  rows=[dict(rank=r, adjusted_pct=100*(c[r]['adjusted']/b[r]['adjusted']-1),
                             raw_pct=100*(c[r]['mse']/b[r]['mse']-1),
                             bill_pct=100*(c[r]['flops']/b[r]['flops']-1)) for r in ranks],
                  prefix_layers_0_9_byte_exact=True, all16_predictions_finite=True,
                  causal_prefix_parent=pin(PARENT_STUDY/'candidate.py'),
                  causal_prefix_receipt=pin(PARENT_STUDY/'native8_extension_assessment.json'),
                  score_comparator_sha256=S32_SHA,
                  targets_read=False, model_executed=False, serving_qualified=False, submitted=False)
    for record in list(EVIDENCE.values()):
        checked(record)
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('admitted', 'n', 'adjusted_pct', 'raw_pct', 'bill_pct', 'adjusted_wins', 'adjusted_ci95_pct', 'gates')}))


if __name__ == '__main__':
    main()
