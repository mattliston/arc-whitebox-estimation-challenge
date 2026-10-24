"""Root-only Gaussian R4/symdiag component TRAIN2/8 assessment; saved predictions, no model or targets.

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
CANDIDATE_SHA = '09dd5166b1edae161ef0b877d66df99b1891689ed2465280cf646d45dadacbfd'
PARENT_SHA = '1571846b21141a16c2677ca371c4175a774d424d375ee656eb49a0b4e65ac5f3'
S32_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
PROTOCOL_SHA = '04645d4f06564dd645ea13d00d653288bb80622bd92a3bc3fb1f0eb9f60e64dc'
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
POLICY_SHA = '26e99f9a54d97a4d6fca11ac477056219f7acf351d5f7327df9b6ff7cf898fa6'
BUILD_SHA = 'b14b262bb0c35cfb615bcee0be04a8e52a8ac739acd93064bb3060a6bc824315'
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
    assert pin(D/'build.json')['sha256'] == BUILD_SHA
    build = load(D/'build.json')
    for key in ('parent', 'candidate', 'builder', 'protocol', 'policy', 'component_comparator', 'bank_comparator',
                'gaussian_source', 'gaussian_build', 'symdiag_build', 'symdiag_source', 'fragment', 'pricing', 'symdiag_review'):
        checked(build[key])
    assert build['candidate'] == pin(D/'candidate.py') and build['candidate']['sha256'] == CANDIDATE_SHA
    assert build['parent'] == pin(PARENT_STUDY/'candidate.py') and build['parent']['sha256'] == PARENT_SHA
    assert build['protocol'] == pin(D/'PROTOCOL.md') and build['protocol']['sha256'] == PROTOCOL_SHA
    assert build['builder'] == pin(D/'build.py')
    assert build['policy'] == pin(H2/'hill_climb_policy_20260928_v2.json') and build['policy']['sha256'] == POLICY_SHA
    assert build['component_comparator'] == build['parent']
    assert build['bank_comparator']['sha256'] == S32_SHA
    parent, child = checked(build['parent']), checked(build['candidate'])
    assert build['fragment']['sha256'] == 'bbbc4bc7995d54a0484ce7cd3240ed0567c4dd0e4cbb6c21cd216ae43083146b'
    suffix = b'\n' + checked(build['fragment'])
    assert build['suffix_bytes'] == len(suffix) == 2486 and build['append_after_gaussian_source'] is True
    assert build['gaussian_source']['sha256'] == 'd3379e664e26d4fc5b88e1c3456d756173eaeeddc9ba36017755e3d682fa82bf'
    assert child == checked(build['gaussian_source']) + suffix
    assert checked(build['symdiag_source']) == parent + suffix
    assert build['selected_partition'] == [128, 64, 64]
    assert build['expected_setup'] == dict(bound=673, fused=649, sides=[200, 259, 190])
    restored = child[:-len(suffix)].decode()
    assert len(build['edits']) == 4 and build['source_inverse_exact'] is True
    for edit in reversed(build['edits']):
        assert restored.count(edit['after']) == 1
        restored = restored.replace(edit['after'], edit['before'], 1)
    assert restored.encode() == parent
    assert build['gaussian_order'] == dict(before=6, after=4, layers=list(range(1, 15)))
    assert build['plane_bond_order'] == build['local_bond_order'] == 2
    assert build['derivative_rows'] == list(range(7))
    assert build['first_gaussian_source_unchanged'] and build['terminal_scalar_source_unchanged']
    checked(pin(D/'assess.py'))
    assert pin(DRIVER)['sha256'] == DRIVER_SHA
    checked(pin(DRIVER))
    return build['candidate']


def parent_receipt():
    # The retained working parent's original bank-gate failure remains false.
    path = PARENT_STUDY/'native8_extension_assessment.json'
    gp = PARENT_STUDY/'native8_extension_assess_guard.json'
    for p in (path, gp):
        assert pin(p)['sha256'] == PARENT_PINS[str(p.relative_to(H2))]
    receipt = load(path)
    assert receipt['passed_integrity'] and receipt['admitted'] is False and receipt['n'] == 8
    assert receipt['candidate'] == pin(PARENT_STUDY/'candidate.py') and receipt['candidate']['sha256'] == PARENT_SHA
    guard(gp, 120, [str(NATIVE), '-B', str(PARENT_STUDY/'assess_extension.py'), '8'])
    return {record['path']: record for record in receipt['evidence']}


def panel(size, role, child, parent_evidence):
    ranks = list(BLOCKS[0 if size == 2 else 1])
    if role == 'candidate':
        tag = f'hill_climb_precision_g4_symdiag_train{size}_v1'
        gp, expected_sha = D/f'native{size}_guard.json', CANDIDATE_SHA
    elif role == 'parent':
        tag = f'hill_climb_precision_symmetry_bundle_train{size}_v1'
        gp, expected_sha = PARENT_STUDY/f'native{size}_guard.json', PARENT_SHA
    else:
        assert role == 's32'
        tag = f'hill_climb_star_projection32_train{size}_v1'
        gp, expected_sha = H2/'hill_climb_star_projection32_v1'/f'train{size}_guard.json', S32_SHA
    folder = H2/tag
    if role == 's32':
        for path in (folder/'manifest.json', folder/'rows.jsonl', gp):
            assert pin(path)['sha256'] == BASE_PINS[str(path.relative_to(H2))]
    elif role == 'parent':
        for path in (folder/'manifest.json', gp):
            assert pin(path)['sha256'] == PARENT_PINS[str(path.relative_to(H2))]
        assert pin(folder/'rows.jsonl') == parent_evidence[str(folder/'rows.jsonl')]
    manifest = load(folder/'manifest.json')
    assert manifest['ranks'] == ranks and manifest['runtime'] == str(NATIVE) and manifest['numpy'] == '2.4.6'
    assert set(manifest['thread_env']) == set(THREADS) and all(x == '1' for x in manifest['thread_env'].values())
    source = pin(manifest['candidate'])
    assert source['sha256'] == manifest['sha256'] == expected_sha
    checked(source)
    if role == 'candidate':
        assert source == child
    if role == 'parent':
        assert source == pin(PARENT_STUDY/'candidate.py')
    command = None if role == 's32' else [str(NATIVE), '-B', str(DRIVER), source['path'], ','.join(map(str, ranks)), tag]
    g = guard(gp, 150 if size == 2 else 300, command)
    setup = manifest['setup']
    assert math.isfinite(setup['setup_s']) and 0 <= setup['setup_s'] <= 4.1
    bound, fused = {'s32': (624, 600), 'parent': (660, 636), 'candidate': (673, 649)}[role]
    assert [r[1] for r in setup['pb_log'] if r[0] == 'bound'] == [bound]
    assert [r[1] for r in setup['pb_log'] if r[0] == 'lvfuse'] == [fused]
    rows = [json.loads(line) for line in checked(pin(folder/'rows.jsonl')).decode().splitlines()]
    assert [r['rank'] for r in rows] == ranks
    predictions = {}
    for row in rows:
        assert all(math.isfinite(row[k]) and row[k] > 0 for k in ('mse', 'adjusted', 'util', 'wall_s', 'cpu_s'))
        assert type(row['flops']) is int and 0 < row['flops'] <= 2**41
        assert row['util'] == row['flops']/2**41
        assert math.isclose(row['adjusted'], row['mse']*max(.1, row['util']), rel_tol=1e-14, abs_tol=0)
        record = pin(folder/f'pred_{row["rank"]}.npy')
        if role != 'candidate':
            assert record == parent_evidence[record['path']]
        checked(record)  # Hash before any array decode, after source/guard/roster validation.
        predictions[row['rank']] = record
        if role != 's32':
            ubc = row['ubc']
            assert ubc['retired'] is True and ubc['retained_cache_bytes'] == ubc['live_cache_bytes'] == 0
            assert 0 <= ubc['max_resident_bytes'] <= 512_000_000
    return rows, predictions, g['max_rss_bytes']


def metrics(base, current, ranks):
    b, c = ({row['rank']: row for row in rows} for rows in (base, current))
    assert set(b) == set(c) == set(ranks)
    delta = lambda key: 100*(sum(c[r][key] for r in ranks)/sum(b[r][key] for r in ranks)-1)
    bb = np.array([b[r]['adjusted'] for r in ranks])
    cc = np.array([c[r]['adjusted'] for r in ranks])
    idx = np.random.default_rng(20260926).integers(0, len(ranks), size=(50000, len(ranks)))
    ci = (100*np.quantile(cc[idx].mean(1)/bb[idx].mean(1)-1, [.025, .975])).tolist()
    return dict(adjusted_pct=delta('adjusted'), raw_pct=delta('mse'), bill_pct=delta('flops'),
                adjusted_wins=sum(c[r]['adjusted'] < b[r]['adjusted'] for r in ranks), adjusted_ci95_pct=ci,
                rows=[dict(rank=r, adjusted_pct=100*(c[r]['adjusted']/b[r]['adjusted']-1),
                           raw_pct=100*(c[r]['mse']/b[r]['mse']-1),
                           bill_pct=100*(c[r]['flops']/b[r]['flops']-1)) for r in ranks])


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
        assert prior['policy'] == pin(H2/'hill_climb_policy_20260928_v2.json')
        assert prior['component_comparator']['sha256'] == PARENT_SHA
        for record in prior['evidence']:
            checked(record)
        guard(D/'native2_assess_guard.json', 120, [str(NATIVE), '-B', str(D/'assess.py'), '2'])
    parent_evidence = parent_receipt()
    rows = {role: [] for role in ('s32', 'parent', 'candidate')}
    predictions = {role: {} for role in rows}
    peaks = []
    for size in ((2,) if count == 2 else (2, 6)):
        for role in rows:
            rr, pp, peak = panel(size, role, child, parent_evidence)
            rows[role].extend(rr)
            predictions[role].update(pp)
            peaks.append(peak)
    ranks = sorted(row['rank'] for row in rows['candidate'])
    assert len(set(ranks)) == count
    for rank in ranks:
        arrays = {}
        for role in rows:
            array = np.load(io.BytesIO(checked(predictions[role][rank])), allow_pickle=False)
            assert array.shape == (16, 1024) and array.dtype == np.dtype('float64') and np.isfinite(array).all()
            arrays[role] = array
        assert arrays['candidate'][0].tobytes() == arrays['parent'][0].tobytes(), ('L0', rank)
        del arrays, array
    component = metrics(rows['parent'], rows['candidate'], ranks)
    s32 = metrics(rows['s32'], rows['candidate'], ranks)
    gates = dict(adjusted_gain=component['adjusted_pct'] < 0 if count == 2 else component['adjusted_pct'] <= -.05)
    if count == 8:
        gates.update(wins=component['adjusted_wins'] >= 6, ci_upper_negative=component['adjusted_ci95_pct'][1] < 0)
    result = dict(passed_integrity=True, admitted=all(gates.values()), n=count, ranks=ranks, **component,
                  candidate=child, protocol=pin(D/'PROTOCOL.md'), assessor=pin(D/'assess.py'),
                  policy=pin(H2/'hill_climb_policy_20260928_v2.json'), gates=gates,
                  component_comparator=pin(PARENT_STUDY/'candidate.py'), versus_s32=s32,
                  bank_comparator=pin(H2/'hill_climb_star_projection32_v1/candidate.py'),
                  bootstrap=dict(resamples=50000, seed=20260926), maximum_guard_rss_bytes=max(peaks),
                  evidence=list(EVIDENCE.values()), layer0_byte_exact=True, all16_predictions_finite=True,
                  parent_bank_gate_failure_preserved=True, parent_result=pin(PARENT_STUDY/'native8_extension_assessment.json'),
                  raw_error_veto=False, component_only=True, bank_admitted=False,
                  targets_read=False, model_executed=False, serving_qualified=False, submitted=False)
    for record in list(EVIDENCE.values()):
        checked(record)
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: result[key] for key in ('admitted', 'n', 'adjusted_pct', 'raw_pct', 'bill_pct', 'adjusted_wins', 'adjusted_ci95_pct', 'gates', 'versus_s32')}))


if __name__ == '__main__':
    main()
