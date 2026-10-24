"""Seal the preregistered CPU confirmation only after every truth chunk passes.

No estimator import, prediction, fit, process control, or native launch.
The original eight and additional four remain separately reported cohorts.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
CORE_SHA = 'f42f48cc5498b0753ccef5a245d80c2c6815f95e6df62249eb119a418a733038'
GUARD_SHA = '884199a8077fb576f49e99339da2211d59f53af630bb5e12750f665ff95a8646'
TEMPLATE_SHA = 'e0f3621a326995f48b539aa8cb3138be7a125334da0592aef880f3e5a4d10bff'
COHORTS = {
    'original': ('', 'production_v1', list(range(7109001, 7109009)),
                 '2a38e4c9fb9102d6f74946941b76120659e4451baf1cd4a89c128951b7af6ea7',
                 '81024b5866c8457e09e5a0acbaa4e37f3ad497c23c76d0b9a9f1d9a80d99fd7f'),
    'extension': ('EXTENSION_', 'production_extension_v1', list(range(7109009, 7109013)),
                  'af84df0f9add5636c1cd974d414efc34dc81bd2db4760425a2864a430772984d',
                  'f87b6812e2c6955e8b4eb477b9d08846a58e842b2d01151ba0af94d2e9d89e70'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def record(root, path, **extra):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(root.resolve())), sha256=sha(path),
                bytes=path.stat().st_size, **extra)


def verify_seed(root, folder, seed, worker_sha):
    """Reject gaps, resource failures, altered checkpoints or incomplete halves."""
    metas = sorted(folder.glob('checkpoint_*.json'))
    guards = sorted(folder.glob('chunk_*_GUARD.json'))
    assert len(metas) == len(guards) == 62, (seed, 'truth must have62 closed chunks')
    assert not list(folder.glob('*.tmp')), (seed, 'incomplete atomic write')
    weights = folder / 'weights.npy'
    weight_sha = sha(weights)
    with weights.open('rb') as stream:
        w = np.load(stream, allow_pickle=False)
    assert w.dtype == np.float32 and w.shape == (16, 1024, 1024)
    raw_sha = hashlib.sha256(memoryview(w).cast('B')).hexdigest()
    del w
    receipts = []
    last = None
    for index, meta in enumerate(metas):
        assert meta.name == f'checkpoint_{index:03d}.json'
        gp = folder / f'chunk_{index:03d}_GUARD.json'
        g, m = read(gp), read(meta)
        assert g['successful'] and g['whole_group_retired'] and g['child_exit'] == 0
        assert not g['cleanup_errors'] and g['guard_sha256'] == GUARD_SHA
        assert g['cgroup'] is True and g.get('max_owned_swap_bytes', 0) == 0
        assert g['thread_provenance']['blas_threads'] == 4
        half, start = ('a' if index < 31 else 'b'), (index % 31) * 32
        stop = min(start + 32, 978)
        assert m['successful'] and m['seed'] == seed and m['index'] == index
        assert (m['half'], m['start'], m['stop']) == (half, start, stop)
        assert (m['count_a'], m['count_b']) == ((stop, 0) if half == 'a' else (978, stop))
        assert m['worker_sha256'] == worker_sha and m['core_sha256'] == CORE_SHA
        assert m['weights_sha256'] == weight_sha and m['raw_weights_sha256'] == raw_sha
        assert m['blas_threads'] == 4 and m['static_os_nice'] == 10
        assert m['complete'] == (index == 61)
        checkpoint = folder / f'checkpoint_{index:03d}.npz'
        assert sha(checkpoint) == m['checkpoint_sha256']
        receipts.append(dict(index=index, metadata=record(root, meta), guard=record(root, gp),
                             checkpoint_sha256=m['checkpoint_sha256']))
        last = m
    truth = folder / f'truth_seed{seed}.npz'
    assert last['truth_sha256'] == sha(truth)
    assert (last['N'], last['N_half']) == (64094208, 32047104)
    with np.load(truth, allow_pickle=False) as data:
        assert int(data['seed']) == seed and str(data['weights_sha256']) == raw_sha
        assert str(data['cpu_protocol_sha256']) == CORE_SHA
        assert (int(data['N']), int(data['N_half']), int(data['batch'])) == (64094208, 32047104, 32768)
        assert np.array_equal(data['rng_seeds'], [seed * 10 + 3, seed * 10 + 1, seed * 10 + 2])
        for key in ('mc_post', 'mc_post_A', 'mc_post_B', 'mc_post_var'):
            value = data[key]
            assert value.shape == (16, 1024) and value.dtype == np.float64
            assert np.isfinite(value).all()
        means = data['batch_final_means']
        assert means.shape == (1956, 1024) and means.dtype == np.float64
        assert np.isfinite(means).all()
        for offset, key in ((0, 'mc_post_A'), (978, 'mc_post_B')):
            np.testing.assert_allclose(means[offset:offset + 978].mean(axis=0),
                                       data[key][-1], rtol=0, atol=2e-13)
    row = dict(weights=record(root, weights, raw_float32_sha256=raw_sha,
                              shape=[16, 1024, 1024], dtype='<f4'),
               truth_A=record(root, truth, key='mc_post_A'),
               truth_B=record(root, truth, key='mc_post_B'))
    return row, dict(seed=seed, checkpoints=receipts, final_metadata=last)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cohort', choices=('original', 'extension', 'all'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.PROOF.json').exists()
    template_path = HERE.parent / 'cloud_fresh4_v1/PLAN_COMPRESSION_CONFIRM_FRESH4_MARKLINE_v1.json'
    assert sha(template_path) == TEMPLATE_SHA
    template = read(template_path)
    names = list(COHORTS) if args.cohort == 'all' else [args.cohort]
    rows, proofs, preregs, splits, candidates = {}, {}, [], {}, None
    for name in names:
        prefix, production, seeds, prereg_sha, worker_sha = COHORTS[name]
        prereg_path = HERE / (prefix + 'CONFIRMATION_PREREGISTER_v1.json')
        assert sha(prereg_path) == prereg_sha
        prereg = read(prereg_path)
        assert prereg['seeds'] == seeds
        choices = prereg['candidate_choices_frozen_before_any_truth_completed']
        if candidates is None:
            candidates = choices
        else:
            assert candidates == choices
        complete_path = HERE / production / 'COMPLETE.json'
        complete = read(complete_path)
        assert complete['successful'] and not complete['failed'] and complete['no_fit']
        assert set(complete['truths']) == set(map(str, seeds))
        preregs.append(record(args.root, prereg_path))
        splits[name] = seeds
        for seed in seeds:
            row, proof = verify_seed(args.root, HERE / production / f'seed{seed}', seed, worker_sha)
            assert complete['truths'][str(seed)] == proof['final_metadata']
            rows[str(seed)] = row
            proofs[str(seed)] = proof
    assert set(candidates) == {'bank337422', 'edge8_nonlinear'}
    for key, candidate in candidates.items():
        path = Path(candidate['path'])
        assert not path.is_absolute() and '..' not in path.parts and not candidate.get('append_tail')
        source = args.root / path
        assert sha(source) == candidate['sha256'] == candidate['effective_sha256']
        assert ast.unparse(ast.parse(source.read_text()).body[-1]) == '_FAST_RESCUE_FRAC = {}'
    # Copy only shared environment fields; no stale historical reference or fixture metadata.
    plan = {key: copy.deepcopy(template[key]) for key in (
        'transfer_manifest', 'packages', 'sdk_sources', 'budget', 'max_array_count',
        'max_array_bytes', 'worker_source_pins')}
    for path, pin in plan['worker_source_pins'].items():
        assert sha(args.root / path) == pin
    seeds = [seed for name in names for seed in splits[name]]
    plan.update(utc=utc(), kind='Preregistered independent CPU truth confirmation',
        fixture_mode='fresh_selected_cold', groups=[[seed] for seed in seeds],
        first7=[], held25=[], rows=rows, candidates=candidates,
        reference_submission_id=337422, reference_candidate='bank337422',
        blas_threads=4, warmup_each=False, confirmation_splits=splits,
        confirmation_preregistrations=preregs, cpu_truth_core_sha256=CORE_SHA,
        N=64094208, N_half=32047104, batch=32768,
        fresh_protocol='Independent cold singletons; one paced setup and scored prediction; no warmup.',
        no_fit_or_model_selection=True, no_release_qualification=True,
        source_plan_sha256=TEMPLATE_SHA)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.open('x').write(json.dumps(plan, indent=2) + '\n')
    proof = dict(utc=utc(), successful=True, plan_sha256=sha(args.out),
                 builder_sha256=sha(__file__), cohort=args.cohort, seeds=seeds,
                 truth_proofs=proofs, metadata_and_fixture_checks_only=True,
                 estimator_imported=False, native_launched=False)
    args.out.with_suffix('.PROOF.json').open('x').write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(dict(plan=str(args.out), sha256=sha(args.out), seeds=seeds)))


if __name__ == '__main__':
    main()
