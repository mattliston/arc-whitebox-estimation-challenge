#!/usr/bin/env python3
"""Pinned same-arithmetic, fixed32, two-look promotion scorer.

No model execution. `early` reads final data only after every frozen pair is
complete. `full` rescores the exact saved predictions against separate64M
references. Both write new directories exclusively. `selftest` uses synthetic
numbers only. Without a subcommand this prints the source contract only.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
O = HERE.parent.parent
RUNNER = O / 'physicist/t85_pinned_judge_v7.py'
RUNNER_SHA = '3d792804140f05879a72efd5d67e9e89f72cb4de52df9c23ddea171f7e9ab2d0'
PLAN_SHA = '71f455d1d7f5f7239503468cacd5710f4db9d595c7f1be389c8d937e4f64a979'
PROTOCOL_SHA = 'a80bbcece92976858eaf5d49534437905b3c1263cb02c6426116182133687eca'
SEEDS = list(range(7100168, 7100200))
ROLES = ('baseline', 'candidate')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def pointer(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path))


def checked_pointer(meta):
    require(isinstance(meta, dict) and set(meta) == {'path', 'sha256'}, 'Malformed artifact pointer')
    require(sha(meta['path']) == meta['sha256'], 'Artifact changed: ' + meta['path'])
    return Path(meta['path'])


def publish(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def load_runner():
    require(sha(RUNNER) == RUNNER_SHA, 'Pinned T85 runner changed')
    spec = importlib.util.spec_from_file_location('a28_pinned_t85_v7', RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # T85 top level is stdlib only, no model import.
    return module


def validate_manifest(path):
    manifest = load_runner().validate(path)
    require(manifest['purpose'] == 'judge' and manifest['seeds'] == SEEDS, 'Exact frozen final32 required')
    require(manifest['schema'] == 'physicist_pinned_pair_judge_v2', 'Wrong judge schema')
    require(manifest['sources'].get(str(RUNNER)) == RUNNER_SHA, 'Wrong frozen runner')
    require(manifest['truth_plan']['sha256'] == PLAN_SHA and
            manifest['promotion_protocol']['sha256'] == PROTOCOL_SHA, 'Wrong prospective plan/protocol')
    require(manifest['promotion_scorer'] == pointer(__file__), 'Scorer not frozen in manifest')
    require(manifest['decision_receipt'] is not None, 'Missing fixed-candidate decision')
    require(manifest['baseline']['request_sha256'] != manifest['candidate']['request_sha256'], 'Identical judge arms')
    protocol = read(checked_pointer(manifest['promotion_protocol']))
    plan = read(checked_pointer(manifest['truth_plan']))
    require(protocol['final_seeds'] == SEEDS and protocol['plan_sha256'] == PLAN_SHA, 'Changed protocol seeds')
    require(protocol['bootstrap'] == dict(resamples=100000, resample_unit='network', paired=True,
            generator='NumPyPCG64', seed=2026101101, quantile_method='linear',
            all32_before_either_summary=True), 'Changed bootstrap')
    require(protocol['early']['N'] == 15990784 and protocol['early']['bootstrap_upper_quantile'] == .995,
            'Changed early look')
    require(protocol['refinement']['N'] == 64094208 and protocol['refinement']['bootstrap_upper_quantile'] == .98,
            'Changed full look')
    for source, digest in plan['source_sha256'].items():
        require(sha(source) == digest, 'Truth source changed')
    return manifest, protocol, plan


def ready32(manifest):
    paths = [Path(manifest['run']) / ('seed%d' % seed) / 'COMPLETE.json' for seed in SEEDS]
    require(all(p.is_file() for p in paths), 'All32 COMPLETE receipts required before any score is read')
    return paths


def prediction_array(values):
    import numpy as np
    result = np.asarray(values, dtype=np.float64)
    require(result.shape == (16, 1024) and np.isfinite(result).all(), 'Prediction shape/finiteness failure')
    return result


def truth_array(entry, expected_N, expected_weights_sha, source_map):
    import numpy as np
    path = Path(entry['path'])
    require(sha(path) == entry['sha256'], 'Truth file hash mismatch')
    with np.load(path, allow_pickle=False) as z:
        require(int(z['seed']) == entry['seed'] and int(z['N']) == expected_N, 'Wrong truth seed/N')
        require(int(z['N_half']) * 2 == expected_N and int(z['batch']) == 32768, 'Wrong halves/batch')
        require(int(z['N_half']) % 32768 == 0 and (int(z['N_half']) // 32768) % 2 == 0, 'Unbalanced batch signs')
        require(str(z['weights_sha256']) == expected_weights_sha, 'Truth weights changed')
        seed = entry['seed']
        require(z['rng_seeds'].tolist() == [seed * 10 + 3, seed * 10 + 1, seed * 10 + 2], 'MC streams changed')
        arrays = {k: np.asarray(z[k], dtype=np.float64) for k in
                  ('mc_post', 'mc_post_A', 'mc_post_B', 'mc_post_var')}
    require(all(a.shape == (16, 1024) and np.isfinite(a).all() for a in arrays.values()), 'Invalid truth arrays')
    require(np.allclose(arrays['mc_post'], (arrays['mc_post_A'] + arrays['mc_post_B']) / 2,
                        rtol=1e-12, atol=1e-14), 'Pooled truth does not match equal halves')
    require(all(sha(p) == s for p, s in source_map.items()), 'Reference source changed')
    return arrays


def pair_metrics(base, candidate, truth, N):
    import numpy as np
    b, c = base[-1], candidate[-1]
    t, A, B = (truth[k][-1] for k in ('mc_post', 'mc_post_A', 'mc_post_B'))
    bm, cm = float(np.mean((b - t) ** 2)), float(np.mean((c - t) ** 2))
    require(math.isfinite(bm) and math.isfinite(cm) and bm > 0 and cm >= 0, 'Invalid paired MSE')
    q = -float(np.mean((c - b) * (A - B)))
    return dict(baseline_mse=bm, candidate_mse=cm, ratio=cm / bm, difference=cm - bm,
                half_projection_q=q, reference_noise_estimate=float(np.mean(truth['mc_post_var'][-1])) / N)


def summarize(rows, look):
    import numpy as np
    require(len(rows) == 32, 'Exactly32 pairs required')
    require(look in ('early', 'full'), 'Only the two prospective looks exist')
    b = np.array([r['baseline_mse'] for r in rows], dtype=np.float64)
    c = np.array([r['candidate_mse'] for r in rows], dtype=np.float64)
    q = np.array([r['half_projection_q'] for r in rows], dtype=np.float64)
    require(np.isfinite(b).all() and np.isfinite(c).all() and np.isfinite(q).all()
            and (b > 0).all() and (c >= 0).all(), 'Nonfinite or invalid scores')
    ratio = float(c.sum() / b.sum())  # Pooled ratio, never mean(c / b).
    rng = np.random.Generator(np.random.PCG64(2026101101))
    index = rng.integers(0, 32, size=(100000, 32))
    bootstrap = c[index].sum(1) / b[index].sum(1)
    quantile = .995 if look == 'early' else .98
    upper = float(np.quantile(bootstrap, quantile, method='linear'))
    qnorm = float(np.sqrt(np.sum(q * q)))
    variance = float(np.std(c - b, ddof=1) / np.sqrt(32))
    return dict(look=look, n=32, pooled_ratio=ratio, pooled_change_pct=100 * (ratio - 1),
                upper_quantile=quantile, bootstrap_upper=upper,
                descriptive_bootstrap_95=[float(v) for v in np.quantile(bootstrap, [.025, .975], method='linear')],
                accuracy_pass=bool(ratio < 1 and upper < 1), native_qualification_required=True,
                automatic_fire=False, wins=int(np.sum(c < b)), mean_paired_difference=float(np.mean(c - b)),
                descriptive_network_SE_difference=variance,
                MC_only_SE_mean_difference=qnorm / 32, MC_only_SE_relative_to_baseline=qnorm / float(b.sum()),
                MC_diagnostic_not_added_to_bootstrap=True,
                bootstrap=dict(resamples=100000, generator='NumPyPCG64', seed=2026101101,
                               paired=True, quantile_method='linear'),
                numpy_version=np.__version__)


def early(args):
    import numpy as np
    manifest, protocol, plan = validate_manifest(args.judge_manifest)
    complete_paths = ready32(manifest)  # Before reading any result or truth values.
    entries = {e['seed']: e for e in plan['entries']}
    for seed in SEEDS:
        e = entries[seed]
        require(e['role'] == 'final' and e['N'] == 15990784 and Path(e['completion_receipt']).is_file(),
                'All32 final truth receipts required')
    out = Path(args.out).resolve()
    out.mkdir()  # Never overwrite/reuse a partial or completed attempt.
    saved_pairs, rows = [], []
    for seed, completed_path in zip(SEEDS, complete_paths):
        completed = read(completed_path)
        require(completed['manifest_identity'] == manifest['identity_sha256'], 'Mixed judge manifests')
        records = completed['records']
        require(len(records) == 2 and records[0]['truth_metadata'] == records[1]['truth_metadata'], 'Unpaired truth metadata')
        e, metadata = entries[seed], records[0]['truth_metadata']
        require(metadata['plan'] == manifest['truth_plan'] and metadata['seed'] == seed and metadata['role'] == 'final',
                'Wrong truth metadata')
        require(all(metadata[k] == e[k] for k in ('path', 'N', 'N_half')), 'Wrong planned truth')
        require(metadata['source_sha256'] == plan['source_sha256'], 'Wrong reference sources')
        receipt = read(checked_pointer(metadata['completion_receipt']))
        require(metadata['completion_receipt']['path'] == e['completion_receipt'], 'Unplanned receipt')
        require(receipt['schema'] == 'analyst_truth_completion_v1', 'Wrong truth receipt schema')
        require(receipt['plan_sha256'] == PLAN_SHA and receipt['source_sha256'] == plan['source_sha256'], 'Receipt plan mismatch')
        require(all(receipt[k] == metadata[k] for k in ('seed', 'path', 'N', 'N_half', 'sha256', 'weights_sha256')),
                'Receipt identity mismatch')
        predictions, saved = [], dict(seed=seed, truth_metadata=metadata, complete=pointer(completed_path))
        for index, role in enumerate(ROLES):
            record_meta = records[index]
            rec = read(checked_pointer(dict(path=record_meta['path'], sha256=record_meta['sha256'])))
            expected_id = 't85_%d_%s_r%d' % (index, role, seed)
            require(record_meta['id'] == expected_id and rec['id'] == expected_id, 'Wrong ordered pair')
            require(rec['rank'] == seed and rec['mode'] == 'eval' and rec['params'] == manifest[role]['request']['params'],
                    'Wrong prediction identity')
            require(rec['cutsim'] == manifest['profile'] and rec['threads'] == manifest['threads'], 'Wrong scientific runtime')
            require(rec['truth_plan'] == manifest['truth_plan'] and rec['truth_metadata'] == metadata, 'Worker truth mismatch')
            require(rec['mse'] == record_meta['mse'], 'COMPLETE/worker MSE mismatch')
            require(identity(rec['prediction']) == record_meta['prediction_sha256'], 'Prediction JSON hash mismatch')
            prediction = prediction_array(rec['prediction'])
            destination = out / ('seed%d_%s.npy' % (seed, role))
            with destination.open('xb') as stream:
                np.save(stream, prediction, allow_pickle=False)
            saved[role] = dict(**pointer(destination), json_prediction_sha256=record_meta['prediction_sha256'],
                               worker_record=dict(path=record_meta['path'], sha256=record_meta['sha256']))
            predictions.append(prediction)
        truth_entry = dict(e, sha256=metadata['sha256'])
        truth = truth_array(truth_entry, 15990784, metadata['weights_sha256'], plan['source_sha256'])
        metrics = pair_metrics(*predictions, truth, 15990784)
        for index, role in enumerate(ROLES):
            require(math.isclose(metrics[role + '_mse'], records[index]['mse'], rel_tol=1e-10, abs_tol=1e-20),
                    'Saved prediction does not reproduce final-layer worker MSE')
        rows.append(dict(seed=seed, N=15990784, truth_sha256=metadata['sha256'], **metrics))
        saved_pairs.append(saved)
    validate_manifest(args.judge_manifest)  # Source and frozen-point stability across scoring.
    capsule = dict(schema='analyst_frozen_prediction_capsule_v1', created_utc=utc(), seeds=SEEDS,
                   judge_manifest=pointer(args.judge_manifest), manifest_identity=manifest['identity_sha256'],
                   promotion_protocol=manifest['promotion_protocol'], truth_plan=manifest['truth_plan'],
                   promotion_scorer=pointer(__file__), pairs=saved_pairs)
    publish(out / 'predictions.json', capsule)
    report = dict(schema='analyst_two_look_report_v1', created_utc=utc(), capsule=pointer(out / 'predictions.json'),
                  protocol=manifest['promotion_protocol'], **summarize(rows, 'early'), rows=rows)
    publish(out / 'report.json', report)
    complete = dict(schema='analyst_early_complete_v1', created_utc=utc(),
                    capsule=pointer(out / 'predictions.json'), report=pointer(out / 'report.json'),
                    judge_manifest=pointer(args.judge_manifest), scorer=pointer(__file__))
    publish(out / 'EARLY_COMPLETE.json', complete)
    print(json.dumps(dict(completion=str(out / 'EARLY_COMPLETE.json'), accuracy_pass=report['accuracy_pass'],
                          pooled_ratio=report['pooled_ratio'], upper=report['bootstrap_upper'], native_required=True)))


def full(args):
    import numpy as np
    first = read(args.early_complete)
    require(first['schema'] == 'analyst_early_complete_v1' and first['scorer'] == pointer(__file__), 'Wrong early completion')
    capsule = read(checked_pointer(first['capsule']))
    early_report = read(checked_pointer(first['report']))
    manifest_path = checked_pointer(first['judge_manifest'])
    manifest, protocol, plan = validate_manifest(manifest_path)
    require(capsule['judge_manifest'] == first['judge_manifest'] and capsule['seeds'] == SEEDS and
            capsule['manifest_identity'] == manifest['identity_sha256'], 'Changed frozen pair')
    require(capsule['promotion_scorer'] == first['scorer'] and capsule['promotion_protocol'] == manifest['promotion_protocol'],
            'Changed capsule contract')
    require(early_report['capsule'] == first['capsule'] and early_report['look'] == 'early', 'Wrong early report')
    require(not early_report['accuracy_pass'], 'Early passed; no second statistical look permitted')
    # Reproduce the already recorded early decision deterministically. No new early look/data.
    require([r['seed'] for r in early_report['rows']] == SEEDS and all(r['N'] == 15990784 for r in early_report['rows']), 'Early row identity changed')
    reproduced = summarize(early_report['rows'], 'early')
    require(all(reproduced[k] == early_report[k] for k in ('bootstrap_upper', 'pooled_ratio', 'accuracy_pass')), 'Early report changed')
    refinement = read(args.truth_manifest)
    require(refinement['schema'] == 'analyst_full_reference_refinement_v1', 'Wrong refinement schema')
    require(refinement['origin_early_complete'] == pointer(args.early_complete), 'Refinement belongs to another candidate')
    require(refinement['promotion_protocol'] == manifest['promotion_protocol'] and refinement['seeds'] == SEEDS,
            'Wrong refinement protocol/seeds')
    require([e['seed'] for e in refinement['entries']] == SEEDS, 'Exactly32 full references required')
    sources = refinement['source_sha256']
    # Core and loader must be the original MC scheme. A new publication wrapper may have its own hash.
    for p, digest in plan['source_sha256'].items():
        if Path(p).name in ('mc_truth.py', 'loader.py'):
            require(sources.get(p) == digest, 'Refinement changed the MC generator/core')
    require(all(sha(p) == digest for p, digest in sources.items()), 'Refinement source changed')
    review_pointer = refinement['publication_review']
    review = read(checked_pointer(review_pointer))
    require(review['schema'] == 'analyst_full_publication_review_v1' and review['review_status'] == 'ROOT_REVIEWED',
            'Full publication wrapper must be reviewed before scoring')
    require(review['source_sha256'] == sources and review['origin_early_complete'] == pointer(args.early_complete)
            and review['promotion_protocol'] == manifest['promotion_protocol'], 'Publication review lineage mismatch')
    require(sources.get(review['publication_wrapper']['path']) == review['publication_wrapper']['sha256'],
            'Missing reviewed publication wrapper pin')
    checked_pointer(review['publication_wrapper'])
    require(all(e['N'] == 64094208 and e['N_half'] == 32047104 and
                Path(e['path']).is_file() and Path(e['completion_receipt']['path']).is_file()
                for e in refinement['entries']), 'All32 full references must be complete before scores')
    require([p['seed'] for p in capsule['pairs']] == SEEDS, 'Prediction capsule lost pairs')
    # One immutable full-reference lineage per early attempt, independent of output directory.
    # Same-manifest deterministic retries do not spend another look; another manifest is forbidden.
    claim = dict(schema='analyst_full_look_claim_v1', early_complete=pointer(args.early_complete),
                 refinement_manifest=pointer(args.truth_manifest), publication_review=review_pointer,
                 scorer=pointer(__file__))
    claim_path = Path(args.early_complete).resolve().parent / 'FULL_LOOK_CLAIM.json'
    try:
        publish(claim_path, claim)
    except FileExistsError:
        require(read(claim_path) == claim, 'A different full-reference look was already claimed')
    out = Path(args.out).resolve()
    out.mkdir()
    rows = []
    for pair, entry in zip(capsule['pairs'], refinement['entries']):
        seed = entry['seed']
        require(Path(entry['path']).resolve() != Path(pair['truth_metadata']['path']).resolve(), 'Refinement would reuse early path')
        completion = read(checked_pointer(entry['completion_receipt']))
        require(completion['schema'] == 'analyst_truth_completion_v1', 'Wrong full receipt schema')
        require(completion['source_sha256'] == sources, 'Full receipt source mismatch')
        require(all(completion[k] == entry[k] for k in ('seed', 'path', 'N', 'N_half', 'sha256', 'weights_sha256')),
                'Full receipt identity mismatch')
        require(entry['weights_sha256'] == pair['truth_metadata']['weights_sha256'], 'Full truth is another network')
        predictions = []
        for role in ROLES:
            saved = pair[role]
            path = checked_pointer(dict(path=saved['path'], sha256=saved['sha256']))
            prediction = prediction_array(np.load(path, allow_pickle=False))
            require(identity(prediction.tolist()) == saved['json_prediction_sha256'], 'Frozen prediction changed')
            predictions.append(prediction)
        truth = truth_array(entry, 64094208, entry['weights_sha256'], sources)
        rows.append(dict(seed=seed, N=64094208, truth_sha256=entry['sha256'], **pair_metrics(*predictions, truth, 64094208)))
    validate_manifest(manifest_path)
    report = dict(schema='analyst_two_look_report_v1', created_utc=utc(), early_complete=pointer(args.early_complete),
                  capsule=first['capsule'], refinement_manifest=pointer(args.truth_manifest),
                  protocol=manifest['promotion_protocol'], **summarize(rows, 'full'), rows=rows)
    publish(out / 'report.json', report)
    publish(out / 'FULL_COMPLETE.json', dict(schema='analyst_full_complete_v1', created_utc=utc(),
            report=pointer(out / 'report.json'), early_complete=pointer(args.early_complete), scorer=pointer(__file__),
            refinement_manifest=pointer(args.truth_manifest), full_look_claim=pointer(claim_path), publication_review=review_pointer))
    print(json.dumps(dict(completion=str(out / 'FULL_COMPLETE.json'), accuracy_pass=report['accuracy_pass'],
                          pooled_ratio=report['pooled_ratio'], upper=report['bootstrap_upper'], native_required=True)))


def selftest(args):
    import numpy as np
    tests = []
    rows = [dict(baseline_mse=float(i + 1), candidate_mse=.9 * (i + 1), half_projection_q=0.0) for i in range(32)]
    a, b = summarize(rows, 'early'), summarize(rows, 'full')
    require(a['accuracy_pass'] and b['accuracy_pass'] and abs(a['pooled_ratio'] - .9) < 1e-14, 'Constant ratio gate')
    require(a['upper_quantile'] == .995 and b['upper_quantile'] == .98, 'Wrong alpha spending')
    require(summarize(rows, 'early')['bootstrap_upper'] == a['bootstrap_upper'], 'Bootstrap nondeterministic')
    tests.append('Exact prescribed deterministic100k bootstrap and both quantiles')
    trap = [dict(baseline_mse=1.0 if i % 2 == 0 else 100.0,
                 candidate_mse=.5 if i % 2 == 0 else 100.5, half_projection_q=0.0) for i in range(32)]
    t = summarize(trap, 'early')
    require(t['pooled_ratio'] == 1 and not t['accuracy_pass'], 'Mean-of-ratios trap')
    tests.append('Pooled ratio rejects misleading mean-of-ratios improvement')
    for bad in (rows[:-1], [dict(r, candidate_mse=float('nan')) for r in rows]):
        try:
            summarize(bad, 'early')
        except ValueError:
            pass
        else:
            raise AssertionError('Missing/nonfinite pairs accepted')
    for bad in (np.zeros((15, 1024)), np.full((16, 1024), np.inf)):
        try:
            prediction_array(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed prediction accepted')
    tests.append('Missing pair, nonfinite metric and prediction shape/finiteness fail closed')
    base, candidate = np.ones((16, 1024)), np.full((16, 1024), 1.2)
    truth = dict(mc_post=np.full((16, 1024), .3), mc_post_A=np.full((16, 1024), .4),
                 mc_post_B=np.full((16, 1024), .2), mc_post_var=np.ones((16, 1024)))
    m = pair_metrics(base, candidate, truth, 15990784)
    require(abs(m['difference'] - .32) < 1e-14 and abs(m['half_projection_q'] + .04) < 1e-14, 'MC cancellation diagnostic')
    require(abs((1.2**2 - 1) - m['difference'] - 2 * .2 * .3) < 1e-14, 'Quadratic MC noise did not cancel')
    tests.append('Shared truth quadratic-noise cancellation and A/B projection formula')
    out = Path(args.out).resolve()
    publish(out, dict(created_utc=utc(), passed=True, source=pointer(__file__), tests=tests,
                      synthetic_only=True, real_seed_data_read=False, model_executed=False))
    print(json.dumps(dict(passed=True, tests=len(tests), output=str(out), real_seed_data_read=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('early')
    p.add_argument('--judge-manifest', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('full')
    p.add_argument('--early-complete', required=True); p.add_argument('--truth-manifest', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('selftest'); p.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command is None:
        require(sha(RUNNER) == RUNNER_SHA, 'Pinned T85 runner changed')
        print(json.dumps(dict(scorer=pointer(__file__), runner=pointer(RUNNER), plan_sha256=PLAN_SHA,
                              protocol_sha256=PROTOCOL_SHA, final_data_read=False, model_executed=False)))
        return
    {'early': early, 'full': full, 'selftest': selftest}[args.command](args)


if __name__ == '__main__':
    main()
