#!/usr/bin/env python3
"""Immutable paired judging through the certified trainer worker, never T79.

prepare is stdlib/source-only and never imports torch or reads generated data.
run-one runs exactly one seed's baseline/candidate pair in one trainer process;
the caller must use the release memory guard and coordinate the compute slot.
No EDBAND request is accepted until its versioned worker is certified.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
O, P2 = HERE.parent, HERE.parent.parent
TRAINER = O / 'trainer'
WORKER = TRAINER / 'worker_generated_v4.py'
WORKER_SHA = '308cc7b08d2291e8e817ac722ae3e7184dc9e95bdc3f13990659a5b958e244a1'
PLAN_SHA = '71f455d1d7f5f7239503468cacd5710f4db9d595c7f1be389c8d937e4f64a979'
PROTOCOL_SHA = 'a80bbcece92976858eaf5d49534437905b3c1263cb02c6426116182133687eca'
PROFILE = O / 'engineer/a22_cutsim/S46_v11_PINNED_PROFILE.json'
PROFILE_ENTRY_SHA = 'bfe53193470738c6c594d338ca4ccb9cf8294f357831e032d62a7dd4006abb96'
FIELDS = set('id mode rank params scale cutsim unfreeze pl attach keep_final keep_all ss f4c ncar hf rg dump15 b6 hsplit hbands b7 featstats cv sched rgw hu lossw truth_plan'.split())
THREAD_KEYS = ('TRAINER_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def source_inventory(profile):
    # Include the card search roots recursively: card modules dynamically import
    # sources outside trainer. Hashes alone are archived; no source is executed.
    roots = [P2 / name for name in ('fable_analyst_v1', 'fable_covhunter_v1',
             'fable_engineer_v1', 'fable_physicist_v1/refit6b', 'fable_porter_v1', 'fable_refit_v1')]
    files = {p.resolve() for root in roots for p in root.rglob('*.py')}
    files.update(p.resolve() for p in (TRAINER / 'frozen').glob('*.py'))
    files.update(TRAINER / name for name in ('worker_generated_v4.py', 'otr_generated_manifest_v1.py', 'otr_engine.py',
                 'otr_common.py', 'otr_m.py', 'otr_pl.py', 'otr_generated_v1.py', 'otr_cut_runtime_v2.py'))
    files.update(HERE / name for name in ('f4b.py', 'b6cols.py', 'b7cols.py', 'pbase_frozen_s4a.py', 'ncar_bank_frozen_s2s.py'))
    files.update((O / 'engineer/a33_starsum').glob('*.py'))
    files.update((O / 'analyst/fresh').glob('*.py'))
    files.add(P2 / 'codex_resume_20260926_v2/hill_climb_pair_rr_inverse_v1/candidate.py')
    files.add(P2 / 'codex_resume_20260919_v1/learnable_joint_basis_implementation_v1/zero.py')
    files.add(P2 / 'astra_phase2_reexamination_v1/current_shared384_causal_p_v1/shared_operator.py')
    files.add(Path(__file__).resolve())
    files.update(Path(profile['module']).parent / name for name in profile['sources'])
    return {str(p): digest(p) for p in sorted(files)}


def point(path, profile, truth_plan=None):
    path = Path(path).resolve()
    request = read(path)
    require(set(request) <= FIELDS, 'Unsupported request fields (including EDBAND): ' + repr(sorted(set(request) - FIELDS)))
    require(not request.get('dump15') and not request.get('featstats'), 'Judge requests must not write diagnostics')
    require(request.get('sched') in (None, '384-320-384'), 'Pinned native push requires 384-320-384')
    require(request.get('cutsim') in (None, profile), 'Request carries a different cutsim profile')
    require(request.get('truth_plan') in (None, truth_plan), 'Request carries different truth plan')
    params = Path(request['params']).resolve()
    require(params.is_file(), 'Missing checkpoint')
    # Normalize only evaluation controls and explicit worker defaults. Preserve
    # every scientific argument, including hu/pl/unfreeze/ncar and head flags.
    normalized = dict(request, params=str(params), mode='eval', cutsim=profile,
                      attach=bool(request.get('attach', False)), f4c=bool(request.get('f4c', True)),
                      keep_all=True, keep_final=False, sched='384-320-384')
    if truth_plan:
        normalized['truth_plan'] = truth_plan
    else:
        normalized.pop('truth_plan', None)
    for key in ('id', 'rank', 'scale', 'lossw'):
        normalized.pop(key, None)
    return dict(source_request=str(path), source_sha256=digest(path), request=normalized,
                request_sha256=identity(normalized), params_sha256=digest(params))


def prepare(args):
    profile = read(PROFILE)
    require(profile['sha256'] == PROFILE_ENTRY_SHA, 'Uncertified entry profile')
    require(profile['opts'] == dict(enabled=True, law_response_fit=True, law_beta=True, law_restore_right=True),
            'All three law flags must be explicit and enabled')
    for name, sha in profile['sources'].items():
        require(digest(Path(profile['module']).parent / name) == sha, 'Changed simulator source: ' + name)
    require(digest(WORKER) == WORKER_SHA, 'Unreviewed truth-plan worker')
    truth_plan = promotion_protocol = promotion_scorer = None
    seeds = [7100104] if args.purpose == 'bridge' else list(range(7100168, 7100200))
    if args.purpose == 'judge':
        require(args.truth_plan and args.promotion_protocol and args.promotion_scorer and args.scorer_sha256,
                'Judge needs explicit truth plan, prospective protocol and reviewed scorer SHA')
        truth_plan = dict(path=str(Path(args.truth_plan).resolve()), sha256=PLAN_SHA)
        promotion_protocol = dict(path=str(Path(args.promotion_protocol).resolve()), sha256=PROTOCOL_SHA)
        promotion_scorer = dict(path=str(Path(args.promotion_scorer).resolve()), sha256=args.scorer_sha256)
        for meta in (truth_plan, promotion_protocol, promotion_scorer):
            require(digest(meta['path']) == meta['sha256'], 'Unreviewed plan/protocol/scorer')
        plan, protocol = read(truth_plan['path']), read(promotion_protocol['path'])
        require(protocol['final_seeds'] == seeds and protocol['plan_sha256'] == PLAN_SHA,
                'Prospective seed/plan contract changed')
        final_entries = [e for e in plan['entries'] if e['role'] == 'final']
        require([e['seed'] for e in final_entries] == seeds, 'Final plan seeds changed')
        require(all(e['N'] == protocol['early']['N'] and e['N_half'] == protocol['early']['N_half']
                    for e in final_entries), 'Early exact N mismatch')
    else:
        require(not any((args.truth_plan, args.promotion_protocol, args.promotion_scorer, args.scorer_sha256)),
                'Used104 bridge retains existing64M legacy adapter; no final plan allowed')
    baseline, candidate = point(args.baseline, profile, truth_plan), point(args.candidate, profile, truth_plan)
    decision = None
    if args.purpose == 'judge':
        require(args.decision_receipt, 'Judge preparation requires an existing fixed-candidate decision receipt')
        receipt = Path(args.decision_receipt).resolve()
        require(receipt.is_file(), 'Missing fixed-candidate decision receipt')
        decision = dict(path=str(receipt), sha256=digest(receipt))
        require(baseline['request_sha256'] != candidate['request_sha256'], 'Judge arms must differ')
    run = Path(args.out).resolve()
    run.relative_to(HERE)
    require(1 <= args.threads <= 4, 'Expected one worker with 1-4 threads')
    sources = source_inventory(profile)
    manifest = dict(schema='physicist_pinned_pair_judge_v2', created_utc=utc(), purpose=args.purpose,
                    root_seed_assignment='136-151 sizing; 152-167 acceptance; 168-199 fixed-candidate judge',
                    seeds=seeds, threads=args.threads, profile=profile, profile_sha256=digest(PROFILE),
                    baseline=baseline, candidate=candidate, decision_receipt=decision,
                    sources=sources, worker=str(WORKER), run=str(run),
                    truth_plan=truth_plan, promotion_protocol=promotion_protocol, promotion_scorer=promotion_scorer)
    manifest['identity_sha256'] = identity(manifest)
    run.mkdir()  # An existing run is never reused or overwritten.
    publish(run / 'manifest.json', manifest)
    for seed in seeds:
        lane = run / ('seed%d' % seed)
        lane.mkdir()
        for folder in ('claimed', 'queue', 'results'):
            (lane / folder).mkdir()
        (lane / 'STOP').open('x').close()  # Only preclaimed one-shot work is served.
        with (lane / 'allowed_seeds.txt').open('x') as stream:
            stream.write(str(seed) + '\n')
        for index, role in enumerate(('baseline', 'candidate')):
            req = dict(manifest[role]['request'], rank=seed, id='t85_%d_%s_r%d' % (index, role, seed))
            publish(lane / 'claimed' / (req['id'] + '.w0.json'), req)
    print(json.dumps(dict(prepared=str(run / 'manifest.json'), identity_sha256=manifest['identity_sha256'],
                          data_read=False, seeds=seeds, source_count=len(sources))))


def validate(path):
    manifest = read(path)
    frozen = dict(manifest)
    expected = frozen.pop('identity_sha256')
    require(identity(frozen) == expected, 'Manifest changed')
    require(manifest['schema'] == 'physicist_pinned_pair_judge_v2', 'Unsupported manifest schema')
    for name, sha in manifest['sources'].items():
        require(digest(name) == sha, 'Source changed since freeze: ' + name)
    require(digest(PROFILE) == manifest['profile_sha256'], 'Profile changed since freeze')
    require(manifest['worker'] == str(WORKER) and digest(WORKER) == WORKER_SHA, 'Worker pin changed')
    for name in ('truth_plan', 'promotion_protocol', 'promotion_scorer'):
        meta = manifest[name]
        if meta:
            require(digest(meta['path']) == meta['sha256'], 'Frozen metadata changed: ' + name)
    if manifest['purpose'] == 'judge':
        require(manifest['seeds'] == list(range(7100168, 7100200)), 'Exact final32 required')
        require(all(manifest[name] for name in ('truth_plan', 'promotion_protocol', 'promotion_scorer')), 'Missing promotion pins')
        require(manifest['truth_plan']['sha256'] == PLAN_SHA and manifest['promotion_protocol']['sha256'] == PROTOCOL_SHA, 'Unreviewed promotion contract')
    for role in ('baseline', 'candidate'):
        p = manifest[role]
        require(digest(p['source_request']) == p['source_sha256'], 'Original request changed: ' + role)
        require(digest(p['request']['params']) == p['params_sha256'], 'Checkpoint changed: ' + role)
        require(p['request']['cutsim'] == manifest['profile'], 'Mismatched arm arithmetic')
        require(p['request'].get('truth_plan') == manifest['truth_plan'], 'Arm truth-plan mismatch')
    if manifest['decision_receipt']:
        p = manifest['decision_receipt']
        require(digest(p['path']) == p['sha256'], 'Fixed-candidate decision changed')
    return manifest


def truth_ready(manifest, seed):
    """Read completion metadata only; never import a neural module or truth array."""
    if not manifest['truth_plan']:
        return True  # The sole legacy bridge seed104 was already validated.
    import importlib.util
    path = TRAINER / 'otr_generated_manifest_v1.py'
    spec = importlib.util.spec_from_file_location('_t85_truth_readiness', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    meta = manifest['truth_plan']
    plan = module.TruthPlan(meta['path'], meta['sha256'], manifest['seeds'])
    # Root's final launch gate: all32 references complete before any estimator pair.
    return all(plan.ready(s) for s in manifest['seeds'])


def ready(args):
    manifest = validate(args.manifest)
    require(args.seed in manifest['seeds'], 'Seed outside frozen allowlist')
    print(json.dumps(dict(seed=args.seed, ready=truth_ready(manifest, args.seed), data_arrays_read=False)))


def run_one(args):
    manifest = validate(args.manifest)
    require(args.seed in manifest['seeds'], 'Seed is outside the frozen manifest allowlist')
    require(truth_ready(manifest, args.seed), 'Full32 truth/completion batch not ready; no STARTED/results written; retry exact pair later')
    lane = Path(manifest['run']) / ('seed%d' % args.seed)
    require((lane / 'STOP').exists() and not list((lane / 'queue').iterdir()), 'One-shot run contract changed')
    require(not list((lane / 'results').iterdir()), 'Existing results/errors: archive this attempt and prepare a new version')
    claims = sorted((lane / 'claimed').glob('*.w0.json'))
    require(len(claims) == 2, 'Expected one baseline and one candidate')
    for index, role in enumerate(('baseline', 'candidate')):
        expected = dict(manifest[role]['request'], rank=args.seed, id='t85_%d_%s_r%d' % (index, role, args.seed))
        require(read(claims[index]) == expected, 'Claim does not match frozen arm')
    require((lane / 'allowed_seeds.txt').read_text().strip() == str(args.seed), 'Allowlist changed')
    # Discard inherited trainer knobs so two invocations have the same defaults.
    for key in list(os.environ):
        if key.startswith(('TRAINER_', 'NCAR6_')):
            del os.environ[key]
    os.environ.update({key: str(manifest['threads']) for key in THREAD_KEYS})
    os.environ.update(TRAINER_PL='1', TRAINER_CARD='M', TRAINER_F4C='1', TRAINER_STAR_ATTACH='0',
                      TRAINER_GENERATED_ALLOW_FILE=str(lane / 'allowed_seeds.txt'), TRAINER_RELIEF='1')
    if manifest['truth_plan']:
        os.environ.update(TRAINER_TRUTH_PLAN=manifest['truth_plan']['path'],
                          TRAINER_TRUTH_PLAN_SHA256=manifest['truth_plan']['sha256'])
    sys.path = [str(TRAINER)] + [p for p in sys.path if Path(p or os.getcwd()).resolve() != HERE]
    sys.argv = [str(WORKER), str(lane), '0']
    publish(lane / 'STARTED.json', dict(utc=utc(), manifest_sha256=digest(args.manifest), pid=os.getpid()))
    with (lane / 'worker.log').open('x') as log, redirect_stdout(log), redirect_stderr(log):
        runpy.run_path(str(WORKER), run_name='__main__')
    records = []
    for claim in claims:
        req = read(claim)
        result = lane / 'results' / (req['id'] + '.json')
        require(result.is_file(), 'Worker failed: inspect immutable result/error files')
        rec = read(result)
        require(rec['rank'] == args.seed and rec['mode'] == 'eval' and rec['cutsim'] == manifest['profile'], 'Result identity mismatch')
        require(rec['threads'] == manifest['threads'] and rec['params'] == req['params'], 'Worker configuration mismatch')
        require(math.isfinite(rec['mse']) and rec['mse'] > 0, 'Nonfinite result')
        require(rec['truth_plan'] == manifest['truth_plan'], 'Worker result truth plan mismatch')
        prediction = rec['prediction']
        require(len(prediction) == 16 and all(len(row) == 1024 for row in prediction), 'Prediction shape changed')
        require(all(math.isfinite(v) for row in prediction for v in row), 'Nonfinite prediction')
        if manifest['truth_plan']:
            metadata = rec['truth_metadata']
            require(metadata and metadata['plan'] == manifest['truth_plan'] and metadata['seed'] == args.seed
                    and metadata['role'] == 'final', 'Wrong truth identity/role')
            entry = next(e for e in read(manifest['truth_plan']['path'])['entries'] if e['seed'] == args.seed)
            require(all(metadata[k] == entry[k] for k in ('path', 'N', 'N_half')), 'Exact truth metadata mismatch')
            require(digest(metadata['path']) == metadata['sha256'], 'Truth changed during pair')
            completion = metadata['completion_receipt']
            require(completion and digest(completion['path']) == completion['sha256'], 'Truth receipt changed')
        records.append(dict(id=req['id'], path=str(result), sha256=digest(result), mse=rec['mse'],
                            prediction_sha256=identity(prediction), truth_metadata=rec['truth_metadata']))
    require(records[0]['truth_metadata'] == records[1]['truth_metadata'], 'Arms used different truth')
    if manifest['purpose'] == 'bridge' and manifest['baseline']['request_sha256'] == manifest['candidate']['request_sha256']:
        first, second = (read(r['path']) for r in records)
        require(first['mse_layers'] == second['mse_layers'], 'Identical-point bridge has state carry or nondeterminism')
        require(first['prediction'] == second['prediction'], 'Identical-point predictions differ')
    validate(args.manifest)  # Fail if any source/parameter changed during the pair.
    publish(lane / 'COMPLETE.json', dict(utc=utc(), manifest_identity=manifest['identity_sha256'], records=records,
                                       relative_change=records[1]['mse'] / records[0]['mse'] - 1))
    print(json.dumps(dict(completed_seed=args.seed, numeric_results_sealed_until_full_batch=True)))


def summarize(args):
    manifest = validate(args.manifest)
    require(all((Path(manifest['run']) / ('seed%d' % seed) / 'COMPLETE.json').is_file() for seed in manifest['seeds']),
            'Whole batch must be complete before loading any scores')
    require(manifest['purpose'] == 'bridge', 'Judge promotion requires the pinned pooled-MSE two-look scorer; this summary is bridge-only')
    rows = []
    for seed in manifest['seeds']:
        receipt = read(Path(manifest['run']) / ('seed%d' % seed) / 'COMPLETE.json')
        require(receipt['manifest_identity'] == manifest['identity_sha256'], 'Mixed manifest results')
        for rec in receipt['records']:
            require(digest(rec['path']) == rec['sha256'], 'Result changed')
        rows.append(dict(seed=seed, relative_change=receipt['relative_change']))
    values = [r['relative_change'] for r in rows]
    print(json.dumps(dict(manifest_identity=manifest['identity_sha256'], purpose=manifest['purpose'], n=len(values),
                          mean_relative_change=statistics.mean(values),
                          se_relative_change=statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None,
                          improved=sum(v < 0 for v in values), rows=rows), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--baseline', required=True); p.add_argument('--candidate', required=True)
    p.add_argument('--out', required=True); p.add_argument('--purpose', choices=('bridge', 'judge'), required=True)
    p.add_argument('--decision-receipt'); p.add_argument('--threads', type=int, default=3)
    p.add_argument('--truth-plan'); p.add_argument('--promotion-protocol')
    p.add_argument('--promotion-scorer'); p.add_argument('--scorer-sha256')
    p = sub.add_parser('run-one'); p.add_argument('manifest'); p.add_argument('seed', type=int)
    p = sub.add_parser('summarize'); p.add_argument('manifest')
    p = sub.add_parser('ready'); p.add_argument('manifest'); p.add_argument('seed', type=int)
    args = parser.parse_args()
    {'prepare': prepare, 'run-one': run_one, 'summarize': summarize, 'ready': ready}[args.command](args)


if __name__ == '__main__':
    main()
