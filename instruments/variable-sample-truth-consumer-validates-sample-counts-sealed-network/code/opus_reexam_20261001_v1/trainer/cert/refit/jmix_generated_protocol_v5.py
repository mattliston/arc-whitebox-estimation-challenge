"""Torch-free, fresh-launch preflight for generated refit sizing and checking.

Run from O/trainer with --gradient-ranks and --gradient-selection; exactly
one prescribed 96-network public batch, first-batch 0, steps 1, no reuse.
TRAINER_GENERATED_ALLOW_FILE must cover sizing/check and exclude judge32.
The rank file is recomputed from metadata-only eligibility and pinned holdout
sources. Historical gradient/results values never participate in selection.

This protocol deliberately has no resume bypass. A successful manifest is
created exclusively, and any prior trainer use of sizing/check seeds blocks
a fresh launch. Includes run_*/ and cert/**/ worker record directories only.
All sizing truths must exist at launch. Check truths may still be pending;
workers leave their requests queued until truth exists, and the driver waits
for every baseline/candidate check result before an acceptance decision.
Truth contents and gradient tensors are never read here.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path
import subprocess


_V1_PATH = Path(__file__).with_name('jmix_protocol_v1.py')
_SPEC = importlib.util.spec_from_file_location('jmix_generated_preflight_parent_v1', _V1_PATH)
_V1 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_V1)
TRUTH_DIR = Path(__file__).resolve().parents[3] / 'analyst/fresh'
GENERATED_FLOOR = 7_000_000


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def fingerprint(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return {'path': str(path), 'sha256': digest.hexdigest()}


def record_directories():
    """Only worker record directories: run_*/... and cert/**/..., deduplicated."""
    roots = set()
    for section in ('queue', 'claimed', 'results'):
        paths = list(Path('.').glob('run_*/' + section)) + list(Path('cert').rglob(section))
        roots.update(path.resolve() for path in paths if path.is_dir())
    return sorted(roots)


def prior_trainer_uses(seeds):
    """Inspect pool and certificate records, never parameter/head/truth files."""
    found, inspected, seen = [], 0, set()
    for root in record_directories():
        for path in sorted(root.glob('*')):
            if not path.is_file() or path.suffix not in ('.json', '.pt', '.error'):
                continue
            canonical = path.resolve()
            if canonical in seen:
                continue
            seen.add(canonical)
            inspected += 1
            rank = _V1.file_rank(path)
            ranks = set() if rank is None else {rank}
            if path.suffix == '.json':
                record = json.loads(path.read_text())
                require(isinstance(record, dict), 'Unrecognized trainer record: %s' % path)
                if 'rank' in record:
                    ranks.add(int(record['rank']))
            require(ranks, 'Cannot identify trainer record network: %s' % path)
            common = ranks & seeds
            if common:
                found.append({'path': str(path), 'ranks': sorted(common)})
    return found, inspected


def verify_reused_gradient(args, expected_request):
    if not args.gradient_label:
        return None
    require(args.params and args.head, 'Reused gradients require --params and --head')
    params = Path(args.params).resolve()
    require(params.is_file(), 'Missing start parameters: %s' % params)
    head = json.loads(Path(args.head).read_text())
    checked = []
    for result in sorted(Path('run_M/results').glob('a_%s_r*.pt' % args.gradient_label)):
        rank = _V1.file_rank(result)
        claims = sorted(Path('run_M/claimed').glob(result.stem + '.w*.json'))
        require(claims, 'Missing saved request for reused gradient: %s' % result)
        for claim in claims:
            request = json.loads(claim.read_text())
            require(request.get('id') == result.stem and request.get('rank') == rank
                    and request.get('mode') == 'grad', 'Gradient request identity mismatch: %s' % claim)
            require(Path(request.get('params', '')).resolve() == params,
                    'Gradient parameter path mismatch: %s' % claim)
            require(request.get('hu') == head, 'Gradient head weights mismatch: %s' % claim)
            require(request.get('cutsim') == expected_request.get('cutsim'),
                    'Gradient simulator mismatch: %s' % claim)
            for key, value in expected_request.items():
                require(request.get(key) == value, 'Gradient request %s mismatch: %s' % (key, claim))
            checked.append(fingerprint(claim))
    return {'label': args.gradient_label, 'checked_claims': checked}



# Preserve the original predeclared pool exclusion from jmix21, not outcome selection.
LEGACY_USED36 = (597, 165, 667, 25, 309, 65, 642, 125, 164, 288, 297, 330, 124, 662, 603, 50,
                 182, 323, 381, 90, 201, 208, 17, 563, 421, 154, 85, 594, 411, 115, 605, 678, 384, 647, 91, 60)
_COMMON_PATH = Path(__file__).resolve().parents[2] / 'otr_common.py'
_COMMON_SPEC = importlib.util.spec_from_file_location('jmix_first96_common', _COMMON_PATH)
_COMMON = importlib.util.module_from_spec(_COMMON_SPEC)
_COMMON_SPEC.loader.exec_module(_COMMON)


def selection_plan(args):
    """Deterministic metadata-only prescription, before observing any new gradient."""
    core = [Path('cert/ls16_ranks.txt'), Path('cert/runB_val.txt'), Path('cert/scan16_ranks.txt')]
    core += sorted(Path('cert').glob('run*_val*.txt'))
    core += sorted(Path('cert/refit').glob('sz*_ranks.txt'))
    configs = []
    for path in sorted(Path('.').glob('run_*/config.json')):
        config = json.loads(path.read_text())
        configs.append(fingerprint(path))
        if config.get('val_file'):
            core.append(Path(config['val_file']))
    diagnostics = [Path('cert/fresh60_ranks.txt'), Path('cert/fresh_at36_ranks.txt'), Path('cert/train_a_ranks.txt')]
    generated = [Path(args.sizing), Path(args.check), Path(args.judge)]
    roles = {}
    for role, paths in (('public_holdout', core), ('conservative_historical_evaluation_exclusion', diagnostics),
                        ('current_generated_holdout', generated)):
        for path in paths:
            roles.setdefault(path.resolve(), set()).add(role)
    holds = []
    for path, role in sorted(roles.items(), key=lambda item: str(item[0])):
        ranks = sorted(_V1.read_ranks(path))
        require(ranks, 'Empty hold/exclusion file: %s' % path)
        if 'current_generated_holdout' not in role:
            require(all(0 <= rank < GENERATED_FLOOR for rank in ranks), 'Non-public rank in public hold file: %s' % path)
        holds.append(dict(fingerprint(path), roles=sorted(role), ranks=ranks))
    split = json.loads(_COMMON.SPLIT.read_text())
    train = _COMMON.splits()['train']
    held = set(split['valid40']) | set(split['panel8'])
    for item in holds:
        held.update(item['ranks'])
    eligible = [rank for rank in train if rank not in held and rank not in LEGACY_USED36]
    require(len(eligible) == len(set(eligible)) and len(eligible) >= 96, 'Fewer than 96 unique eligible public training ranks')
    shuffled = [rank for rank in train if rank not in LEGACY_USED36]
    random.Random(20261010).shuffle(shuffled)
    chosen = [rank for rank in shuffled if rank not in held][:96]
    require(len(set(chosen)) == 96 and all(0 <= rank < GENERATED_FLOOR for rank in chosen), 'Invalid 96-rank prescription')
    return dict(schema='astra_cut_first96_selection_v1', seed=20261010,
                algorithm='preserve train split order; exclude legacy used36; Random(seed).shuffle; filter all listed holds/evaluation lists; first96',
                source_files={'prescription_protocol': fingerprint(__file__), 'common': fingerprint(_COMMON_PATH), 'split': fingerprint(_COMMON.SPLIT), 'run_configs': configs},
                split_hold_roles={'valid40': split['valid40'], 'panel8': split['panel8']},
                hold_files=holds, legacy_used36=list(LEGACY_USED36), eligible_count=len(eligible),
                eligible_order=eligible, selected_order=chosen,
                selection_uses_gradient_or_result_values=False,
                caveat='Fresh60/fresh_at36/train_a are historical training diagnostics conservatively excluded, not relabeled permanent holds.')


def prescribed_gradients(args):
    require(args.steps == 1 and args.first_batch == 0 and not args.gradient_label,
            'First-point protocol requires steps=1, first-batch=0 and no reused gradient label')
    saved = json.loads(Path(args.gradient_selection).read_text())
    expected = selection_plan(args)
    require(set(saved) == set(expected) | {'utc', 'rank_file'}, 'Selection manifest fields changed')
    for key, value in expected.items():
        require(saved[key] == value, 'Selection manifest no longer matches metadata prescription: %s' % key)
    require(saved['rank_file'] == fingerprint(args.gradient_ranks), 'Gradient rank file hash/path changed')
    ordered = [int(x) for x in Path(args.gradient_ranks).read_text().strip().split(',')]
    require(len(ordered) == len(set(ordered)) == 96, 'Exactly 96 unique gradient ranks required')
    require(ordered == expected['selected_order'], 'Gradient rank order differs from prescribed selection')
    return ordered, saved


def first_point_plan(args, gradient_ranks):
    """The v1 holdout/history audit, adapted to one real exact96 batch."""
    sizing, check = _V1.read_ranks(args.sizing), _V1.read_ranks(args.check)
    judge = _V1.read_ranks('cert/runM_val50.txt')
    require(len(sizing) == len(check) == 16 and len(judge) == 50, 'Invalid sizing/check/judge50 cardinalities')
    sets = {'sizing': sizing, 'check': check, 'judge50': judge}
    for name, ranks in sets.items():
        for other, others in sets.items():
            if name < other:
                require(not ranks & others, 'Holdout sets overlap: %s/%s' % (name, other))
        require(not set(gradient_ranks) & ranks, 'Gradient ranks overlap %s' % name)
    previous_gradients = {}
    for result_dir in Path('.').glob('run_*/results'):
        for result in result_dir.glob('*.pt'):
            rank = _V1.file_rank(result)
            if rank is not None:
                previous_gradients[rank] = previous_gradients.get(rank, 0) + 1
    history = {}
    files = sorted(Path('cert/refit').glob('sz*_ranks.txt')) + [Path('cert/ls16_ranks.txt')]
    for name in ('sizing', 'check'):
        chosen = sets[name]
        history[name] = {'prior_gradient_result_count_by_rank': {str(rank): previous_gradients.get(rank, 0) for rank in sorted(chosen)},
                         'selection_file_overlaps': {str(path): sorted(chosen & _V1.read_ranks(path))
                                                     for path in files if chosen & _V1.read_ranks(path)}}
    return {'sets': {name: sorted(ranks) for name, ranks in sets.items()},
            'planned_gradients': [{'source': 'prescribed_first96', 'ranks': list(gradient_ranks)}], 'history': history}

def audit(args):
    gradient_ranks, selection = prescribed_gradients(args)
    result = first_point_plan(args, gradient_ranks)
    sizing, check = set(result['sets']['sizing']), set(result['sets']['check'])
    judge = _V1.read_ranks(args.judge)
    require(len(judge) == 32, '32 reserved generated judging seeds required')
    generated_sets = {'sizing': sizing, 'check': check, 'generated_judge32': judge}
    for name, seeds in generated_sets.items():
        require(all(seed >= GENERATED_FLOOR for seed in seeds), '%s must contain generated seeds only' % name)
    require(not judge & (sizing | check | set(result['sets']['judge50'])),
            'Generated judging seeds overlap sizing/check/judge50')
    for gradient in result['planned_gradients']:
        require(all(0 <= rank < GENERATED_FLOOR for rank in gradient['ranks']),
                'Reused/planned gradients must contain public ranks only')
        require(not judge & set(gradient['ranks']), 'Generated judge overlaps planned gradients')
    allow_file = os.environ.get('TRAINER_GENERATED_ALLOW_FILE')
    require(allow_file, 'TRAINER_GENERATED_ALLOW_FILE is required')
    allowed = _V1.read_ranks(allow_file)
    require(all(seed >= GENERATED_FLOOR for seed in allowed), 'Worker allowlist must contain generated seeds only')
    require((sizing | check) <= allowed, 'Worker allowlist does not cover sizing/check')
    require(not judge & allowed, 'Worker allowlist exposes reserved generated judging seeds')
    uses, inspected = prior_trainer_uses(sizing | check)
    require(not uses, 'Sizing/check seeds already have trainer requests/results: %s' % json.dumps(uses))
    truth_readiness = {}
    for role, seeds in (('sizing', sizing), ('check', check)):
        ready, missing = [], []
        for seed in sorted(seeds):
            (ready if (TRUTH_DIR / ('truth_seed%d.npz' % seed)).is_file() else missing).append(seed)
        truth_readiness[role] = {'ready': ready, 'missing': missing, 'required_at_launch': role == 'sizing'}
    require(not truth_readiness['sizing']['missing'],
            'Sizing truths not ready: %s' % truth_readiness['sizing']['missing'])
    expected_request = json.loads(os.environ.get('JM_REQ') or '{}')
    require(isinstance(expected_request, dict), 'JM_REQ must be a JSON object')
    reused = verify_reused_gradient(args, expected_request)
    result['sets']['generated_judge32'] = sorted(judge)
    result.update(
        utc=subprocess.check_output(['date', '-u', '+%FT%TZ'], text=True).strip(),
        protocol='jmix_generated_protocol_v5',
        files={name: fingerprint(path) for name, path in {
            'sizing': args.sizing, 'check': args.check, 'generated_judge32': args.judge,
            'gradient_ranks': args.gradient_ranks, 'gradient_selection': args.gradient_selection,
            'worker_allowlist': allow_file, 'protocol_v1': _V1_PATH, 'protocol_v5': __file__,
            **({'params': args.params} if args.params else {}),
            **({'head': args.head} if args.head else {}),
        }.items()},
        expected_request=expected_request,
        reused_gradient=reused, prescribed_gradient_count=96, gradient_selection=selection,
        fresh_launch={'trainer_records_inspected': inspected, 'record_directories': [str(path) for path in record_directories()],
                      'prior_sizing_check_uses': uses,
                      'truth_readiness': truth_readiness,
                      'truth_readiness_basis': 'file existence only; no truth contents read',
                      'sizing_check_truth_files_exist': not truth_readiness['check']['missing'],
                      'check_truths_pending_permitted': True, 'resume_permitted': False},
        verdict=('PASS: fresh generated sizing16/check16; sizing truths ready; %d check truths pending; '
                 'reserved judge32 excluded; exactly96 prescribed public gradients; first point only; allowlist safe' % len(truth_readiness['check']['missing'])),
        caveat=('Freshness audit covers run_* and cert/** worker queues/claims/results. Lead seed reservations and outside-lane '
                'usage audit remain authoritative. Pending check truths are allowed at launch: the qualified worker '
                'skips missing-truth check requests, and exact baseline/candidate check results are required before '
                'any acceptance decision. Reused check16 is selection data; public judge50 is '
                'secondary. A frozen candidate still requires independent judging on reserved generated seeds.'),
    )
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizing', required=True)
    parser.add_argument('--check', required=True)
    parser.add_argument('--judge', required=True)
    parser.add_argument('--gradient-ranks', required=True)
    parser.add_argument('--gradient-selection', required=True)
    parser.add_argument('--first-batch', required=True, type=int)
    parser.add_argument('--steps', required=True, type=int)
    parser.add_argument('--gradient-label', default='')
    parser.add_argument('--params')
    parser.add_argument('--head')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    require(not Path(args.out).exists(), 'Protocol manifest already exists: %s' % args.out)
    result = audit(args)
    with Path(args.out).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(result['verdict'])
    print('Protocol and history:', args.out)
