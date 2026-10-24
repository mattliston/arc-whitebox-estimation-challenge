"""Torch-free preflight for independent refit sizing/checking; preserves prior work.

Run from O/trainer. Records previous gradient usage and prior sizing/check files;
fresh means no observed use, not a promise inferred from a filename.
"""
import argparse
import json
from pathlib import Path
import re
import subprocess


def read_ranks(path):
    values = [int(x) for x in Path(path).read_text().strip().split(',')]
    assert len(values) == len(set(values)), (path, 'duplicate ranks')
    return set(values)


def file_rank(path):
    match = re.search(r'_r([0-9]+)', path.name)
    return int(match.group(1)) if match else None


def audit(args):
    sizing, check = read_ranks(args.sizing), read_ranks(args.check)
    judge = read_ranks('cert/runM_val50.txt')
    assert len(sizing) == len(check) == 16, '16 networks required in each set'
    sets = {'sizing': sizing, 'check': check, 'judge50': judge}
    for name, ranks in sets.items():
        for other, others in sets.items():
            if name < other:
                assert not ranks & others, (name, other, sorted(ranks & others))
    batches = [{int(x) for x in batch.split(',')} for batch in args.batches.split()]
    assert len(batches) == 12
    gradients = []
    for step in range(args.steps):
        if step == 0 and args.gradient_label:
            files = sorted(Path('run_M/results').glob('a_%s_r*.pt' % args.gradient_label))
            ranks = {file_rank(path) for path in files}
            assert len(files) == len(ranks) == 36, 'reused first gradient must have 36 results'
            source = args.gradient_label
        else:
            index = (args.first_batch + step) % len(batches)
            ranks, source = batches[index], 'batch%d' % index
        assert ranks and None not in ranks
        for name, held in sets.items():
            assert not ranks & held, (source, name, sorted(ranks & held))
        gradients.append({'source': source, 'ranks': sorted(ranks)})
    previous_gradients = {}
    for result_dir in Path('.').glob('run_*/results'):
        for result in result_dir.glob('*.pt'):
            rank = file_rank(result)
            if rank is not None:
                previous_gradients[rank] = previous_gradients.get(rank, 0) + 1
    history = {}
    selection_files = sorted(Path('cert/refit').glob('sz*_ranks.txt')) + [Path('cert/ls16_ranks.txt')]
    for name in ('sizing', 'check'):
        chosen = sets[name]
        files = {}
        for path in selection_files:
            common = chosen & read_ranks(path)
            if common:
                files[str(path)] = sorted(common)
        history[name] = {
            'prior_gradient_result_count_by_rank': {str(rank): previous_gradients.get(rank, 0) for rank in sorted(chosen)},
            'selection_file_overlaps': files,
        }
    return {
        'utc': subprocess.check_output(['date', '-u', '+%FT%TZ'], text=True).strip(),
        'sets': {name: sorted(ranks) for name, ranks in sets.items()},
        'planned_gradients': gradients,
        'history': history,
        'verdict': 'PASS: sizing/check/judge are disjoint; every planned gradient is disjoint from all three',
        'caveat': 'Historical rank use is disclosed. The repeatedly selected judge50 is secondary; untouched generated seeds are primary.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizing', required=True)
    parser.add_argument('--check', required=True)
    parser.add_argument('--batches', required=True)
    parser.add_argument('--first-batch', required=True, type=int)
    parser.add_argument('--steps', required=True, type=int)
    parser.add_argument('--gradient-label', default='')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = audit(args)
    with Path(args.out).open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(result['verdict'])
    print('Protocol and history:', args.out)
