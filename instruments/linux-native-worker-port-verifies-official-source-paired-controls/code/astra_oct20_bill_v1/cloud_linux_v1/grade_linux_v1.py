"""Paired cross-half scoring on Linux; same float64 arithmetic as grade_store_v1.

--mac-reference is a local offline validation mode only. It permits the existing
Mac receipt schema; it cannot qualify cloud output against a Mac denominator.
"""
import argparse
import json
import math
from pathlib import Path
import re
import statistics
import numpy as np
from common_linux_v1 import load_plan, sha, utc, verified, write_new


def collect(root, mac=False, fresh=False):
    rows = {}; sources = set(); modes = set()
    if (root / 'rows.jsonl').exists() and (root / 'manifest.json').exists():
        paths = [root / 'rows.jsonl']
    else:
        pattern = r'candidate_part\d+' if mac else r'part\d+'
        paths = sorted(p for p in root.glob('*/rows.jsonl') if re.fullmatch(pattern, p.parent.name))
    assert paths, 'no canonical worker groups found: ' + str(root)
    for path in paths:
        folder = path.parent; manifest = json.loads((folder / 'manifest.json').read_text())
        if not mac:
            guard = json.loads((folder.parent / (folder.name + '_GUARD.json')).read_text())
            launcher = json.loads((folder / 'launcher.json').read_text())
            assert guard['successful'] and guard['whole_group_retired'] and guard['child_exit'] == 0
            assert launcher['client_exit'] == 0 and launcher['error'] is None
            assert manifest['environment']['sdk_source_hashes_verified']
            assert manifest['warmup_each'] == fresh, 'warmup mode must match frozen fixture plan'
            modes.add(manifest['setup_mode'])
        else:
            modes.add('mac_reference_only')
        sources.add(manifest['sha256'])
        group = manifest['ranks']
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert [r['rank'] for r in records] == group, ('incomplete or reordered group', path)
        for row in records:
            rank = int(row['rank']); assert rank not in rows, ('duplicate', rank)
            pred = folder / f'pred_{rank}.bin'
            assert pred.stat().st_size == 65536
            if not mac:
                assert sha(pred) == row['prediction_sha256']
            assert row['nonfinite'] == 0
            value = np.fromfile(pred, dtype='<f4').reshape(16, 1024).astype(np.float64)
            assert np.isfinite(value).all()
            rows[rank] = dict(row=row, prediction=value, prediction_sha256=sha(pred), group=group)
    assert len(sources) == 1 and len(modes) == 1, (sources, modes)
    return rows, next(iter(sources)), next(iter(modes))


def summary(rows):
    if not rows:
        return dict(n=0)
    result = dict(n=len(rows), lower=int(sum(r['adjusted_pct'] < 0 for r in rows)))
    for key in ('raw_pct', 'bill_pct', 'adjusted_pct'):
        values = [r[key] for r in rows]
        result[key] = statistics.mean(values)
        result[key + '_se'] = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return result


def main():
    p = argparse.ArgumentParser()
    for key in ('root', 'plan', 'candidate', 'bank', 'out'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--ranks', help='explicit subset for bridge pilot; default all32')
    p.add_argument('--mac-reference', action='store_true')
    a = p.parse_args(); plan = load_plan(a.root, a.plan)
    ranks = [int(r) for r in a.ranks.split(',')] if a.ranks else [r for g in plan['groups'] for r in g]
    fresh = plan.get('fixture_mode') == 'fresh_selected'
    assert not (fresh and a.mac_reference)
    candidate, cs, cmode = collect(a.candidate, a.mac_reference, fresh)
    bank, bs, bmode = collect(a.bank, a.mac_reference, fresh)
    assert cmode == bmode and set(ranks) == set(candidate) == set(bank), 'matched complete ranks required'
    if not a.mac_reference:
        assert bs == plan['candidates']['bank337393']['effective_sha256'], 'Linux bank337393 denominator required'
    result = []
    for rank in ranks:
        c, b = candidate[rank], bank[rank]
        assert c['group'] == b['group'], 'same original cold/warm order required'
        truths = []
        for half in ('A', 'B'):
            rec = plan['rows'][str(rank)]['truth_' + half]
            path = verified(a.root, rec)
            with np.load(path, allow_pickle=False) as data:
                truth = data[rec['key']].astype(np.float64)
            assert truth.shape == (16, 1024) and np.isfinite(truth).all()
            truths.append(truth)
        A, B = truths
        u1 = np.mean((c['prediction'] - A) * (c['prediction'] - B), axis=1)
        u0 = np.mean((b['prediction'] - A) * (b['prediction'] - B), axis=1)
        assert u0[15] > 0 and u1[15] > 0
        fr = c['row']['flops'] / b['row']['flops']
        result.append(dict(rank=rank, raw_pct=float(100 * (u1[15] / u0[15] - 1)), bill_pct=100 * (fr - 1),
                           adjusted_pct=float(100 * (u1[15] / u0[15] * fr - 1)), u1=float(u1[15]), u0=float(u0[15]),
                           flops=c['row']['flops'], bank_flops=b['row']['flops'],
                           requests=c['row']['requests'], bank_requests=b['row']['requests'],
                           prediction_sha256=c['prediction_sha256'], bank_prediction_sha256=b['prediction_sha256'],
                           max_abs_dy=float(np.max(np.abs(c['prediction'] - b['prediction']))),
                           layer_raw_pct=[100 * (float(x) / float(y) - 1) if y else None for x, y in zip(u1, u0)]))
    report = dict(utc=utc(), reference='paired same-host bank337393', candidate_sha256=cs, bank_sha256=bs,
                  plan_sha256=sha(a.plan), mac_reference_only=a.mac_reference, setup_mode=cmode,
                  all=summary(result), first7=summary([r for r in result if r['rank'] in plan['first7']]),
                  held25=summary([r for r in result if r['rank'] in plan['held25']]), rows=result,
                  release_qualified=False)
    write_new(a.out, report)
    print(json.dumps({key: report[key] for key in ('all', 'first7', 'held25', 'mac_reference_only')}))


if __name__ == '__main__':
    main()
