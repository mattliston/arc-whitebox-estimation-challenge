"""Public-only board refresh; immutable snapshots; no credentials or forum.

The old merged leaderboard crashes on graded records without ARC aggregates
and on teams whose first submission is after the requested cutoff. Preserve
those raw records but classify them as unverified instead of ranking them.
Usage: python -B a11_rescan_astra_v2.py [--scan]
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

A = Path(__file__).resolve().parent
O = A.parent


def utc():
    return subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def load():
    rows = {}
    for p in [O / 'board_evals/scan_332296_333380.jsonl'] + sorted(A.glob('scan_3*.jsonl')):
        for line in p.read_text().splitlines():
            r = json.loads(line)
            old = rows.get(r['id'], {})
            if old.get('status') == 'GRADED' and r.get('status') != 'GRADED':
                continue
            rows[r['id']] = r
    return rows


def eligibility(r, cutoff):
    if r.get('http') != 200 or r.get('status') != 'GRADED':
        return 'not_graded'
    if not r.get('created') or r['created'] > cutoff:
        return 'after_cutoff'
    agg = r.get('agg') or {}
    if not all(k in agg for k in ('final_layer_mse', 'all_layers_mse', 'mean_compute_utilization', 'n_failed_mlps')):
        return 'missing_arc_aggregates'
    if agg['n_failed_mlps'] != 0:  # Public leaderboard ignores private-only failures.
        return 'failed_public_networks'
    pubs = r.get('pub') or []
    if len(pubs) != 50 or len({p[0] for p in pubs}) != 50:
        return 'incomplete_public50'
    score = r.get('score')
    if not isinstance(score, (int, float)) or not math.isfinite(score) or score <= 0:
        return 'invalid_score'
    return None


def board(rows, cutoff):
    best, excluded = {}, []
    for r in rows.values():
        why = eligibility(r, cutoff)
        if why:
            if r.get('status') == 'GRADED':
                excluded.append(dict(id=r['id'], reason=why))
            continue
        team = r.get('team') or r.get('who')
        if not team:
            excluded.append(dict(id=r['id'], reason='missing_team'))
            continue
        if team not in best or r['score'] < best[team]['score']:
            best[team] = dict(team=team, id=r['id'], score=r['score'], created=r['created'], known_all_network_failures=len(r.get('failed') or []))
    ranked = sorted(best.values(), key=lambda r: r['score'])
    return [dict(place=i + 1, **r) for i, r in enumerate(ranked)], excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', action='store_true')
    args = ap.parse_args()
    started = utc()
    stamp = started.replace('-', '').replace(':', '')
    home = A / ('astra_board_' + stamp)
    home.mkdir(exist_ok=False)
    rows = load()
    if args.scan:
        last = max(i for i, r in rows.items() if r.get('http') == 200)
        pending = sorted(i for i, r in rows.items() if r.get('http') == 200 and
                         r.get('status') == 'SUBMITTED' and i > last - 60)
        out = A / ('scan_%d_plus_astra_%s.jsonl' % (last + 1, stamp))
        assert not out.exists()
        with (home / 'scan.log').open('x') as f:
            subprocess.run([sys.executable, str(A / 'a5_scan_driver.py'), str(last + 1),
                            str(out), '6', ','.join(map(str, pending))],
                           check=True, stdout=f, stderr=subprocess.STDOUT)
        rows = load()
    cutoff = utc()
    ranked, excluded = board(rows, cutoff)
    report = dict(started=started, completed=cutoff, live_scan=args.scan,
                  last_http200=max(i for i, r in rows.items() if r.get('http') == 200),
                  top12=ranked[:12], excluded_graded=excluded)
    with (home / 'report.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps(dict(report=str(home / 'report.json'), top6=ranked[:6],
                          excluded_graded=len(excluded))))


if __name__ == '__main__':
    main()
