#!/usr/bin/env python3
"""Read-only smoke/screen/confirmation grading. No imports of the model/runtime."""
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
O = HERE.parents[2]


def read(run, arm, seed):
    p = HERE / run / 'results' / f'k4metric_{arm}_r{seed}.json'
    return json.loads(p.read_text()) if p.exists() else None


def changes(arm, confirmation=False):
    result = []
    seeds = range(7100592, 7100608) if confirmation else range(7100632, 7100648)
    for seed in seeds:
        a = read('run_confirm' if confirmation else 'run_screen', arm, seed)
        if confirmation:
            p = O / 'physicist/t141_k4l_ext/results' / f'ph141_r{seed}_c.json'
            b = json.loads(p.read_text()) if p.exists() else None
        else:
            b = read('run_screen', 'z', seed)
        if a is None or b is None:
            continue
        assert a['rank'] == b['rank'] == seed
        assert a['mse'] > 0 and b['mse'] > 0
        d = math.log(a['mse'] / b['mse'])
        assert math.isfinite(d)
        result.append(d)
    return result


def stats(values):
    n = len(values)
    return dict(n=n, mean_log_pct=100 * statistics.mean(values) if n else None,
                se_pct=100 * statistics.stdev(values) / math.sqrt(n) if n > 1 else None,
                lower=sum(x < 0 for x in values))


def smoke():
    a = read('run_smoke', 'z', 7100584)
    b = json.loads((O / 'physicist/t141_k4l_ext/results/ph141_r7100584_c.json').read_text())
    fields = ('mse', 'mse_layers', 'final_prediction')
    checks = {key: a[key] == b[key] for key in fields} if a else {}
    return dict(complete=a is not None, checks=checks,
                bitexact=bool(checks) and all(checks.values()))


def screen():
    arms = {arm: stats(changes(arm)) for arm in ('h', 'f')}
    complete = all(row['n'] == 16 for row in arms.values())
    smoke_ok = smoke()['bitexact']
    passing = [arm for arm, row in arms.items()
               if complete and smoke_ok and row['mean_log_pct'] <= -0.15]
    winner = min(passing, key=lambda arm: (arms[arm]['mean_log_pct'], arm != 'h')) if passing else None
    return dict(complete=complete, smoke_bitexact=smoke_ok, arms=arms, winner=winner)


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'smoke':
        print(json.dumps(smoke(), indent=2))
    elif mode == 'screen':
        print(json.dumps(screen(), indent=2))
    elif mode == 'confirm':
        decision = screen()
        winner = decision['winner']
        assert decision['complete'] and winner, 'No eligible complete screen winner'
        fresh, confirm = changes(winner), changes(winner, True)
        all32 = stats(fresh + confirm)
        oos = stats(confirm)
        passed = (len(confirm) == 16 and all32['n'] == 32
                  and all32['mean_log_pct'] <= -0.15 and oos['mean_log_pct'] < 0)
        print(json.dumps(dict(winner=winner, all32=all32, confirmation=oos,
                              passed=passed), indent=2))
    else:
        raise SystemExit('usage: table_v1.py smoke|screen|confirm')
