#!/usr/bin/env python3
"""TAG fable-porter (PT-48b): report for star_nys_test.py rows (torch VALID16) + the native flops cut (parity_layers.py outputs
stack/parity_S23_vs_lrnys_q{0,1}.txt, nets 37/39): per arm vs ctrl, pooled raw (sum MSE ratio) with a 20000-resample paired net bootstrap
CI, wins, per-net deltas, the kept plane's energy capture; adjusted = (1 + raw)(1 + bill) - 1 with bill = mean native flops change.
Bar (lead 23:5x): raw <= +0.3% for the bill cut. usage: star_nys_report.py <rows.jsonl>"""
import sys, json, re
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(sys.argv[1])]
M = {}
for r in rows: M.setdefault(r['arm'], {})[r['net']] = r
nets = [n for n in M['ctrl'] if all(n in M[a] for a in ('nys', 'nysp'))]
lz = [(n, M['lz'][n]['mse'], M['ctrl'][n]['mse']) for n in M.get('lz', {})]
print(f"TAG fable-porter PT-48b: star Nystrom range finder, torch VALID16 ({len(nets)} nets), S23-structure card, loopR1 + round-1 dials, no refit")
print("certificate (ctrl copy vs the physicist's lrstar_s1zero_hook): " + "; ".join(f"net {n}: {'EQUAL' if a == b else f'DIFF {b / a - 1:+.2e}'}" for n, a, b in lz))
bill = {}
for v, arm in (('q0', 'nys'), ('q1', 'nysp')):
    t = (HERE / 'stack' / f'parity_S23_vs_lrnys_{v}.txt').read_text()
    bill[arm] = [float(x) / 100 for x in re.findall(r'\(B/A ([+-][0-9.]+)%\)', t)]
c = np.array([M['ctrl'][n]['mse'] for n in nets]); boot = np.random.default_rng(0).integers(0, len(nets), (20000, len(nets)))
print(f"ctrl: mean MSE {c.mean():.4e}; kept-plane energy mean {np.mean([M['ctrl'][n]['e_mean'] for n in nets]):.4f} (min over calls {np.min([M['ctrl'][n]['e_min'] for n in nets]):.4f})")
for arm, label in (('nys', 'Nystrom columns, no power step'), ('nysp', 'Nystrom columns + power step')):
    a = np.array([M[arm][n]['mse'] for n in nets]); rel = a / c - 1
    pooled = a.sum() / c.sum() - 1; bc = a[boot].sum(1) / c[boot].sum(1) - 1; lo, hi = np.percentile(bc, [2.5, 97.5])
    b = float(np.mean(bill[arm])); adj = (1 + pooled) * (1 + b) - 1
    print(f"{arm} ({label}): raw pooled {100 * pooled:+.3f}% [95% CI {100 * lo:+.3f}, {100 * hi:+.3f}], better on {int((rel < 0).sum())}/{len(nets)} nets; "
          f"per net min {100 * rel.min():+.2f}% median {100 * np.median(rel):+.2f}% max {100 * rel.max():+.2f}%")
    print(f"    native bill {100 * b:+.3f}% (nets 37/39: {', '.join(f'{100 * x:+.3f}%' for x in bill[arm])}); adjusted estimate {100 * adj:+.3f}%; "
          f"kept-plane energy mean {np.mean([M[arm][n]['e_mean'] for n in nets]):.4f} (min {np.min([M[arm][n]['e_min'] for n in nets]):.4f}); "
          f"bar raw <= +0.3%: {'PASS' if pooled <= 0.003 else 'FAIL'}")
print("per net (nys / nysp vs ctrl): " + ", ".join(f"{n} {100 * (M['nys'][n]['mse'] / M['ctrl'][n]['mse'] - 1):+.1f}/{100 * (M['nysp'][n]['mse'] / M['ctrl'][n]['mse'] - 1):+.2f}" for n in nets))
