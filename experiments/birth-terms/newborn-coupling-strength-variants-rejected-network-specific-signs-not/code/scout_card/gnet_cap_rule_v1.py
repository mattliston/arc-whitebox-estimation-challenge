"""TAG scout-card (2026-10-13 03:4xZ): the physicist's PRE-REGISTERED large-s rule for the gated beta's fresh confirm, evaluated on the G-net c 2
screen (arm gk+gbgnet2, 7100584-607, results in run_k6 + run_k7, paired vs run_k1's gk). Runtime z per layer = the 7 'gbg net L<l> f <f> z <z>' lines
the worker printed right before that eval's '"event": "done"' line (O/scout_card/<run>/worker_w*.log).
Groups: A (large s) = networks with max z over L8-14 > 1 (some layer's gate 2 clip(z, 0, 2) > 2); B (moderate) = 0 < max z <= 1; max z <= 0 = ungated,
left out. Rule: mean log effect (vs gk) over A > 0 AND over B < 0 -> the confirm caps s at 2 (z clipped at 1: cutsim_sc9 gbg_zmax 1, atom
gbgnet2cap); otherwise the confirm keeps c 2 uncapped. Prints one line."""
import glob
import json
import math
import re
from pathlib import Path

S = Path(__file__).resolve().parent
ARM = 'gk+gbgnet2'


def runtime_z():
    out = {}
    for log in sorted(glob.glob(str(S / 'run_k[67]' / 'worker_w*.log'))):
        gate = []
        for line in Path(log).read_text().splitlines():
            m = re.match(r'gbg net L(\d+) f [0-9.]+ z ([-+0-9.]+)$', line.strip())
            if m:
                gate.append((int(m.group(1)), float(m.group(2)))); continue
            if '"event": "done"' in line and ARM in line:
                rid = json.loads(line)['id']; seed = int(rid.split('_r')[1].split('_')[0])
                last = gate[-7:]; assert [l for l, _ in last] == list(range(8, 15)), (rid, last)
                out[seed] = [z for _, z in last]
            if '"event": "done"' in line:
                gate = []
    return out


def mse(run_glob, arm):
    res = {}
    for f in sorted(glob.glob(str(S / run_glob / 'results' / f'sc_*_{arm}.json'))):
        r = json.loads(Path(f).read_text()); res.setdefault(int(r['id'].split('_r')[1].split('_')[0]), r['mse'])
    return res


z = runtime_z(); g = mse('run_k1', 'gk'); c2 = mse('run_k[67]', ARM)
eff = {s: 100 * math.log(c2[s] / g[s]) for s in c2 if s in g}
A = sorted(s for s in eff if s in z and max(z[s]) > 1); B = sorted(s for s in eff if s in z and 0 < max(z[s]) <= 1)
mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
ma, mb = mean([eff[s] for s in A]), mean([eff[s] for s in B])
cap = bool(A and B and ma > 0 and mb < 0)
print(f'large-s rule (pre-registered 03:4xZ): runtime z found for {len(z)} of {len(eff)} networks; A (max z > 1) n {len(A)} mean log {ma:+.3f}% '
      f'({" ".join(str(s)[-3:] for s in A)}); B (0 < max z <= 1) n {len(B)} mean log {mb:+.3f}% ({" ".join(str(s)[-3:] for s in B)}); '
      f'-> confirm form: {"CAPPED (c 2, z clipped at 1, s <= 2)" if cap else "c 2 UNCAPPED (as frozen)"}')
