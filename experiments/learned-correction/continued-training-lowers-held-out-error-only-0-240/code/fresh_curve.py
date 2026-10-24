"""TAG opus-trainer (2026-10-01): the pilot's EMA parameters on networks nobody has trained on, as far as evaluated.
Reads cert/fresh60_ema00018.json, cert/fresh60_ema00023.json (60 newer-rank training networks the run had not reached) and, once
cert/fresh_at36.sh has produced it, cert/fresh36_ema00036.json (the 36 of them still unreached after 36 updates). Untrained
losses come from the probe captures. Prints the pooled change against the untrained parameters on the common networks.
usage: python -B fresh_curve.py        (from O/trainer/)"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
files = [(18, 'fresh60_ema00018.json'), (23, 'fresh60_ema00023.json'), (36, 'fresh36_ema00036.json')]
have = [(k, json.loads((HERE / 'cert' / name).read_text())) for k, name in files if (HERE / 'cert' / name).exists()]
common = sorted(set.intersection(*[set(row['ranks']) for _, row in have]))
base = []
for rank in common:
    with np.load(HERE / 'probe/untrained' / ('probe_r%04d.npz' % rank)) as saved:
        base.append(float(((np.maximum(saved['m'], 0) - saved['truth']) ** 2).mean()))
base = np.array(base)
print('%d networks nobody has trained on, common to all evaluated points; final-layer MSE, EMA parameters vs untrained' % len(common))
previous = None
for k, row in have:
    mse = np.array([row['final_layer_mse'][str(rank)] for rank in common])
    change = 100 * (mse / base - 1)
    text = 'after %2d updates: %+.3f%% (se %.3f, better on %d/%d)' % (k, 100 * (mse.mean() / base.mean() - 1),
                                                                  change.std(ddof=1) / len(common) ** 0.5, (mse < base).sum(), len(common))
    if previous is not None:
        step = 100 * (mse / previous - 1)
        text += '   against the point before, same networks: %+.3f%% (se %.3f)' % (100 * (mse.mean() / previous.mean() - 1),
                                                                                step.std(ddof=1) / len(common) ** 0.5)
    print(text)
    previous = mse
