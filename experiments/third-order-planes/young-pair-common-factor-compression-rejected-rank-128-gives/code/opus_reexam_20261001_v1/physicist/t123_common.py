"""TAG opus-physicist T123 helpers: results of an arm (t123_<name>) and the plain card's base per network (T115's base 7100136-150, T123's off151 for 7100151)."""
import json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent


def arm(name):
    out = {}
    for f in (HERE / ('t123_' + name) / 'results').glob('*.json'):
        d = json.loads(f.read_text()); out[int(d['id'].split('_')[1])] = np.array(d['mse_layers'], float)
    return out


def base():
    out = {}
    for f in (HERE / 't115_base' / 'results').glob('*.json'):
        d = json.loads(f.read_text()); out[int(d['id'].split('_')[1])] = np.array(d['mse_layers'], float)
    out.update(arm('off151'))
    return out
