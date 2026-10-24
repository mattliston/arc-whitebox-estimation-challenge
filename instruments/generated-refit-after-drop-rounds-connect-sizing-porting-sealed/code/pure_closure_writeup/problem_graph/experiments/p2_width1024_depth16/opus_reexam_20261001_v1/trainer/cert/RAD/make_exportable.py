"""TAG opus-trainer RAD (10-12 07:15Z): the exportable copy of a frozen point for export_dials_s46_v4.py: per-layer shifts at the layers where a
family does not act are filled with the nearest acting layer's value (no prediction changes; the exporter's flatness checks then see the acting layers
only): law sources / dscale, k31l ages, bsrc, kappa act at 1-14 (layers 0 / 15 <- 1 / 14); F4 / F4C (f4a*, x1a*, x2a*) act at 0-14 (layer 15 <- 14);
K4L (k4la*) acts at 1-15 (layer 0 <- 1). Everything else is copied as it is (iso stays per layer for the sidecar).
usage (from O/trainer): python -B cert/RAD/make_exportable.py <params.pt> <new exportable .pt>"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, '.')
from otr_common import load_params, load_pl, save_params

src, out = Path(sys.argv[1]), Path(sys.argv[2])
assert not out.exists()
A, d = load_params(torch, src); pl = {k: [float(x) for x in v] for k, v in load_pl(torch, src).items()}
filled = []
for f, v in pl.items():
    if f in ('nfd', 'n6b', 'n6r', 'n6d', 'nkl1', 'nkl2', 'nkl3', 'bsrc', 'kappa'):
        v[0], v[15] = v[1], v[14]; filled.append(f)
    elif f in ('f4a1', 'f4a2', 'x1a1', 'x1a2', 'x2a1', 'x2a2'):
        v[15] = v[14]; filled.append(f)
    elif f in ('k4la1', 'k4la2'):
        v[0] = v[1]; filled.append(f)
save_params(torch, out, A, d, pl)
print('exportable written to %s; filled %s' % (out, filled))
