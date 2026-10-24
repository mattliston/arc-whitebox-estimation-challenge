"""TAG opus_engineer: a run M parameter file exported the way the trainer's announcer exports it (O/trainer/announce.py,
export_candidate), for a file the announcer has not exported itself: the trainer's exporter (O/trainer/export.py, unchanged) with
the families it has a native form for (rc, edge, bsrc, iso, k21n, k31), and the scale of the learned mean correction (ds) folded
into the two literals the correction is linear in (head and head_b of layers 1-15), exactly as the announcer does. The remaining
otr_m.py families (f4a1, f4a2, k4la1, k4la2, k22) are levers of stack/build.py (levers_from_params.py prints them).
usage: python -B export_for_build.py <params.pt> <new candidate.py>"""
import json
import sys
from pathlib import Path

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'trainer'))
import numpy as np
import torch
import export as X
from otr_common import load_params, load_pl, sha256

out = Path(sys.argv[2]).resolve()
arrays, dials = load_params(torch, sys.argv[1])
pl = load_pl(torch, sys.argv[1])
arrays = {key: value.numpy().copy() for key, value in arrays.items()}
if 'ds' in pl:   # as O/trainer/announce.py: the card does not scale layer 0
    scale = 1.0 + np.asarray(pl['ds'], dtype=np.float64)
    scale[0] = 1.0
    assert arrays['p.head'].shape == (16, 64) and arrays['p.head_b'].shape == (16,)
    arrays['p.head'] = (arrays['p.head'] * scale[:, None]).astype(np.float32)
    arrays['p.head_b'] = (arrays['p.head_b'] * scale).astype(np.float32)
changed = X.export(arrays, dials, out, {family: values for family, values in pl.items() if family in X.NATIVE})
print(json.dumps(dict(candidate=str(out), sha256=sha256(out), params=str(Path(sys.argv[1]).resolve()), changed_statements=changed,
                      in_candidate=sorted(set(pl) & (X.NATIVE | {'ds'})), left_to_the_builder={family: pl[family][1] for family in pl if family not in X.NATIVE | {'ds'}})))
