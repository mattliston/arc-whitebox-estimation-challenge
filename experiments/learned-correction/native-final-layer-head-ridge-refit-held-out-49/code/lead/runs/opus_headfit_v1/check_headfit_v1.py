"""[opus-headfit] checks. (1) capture fidelity: captured y15 (S2tr under sys.setprofile) vs the S2tr control natives in opus_retune_v1/ctl_all24
(final layer, all captured control seeds). (2) with --native DIR: the built candidate's native run vs the control on the same seed: FLOPs equal,
layers 0-14 byte-identical, layer 15 vs the offline refit y = (y15 - delta) + s (state @ head' + b') (float32 rounding expected)."""
import json, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; O = H.parents[2]; CTL = O/'lead/runs/opus_retune_v1/ctl_all24'
rows = lambda p: {r['seed']: r for r in (json.loads(p.read_text())['rows'] if isinstance(json.loads(p.read_text())['rows'], list) else [])}
worst = 0.0; n = 0
for p in sorted(CTL.glob('native_*_pred.npz')):
    s = int(p.name.split('_')[1]); c = H/f'cap/head_{s}.npz'
    if not c.exists(): continue
    a = np.load(c)['y15'].astype(np.float64); b = np.load(p)[str(s)][-1].astype(np.float64)
    worst = max(worst, float(np.max(np.abs(a - b)) / np.max(np.abs(b)))); n += 1
print(f'capture fidelity: {n} control seeds, max rel |y15_capture - y15_control| = {worst:.2e}')
if '--native' in sys.argv:
    D = Path(sys.argv[sys.argv.index('--native') + 1]); z = np.load(H/'REFIT_head15_v1.npz'); th = z['theta']
    for p in sorted(D.glob('native_*_pred.npz')):
        s = int(p.name.split('_')[1]); new = np.load(p)[str(s)]; ctl = np.load(CTL/f'native_{s}_pred.npz')[str(s)]
        fn = json.loads((D/f'native_{s}.json').read_text())['rows'][0]['flops']; fc = json.loads((CTL/f'native_{s}.json').read_text())['rows'][0]['flops']
        c = np.load(H/f'cap/head_{s}.npz'); sc = c['sigma'] * float(np.ravel(c['rms'])[0]) * float(np.ravel(c['gain'])[0])
        X = np.concatenate([c['state'], np.ones((c['state'].shape[0], 1))], axis=1); off = (c['y15'] - c['delta']) + sc * (X @ th)
        d15 = float(np.max(np.abs(new[-1] - off)) / np.max(np.abs(off))); moved = float(np.sqrt(np.mean((new[-1].astype(np.float64) - ctl[-1]) ** 2)) / np.sqrt(np.mean(ctl[-1].astype(np.float64) ** 2)))
        print(f'seed {s}: flops {fn} vs control {fc} equal={fn == fc}; layers 0-14 byte-identical={new[:15].tobytes() == ctl[:15].tobytes()}; '
              f'L15 vs offline refit max rel {d15:.2e}; L15 moved vs control rel rms {moved:.2e}')
