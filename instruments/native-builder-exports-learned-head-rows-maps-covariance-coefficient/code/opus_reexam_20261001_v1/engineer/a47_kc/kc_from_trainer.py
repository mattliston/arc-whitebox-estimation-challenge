"""TAG opus_engineer a47 v2: the trainer's kappa_C (layers 1-14) from a hand-over's physics_dials.json -> a47_kc/kc/<name>.json as kc_l.
The trainer's export (O/trainer/export_dials.py) writes 'kappa_C_per_layer' = {'values': [kappa_l for l = 0..15], ...} with kappa = 1 - kc
(the share of the rank-2 update the step keeps; the card as fired: 1); kc_l = 1 - kappa_l. A key 'kc' (kc itself: a list of 16 = layers
0..15 or 14 = layers 1-14, a dict {layer: value}, or a dict with 'value' / 'values') is read as it is. Looked for at the top level, in
'dials' and in 'pair_law_dials'. Prints the layers and kc values, or 'none'.
usage: kc_from_trainer.py <physics_dials.json> <name>"""
import json
import sys
from pathlib import Path
d = json.load(open(sys.argv[1]))
found, kappa = None, False
for scope in (d, d.get('dials', {}), d.get('pair_law_dials', {})):
    if not isinstance(scope, dict):
        continue
    if 'kappa_C_per_layer' in scope:
        found, kappa = scope['kappa_C_per_layer'], True
        break
    if 'kc' in scope:
        found = scope['kc']
        break
if found is None:
    print('none')
    sys.exit(0)
if isinstance(found, dict) and ('values' in found or 'value' in found):
    found = found.get('values', found.get('value'))
if isinstance(found, list):
    assert len(found) in (14, 16), len(found)
    found = {l: found[l - 1] if len(found) == 14 else found[l] for l in range(1, 15)}
kc = {int(k): (1.0 - float(v) if kappa else float(v)) for k, v in found.items()}
assert all(0 <= k <= 15 for k in kc), kc
kc = {k: v for k, v in kc.items() if 1 <= k <= 14}
out = Path(__file__).resolve().parent/'kc'/(sys.argv[2] + '.json')
json.dump(dict(kc={str(k): v for k, v in sorted(kc.items())}, source=sys.argv[1], read_as='1 - kappa_C_per_layer' if kappa else 'kc'), open(out, 'w'))
print(' '.join('%d:%.4g' % kv for kv in sorted(kc.items())))
