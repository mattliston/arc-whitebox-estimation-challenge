#!/usr/bin/env python3
"""TAG fable-analyst (AN-52): compare a TRAIN640 refit (A/s6b_T640_<tag>/log.json) against its own VALID-before and the physicist's B2
epoch 1 (VALID16: B2 run log epochs[1]; VALID40: fable_physicist_v1/refit6b/s6b_B2_ep1_valid40/eval_n40.json). Pooled = ratio of the
mean MSE over the nets; wins = nets with lower MSE. Bars (RESTART doc): pooled <= -0.3% vs before with >= 24/40 wins; replace B2 ep1 only
if pooled VALID40 is below B2 ep1's. usage: compare_t640.py <tag> <mid|end>"""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
tag, stage = sys.argv[1], sys.argv[2]
L = json.loads((HERE / f's6b_T640_{tag}' / 'log.json').read_text())
V = [str(r) for r in json.loads((P2 / 'codex_resume_20260919_v1/learnable_joint_basis_fit_v1/inputs.json').read_text())['valid']]
B2_16 = json.loads((P2 / 'fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/log.json').read_text())['epochs'][1]['valid']
B2_40 = json.loads((P2 / 'fable_physicist_v1/refit6b/s6b_B2_ep1_valid40/eval_n40.json').read_text())['valid']
before = L['valid_before']
def cmp(new, ref, nets):
    return dict(rel=float(np.mean([new[k] for k in nets]) / np.mean([ref[k] for k in nets]) - 1), wins=int(sum(new[k] < ref[k] for k in nets)), n=len(nets))
out = dict(tag=tag, stage=stage)
if stage == 'mid':
    m = L['mid']['valid16']; nets = [k for k in V[:16]]
    out.update(update=L['mid']['update'], valid16=float(np.mean([m[k] for k in nets])), vs_before=cmp(m, before, nets), vs_B2ep1=cmp(m, B2_16, nets))
else:
    e = L['epochs'][0]['valid']; n40, n16 = V[:40], V[:16]
    out.update(valid40=float(np.mean([e[k] for k in n40])), vs_before_40=cmp(e, before, n40), vs_B2ep1_40=cmp(e, B2_40, n40),
               vs_before_16=cmp(e, before, n16), vs_B2ep1_16=cmp(e, B2_16, n16), B2ep1_valid40=float(np.mean([B2_40[k] for k in n40])))
    out['pass_bar'] = out['vs_before_40']['rel'] <= -0.003 and out['vs_before_40']['wins'] >= 24
    out['beats_B2ep1'] = out['vs_B2ep1_40']['rel'] < 0
if L.get('epochs') and L['epochs'][0].get('knobs'): out['knobs_end'] = L['epochs'][0]['knobs']
print(json.dumps(out, indent=1))
