#!/usr/bin/env python3
"""TAG fable-covhunter (CH-48): function-preserving 2x widening of the GNN trunk's two hidden layers (enc0: 126->128, enc2: 128->128
become 126->256, 256->256; enc4 256->64), starting from B2 epoch 1. Blocks: enc0 new rows ~ N(0, std(enc0_w)^2), new bias 0;
enc2 = [[old, 0], [new]] (old H2 units ignore the new H0 units; new H2 rows ~ N(0, (std(enc2_w)/sqrt 2)^2) over all 256 inputs, bias 0);
enc4 = [old, 0] (new H2 units start with zero output weight). So the state, delta and card output equal B2 epoch 1 up to BLAS rounding
at step 0; msg (cm 16), state (64), heads and everything else are untouched (the message edge source needs cm 16, state 64).
Writes wide2x_B2ep1_init.pt (same p/a/c/e format; seed 0)."""
from pathlib import Path
import torch
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
SRC = P2 / 'fable_physicist_v1/refit6b/s6b_B2_fullnet_lr1e4_trunk5e5/params_epoch1.pt'; OUT = HERE / 'wide2x_B2ep1_init.pt'
assert not OUT.exists(), OUT
sv = torch.load(SRC); p = dict(sv['p']); g = torch.Generator().manual_seed(0)
H0, H2 = p['enc0_w'].shape[0], p['enc2_w'].shape[0]; assert (H0, H2) == (128, 128)
e0, e2, e4 = p['enc0_w'], p['enc2_w'], p['enc4_w']
new0 = torch.randn(H0, e0.shape[1], generator=g) * e0.std()
new2 = torch.randn(H2, 2 * H0, generator=g) * (e2.std() / 2 ** 0.5)
p['enc0_w'] = torch.cat([e0, new0], 0); p['enc0_b'] = torch.cat([p['enc0_b'], torch.zeros(H0)])
p['enc2_w'] = torch.cat([torch.cat([e2, torch.zeros(H2, H0)], 1), new2], 0); p['enc2_b'] = torch.cat([p['enc2_b'], torch.zeros(H2)])
p['enc4_w'] = torch.cat([e4, torch.zeros(e4.shape[0], H2)], 1)
p = {k: v.to(torch.float32).contiguous() for k, v in p.items()}
torch.save(dict(p=p, a=sv['a'], c=sv['c'], e=sv['e']), OUT)
print({k: tuple(p[k].shape) for k in ('enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')},
      'trunk params', sum(p[k].numel() for k in ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')))
