#!/usr/bin/env python3
"""TAG fable-engineer (E-64, lead 19:10: "profile the refit step"): where does one refit training step spend its time? Mirrors
fable_covhunter_v1/refit6b_aux.py's step on the S18 physics: card = s6bcard_j05 (S17 literals, s6b_B2ep1_j05_1ep/params_epoch0.pt) with
fine_ckpt=True + lrstar_torch_hook (the S18 low-rank star), trainable groups 'full' + trunk ('full+P'), loss = final-layer MSE vs
truthpost, one net (37). Times: forward without grad; forward with grad (checkpointed) and backward, each under torch.profiler (CPU):
top operators by self CPU time with input shapes. usage (python with torch): refit_step_profile.py <threads>"""
import sys, json, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_analyst_v1')); sys.path.insert(0, str(P2 / 'fable_physicist_v1/refit6b')); sys.path.insert(0, str(HERE))
import numpy as np, torch
from torch.profiler import profile, ProfilerActivity
import s6bcard, s6bcard_j05
import lrstar_torch_hook as lh
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
p, a, c, e = s6bcard.payload(torch, params=P2 / 'fable_physicist_v1/refit6b/s6b_B2ep1_j05_1ep/params_epoch0.pt')
full = [p['head'], p['head_b'], e['output'], e['bias'], a, c] + [e[k] for k in e if k not in ('output', 'bias')]
trunk = [p[k] for k in ('msg_w', 'msg_b', 'enc0_w', 'enc0_b', 'enc2_w', 'enc2_b', 'enc4_w', 'enc4_b')]
for x in full + trunk: x.requires_grad_(True)
w = s6bcard.weights(37)
tp = torch.tensor(np.load(P2 / 'public_g22_operator_features_v1/truthpost' / 'truthpost_rank0037.npz')['mc_post'][15].astype(np.float64))
t0 = time.time()
with torch.no_grad():
    lh.install(s6bcard_j05.card(torch, w, p, a, c, e), torch).run(False)
print(json.dumps(dict(forward_no_grad_s=round(time.time() - t0, 2))), flush=True)
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    t0 = time.time()
    card = lh.install(s6bcard_j05.card(torch, w, p, a, c, e, fine_ckpt=True), torch)
    pred = card.run(True); loss = ((pred[15].to(torch.float64) - tp) ** 2).mean()
    t1 = time.time(); loss.backward(); t2 = time.time()
print(json.dumps(dict(forward_grad_s=round(t1 - t0, 2), backward_s=round(t2 - t1, 2), loss=float(loss))), flush=True)
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=25, max_name_column_width=40))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_time_total", row_limit=20, max_name_column_width=30, max_shapes_column_width=60))
