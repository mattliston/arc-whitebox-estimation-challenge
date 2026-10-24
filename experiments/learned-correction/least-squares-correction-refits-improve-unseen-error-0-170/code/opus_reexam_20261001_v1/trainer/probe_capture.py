"""TAG opus-trainer (2026-10-01): capture what the learned mean correction sees and does at the FINAL layer, for the headroom probe.
One forward-only process (torch card, untrained RRPI parameters unless a parameter file is given, no_grad, about 10 s and 3.5 GB
per network). For each network it stores, for the 1024 units of layer 15: the 126 inputs of the message network (H), its second
hidden layer (h2, 128), its output state (64), the unit scale s = sigma x rms[15] x gain[15], the correction delta the card
applied, the uncorrected mean (host = m - delta), the card's prediction m (before the clamp at 0) and the 1e9-sample truth.
The recomputed delta must reproduce the card's own delta (checked per network).
usage: OMP_NUM_THREADS=1 ... python -B probe_capture.py <new out dir> <comma-separated ranks> [params.pt]"""
import sys
from pathlib import Path

import otr_engine as E
from otr_common import data_paths, load_params

torch, np = E.torch, E.np
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
if len(sys.argv) > 3:
    E.set_parameters(*load_params(torch, sys.argv[3]))
P = E.p
for rank in [int(x) for x in sys.argv[2].split(',')]:
    out = OUT / ('probe_r%04d.npz' % rank)
    if out.exists():
        continue
    E.collect()
    weights_path, truth_path = data_paths(rank)
    weights = np.load(weights_path, allow_pickle=False)
    with np.load(truth_path, allow_pickle=False) as saved:
        truth = np.asarray(saved['mc_post'][15], dtype=np.float64).copy()
    E.LC.JT.DIALS_T = E.DL
    grabbed = {}
    with torch.no_grad():
        cd = E.LC.card(torch, weights, P, E.T['a'], E.T['c'], E.e, fine_ckpt=False)
        cd.hooks['k4_15'] = lambda layer, value, ctx: value
        E.install_star(cd)
        inner = cd.env['gnn_step']
        def spy(l, x, sigma, W, state, P_):
            delta, hs = inner(l, x, sigma, W, state, P_)
            if l == 15:
                grabbed.update(x=x, sigma=sigma, W=W, state_in=state, delta=delta, hs=hs)
            return delta, hs
        cd.env['gnn_step'] = spy
        features = cd.env['final_scalar_features']
        def spy_host(layer, host, aux, w):
            grabbed['host'] = host
            return features(layer, host, aux, w)
        cd.env['final_scalar_features'] = spy_host
        original_step = cd.step
        def step(layer, state):
            try:
                result = original_step(layer, state)
                if layer == 15:
                    grabbed['m'] = result[0]
                return result
            finally:
                E.collect()
        cd.step = step
        cd.run(False)
        # the body of the card's gnn_step at layer 15, kept in pieces
        f32 = torch.float32
        x = torch.clip((grabbed['x'].to(f32) - P['centers'][15]) / P['scales'][15], -8, 8)
        x = x * torch.tensor([1] * 13 + [0] * 3 + [1] * 6 + [0] * 7 + [1], dtype=f32)[None, :]
        g = x[:, 5:6]
        src = grabbed['state_in'] @ P['msg_w'].T + P['msg_b']
        cm = P['msg_w'].shape[0] // 3
        w = grabbed['W'].to(f32); w2 = w * w; w3 = w2 * w
        m1 = w.T @ src[:, :cm]; m2 = w2.T @ src[:, cm:2 * cm]; m3 = w3.T @ src[:, 2 * cm:]
        e1 = w2.T @ (src[:, :cm] ** 2); e2 = (w2 * w2).T @ (src[:, cm:2 * cm] ** 2); e3 = (w3 * w3).T @ (src[:, 2 * cm:] ** 2)
        msgs = torch.cat([g * m1, g * m2, g * m3, torch.log1p(e1.clamp_min(0)), torch.log1p(e2.clamp_min(0)),
                          torch.log1p(e3.clamp_min(0))], dim=1)
        H = torch.cat([x, torch.clip(msgs, -8, 8)], dim=1)
        silu = lambda z: z / (1 + torch.exp(-z))
        h1 = silu(H @ P['enc0_w'].T + P['enc0_b'])
        h2 = silu(h1 @ P['enc2_w'].T + P['enc2_b'])
        state = h2 @ P['enc4_w'].T + P['enc4_b']
        scale = grabbed['sigma'].to(f32) * P['rms'][15] * (1 + 0.25 * torch.tanh(P['gain'][15:16]))
        delta = scale * (state @ P['head'][15] + P['head_b'][15])
        mismatch = float((delta - grabbed['delta']).abs().max() / grabbed['delta'].abs().max())
        assert mismatch < 1e-4, ('recomputed correction differs from the card', rank, mismatch)
        m, host = grabbed['m'].to(torch.float64), grabbed['host'].to(torch.float64)
        applied = float(((m - host) - grabbed['delta'].to(torch.float64)).abs().max() / grabbed['delta'].abs().max())
        assert applied < 1e-6, ('the card applied something else than the message network correction', rank, applied)
        np.savez(out, H=H.numpy(), h2=h2.numpy(), state=state.numpy(), scale=scale.numpy(), delta=grabbed['delta'].numpy(),
                 m=m.numpy(), host=host.numpy(), truth=truth, mismatch=mismatch)
    cd.step = original_step
    print('rank %d: final-layer MSE %.6e (uncorrected %.6e), recomputed-delta mismatch %.1e' % (
        rank, float(((m.clamp_min(0).numpy() - truth) ** 2).mean()), float(((host.numpy() - truth) ** 2).mean()), mismatch),
        flush=True)
    del cd, grabbed, weights
