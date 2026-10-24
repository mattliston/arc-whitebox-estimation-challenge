#!/usr/bin/env python3
"""TAG fable-physicist (CR-21): TRUTH-SIDE test of the multi-scale-mixture model of per-unit kurtosis (the one cheap full-depth structure
left after the Gaussian-restart profile, CR-19). A bias-free ReLU net is positively homogeneous, so every pre-activation scales with the
upstream layer norms. If z_i = s_i * y_i with the scale noise eps_i = s_i^2 - 1 linear in the global norm modes q = (|x|^2, |h^0|^2, ...,
|h^{L-1}|^2)/n, then t4_i = 3 Var(eps_i) = 3 c_i^T Cov(q)^{-1} c_i with c_i = kappa(z_i, z_i, q)/v_i: a PARAMETER-FREE per-unit prediction
from cross cumulants. It is a lower bound on the scale-mixture kurtosis the norms carry. MC on MPS (true network, x ~ N(0,I)), two halves.
usage: norm_coupling_mc.py <rank> <samples per half, millions>   -> carry/norm_coupling_r<rank>_h<M>M.npz"""
import sys, json, time, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[2]
sys.path.insert(0, str(P2 / 'fable_engineer_v1'))
import numpy as np, torch, mps_lock
rank = int(sys.argv[1]); NH = int(float(sys.argv[2]) * 1e6)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load(STORE / Path(man['path']).parent / 'weights.npy')
out_path = HERE / f'norm_coupling_r{rank}_h{sys.argv[2]}M.npz'; assert not out_path.exists(), out_path
TGT = (12, 13, 14, 15); BATCH, n, NQ = 131072, 1024, 17           # modes: q0 = |x|^2/n, q_{l+1} = |h^l|^2/n, l = 0..15
dev = torch.device('mps')
print(json.dumps(dict(tag='CR-21 norm_coupling', rank=rank, per_half=NH, targets=TGT)), flush=True)
with mps_lock.hold(f'norm_coupling r{rank} (pid {os.getpid()})'):
    wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
    res = {}; t0 = time.time(); ref = {}; qref = None
    for half, seed in (('A', 7100 + rank), ('B', 9100 + rank)):
        torch.manual_seed(seed)
        S = {L: np.zeros((4, n)) for L in TGT}; X2 = {L: np.zeros((n, NQ)) for L in TGT}; X1 = {L: np.zeros((n, NQ)) for L in TGT}
        Q1 = np.zeros(NQ); Q2 = np.zeros((NQ, NQ)); done = 0
        while done < NH:
            b = min(BATCH, NH - done); x = torch.randn((b, n), device=dev)
            q = [(x * x).sum(1) / n]; zs = {}
            h = x
            for l in range(16):
                z = h @ wg[l]; h = torch.clamp(z, min=0.0); q.append((h * h).sum(1) / n)
                if l in TGT: zs[l] = z
            Q = torch.stack(q, 1)                                        # (b, 17)
            if qref is None: qref = Q.mean(0).clone()
            Qc = Q - qref
            Q1 += Qc.sum(0).cpu().numpy(); Q2 += (Qc.T @ Qc).cpu().numpy().astype(np.float64)
            for L, z in zs.items():
                if L not in ref: ref[L] = z.mean(0).clone()
                d = z - ref[L]; d2 = d * d
                S[L] += torch.stack((d.sum(0), d2.sum(0), (d2 * d).sum(0), (d2 * d2).sum(0))).cpu().numpy().astype(np.float64)
                X2[L] += (d2.T @ Qc).cpu().numpy().astype(np.float64); X1[L] += (d.T @ Qc).cpu().numpy().astype(np.float64)
            done += b
        for L in TGT:
            res[f'S_{half}_{L}'] = S[L] / NH; res[f'X2_{half}_{L}'] = X2[L] / NH; res[f'X1_{half}_{L}'] = X1[L] / NH
        res[f'Q1_{half}'] = Q1 / NH; res[f'Q2_{half}'] = Q2 / NH
        print(json.dumps(dict(half=half, samples=NH, wall=round(time.time() - t0, 1))), flush=True)
    res['N'] = np.array(NH)
    np.savez(out_path, **res)
print(json.dumps(dict(done=True, out=str(out_path), wall=round(time.time() - t0, 1))), flush=True)
