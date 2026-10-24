#!/usr/bin/env python3
"""TAG fable-analyst (AN-66): H1 FALSIFIER, the SUBSPACE-GAUSSIANIZED restart (lead, from the physicist's rivals page section O). At pre-
activation l0, keep the TRUE state in the top-R eigen-subspace of the true covariance and Gaussianize only the complement, then run the TRUE
network forward (row convention z_k = relu(z_{k-1}) @ W[k]) and record per-unit m/v/k3/k4 at every later layer, halves A/B, in the
physicist's gauss_restart format (so refit6b/carry/gauss_restart_eval.py scores it unchanged).
  z = the exact network at pre-l0 (x ~ N(0, I) through W[0..l0] with relus); d = z - mean_pre_l0 (covhunter's 32M mcall);
  U, lam = eigenvectors / eigenvalues of mcall cov_pre_l0, descending, negative eigenvalues clipped at 0; s = d @ U
  z'_R = m + s[:, :R] U[:, :R]^T + (eps[:, R:] sqrt(lam[R:])) U[:, R:]^T      (one eps ~ N(0, I) per batch, shared by every R)
R = 0 is the plain Gaussian restart (the physicist's restart(l0), an independent draw); R = n is the exact network (positive control).
Mean and covariance are exact for every R; within the top-R subspace the joint law is the true one; every higher cumulant that involves
the complement is removed. MPS float32 GEMMs, float64 CPU accumulation per batch; the MPS lock is held in a with-block; launch behind
fable_porter_v1/machine_gate.sh.
Optional 5th argument 'bottom': keep the BOTTOM-R eigen-directions (smallest eigenvalues) instead of the top-R (default 'top';
the complement is still Gaussianized independently, exact because eigen-subspaces are uncorrelated). v1 kept as subspace_restart_mc_v1.py.
usage: subspace_restart_mc.py <rank> <l0 list> <R list> <samples per half, millions> [top|bottom]   -> subrestart_r<rank>_R<R>_h<M>M.npz
(bottom: subrestart_r<rank>_bottomR<R>_h<M>M.npz)"""
import sys, json, time, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_engineer_v1'))
import numpy as np, torch, mps_lock
rank = int(sys.argv[1]); L0S = [int(x) for x in sys.argv[2].split(',')]; RS = [int(x) for x in sys.argv[3].split(',')]; NH = int(float(sys.argv[4]) * 1e6)
BASIS = sys.argv[5] if len(sys.argv) > 5 else 'top'; assert BASIS in ('top', 'bottom')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load(STORE / Path(man['path']).parent / 'weights.npy')
MC = np.load(P2 / 'fable_covhunter_v1' / f'mcall_rank{rank:04d}_P32M.npz')
outs = {R: HERE / (f'subrestart_r{rank}_R{R}_h{sys.argv[4]}M.npz' if BASIS == 'top' else f'subrestart_r{rank}_bottomR{R}_h{sys.argv[4]}M.npz') for R in RS}
for f in outs.values(): assert not f.exists(), f
BATCH, n = 65536, 1024
dev = torch.device('mps')
print(json.dumps(dict(tag='AN-66 subspace_restart', rank=rank, l0s=L0S, Rs=RS, per_half=NH, batch=BATCH, basis=BASIS)), flush=True)
res = {R: {} for R in RS}
with mps_lock.hold(f'subspace_restart r{rank} l0={sys.argv[2]} R={sys.argv[3]} (pid {os.getpid()})'):
    wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
    t0 = time.time()
    for l0 in L0S:
        mean = MC[f'mean_pre_{l0}'].astype(np.float64); C = MC[f'cov_pre_{l0}'].astype(np.float64); C = (C + C.T) / 2
        ev, U = np.linalg.eigh(C); ev, U = ev[::-1].copy(), U[:, ::-1].copy()
        if BASIS == 'bottom': ev, U = ev[::-1].copy(), U[:, ::-1].copy()      # smallest eigenvalues first: s[:, :R] = the bottom-R coordinates
        lam = np.clip(ev, 0, None)
        Ug = torch.from_numpy(U.astype(np.float32)).to(dev); mg = torch.from_numpy(mean.astype(np.float32)).to(dev)
        Fg = torch.from_numpy((U * np.sqrt(lam)[None, :]).T.astype(np.float32)).to(dev)          # row k = sqrt(lam_k) U[:, k]^T
        print(json.dumps(dict(l0=l0, top_eig=[round(float(x), 4) for x in lam[:4]], share_top=[round(float(lam[:R].sum() / lam.sum()), 4) for R in RS])), flush=True)
        for half, seed in (('A', 21000 + 17 * l0 + rank), ('B', 25000 + 17 * l0 + rank)):
            torch.manual_seed(seed); S = {R: np.zeros((16, 4, n)) for R in RS}; ref = {R: {} for R in RS}; done = 0
            while done < NH:
                b = min(BATCH, NH - done)
                z = torch.randn((b, n), device=dev) @ wg[0]
                for l in range(1, l0 + 1): z = torch.clamp(z, min=0.0) @ wg[l]         # exact pre-l0
                s = (z - mg) @ Ug; eps = torch.randn((b, n), device=dev)
                for R in RS:
                    if R >= n: zz = z
                    elif R == 0: zz = mg + eps @ Fg
                    else: zz = mg + s[:, :R] @ Ug[:, :R].T + eps[:, R:] @ Fg[R:, :]
                    for l in range(l0 + 1, 16):
                        zz = torch.clamp(zz, min=0.0) @ wg[l]
                        if l not in ref[R]: ref[R][l] = zz.mean(0).clone()
                        d = zz - ref[R][l]; d2 = d * d
                        S[R][l] += torch.stack((d.sum(0), d2.sum(0), (d2 * d).sum(0), (d2 * d2).sum(0))).cpu().numpy().astype(np.float64)
                done += b
            for R in RS:
                m = np.zeros((16, n)); v = np.zeros((16, n)); k3 = np.zeros((16, n)); k4 = np.zeros((16, n))
                for l in range(l0 + 1, 16):
                    m1, m2, m3, m4 = S[R][l] / NH
                    v[l] = m2 - m1 ** 2; k3[l] = m3 - 3 * m1 * m2 + 2 * m1 ** 3
                    k4[l] = m4 - 4 * m1 * m3 - 3 * m2 ** 2 + 12 * m1 ** 2 * m2 - 6 * m1 ** 4; m[l] = m1 + ref[R][l].cpu().numpy().astype(np.float64)
                for key, arr in (('m', m), ('v', v), ('k3', k3), ('k4', k4)): res[R][f'{key}_{half}_{l0}'] = arr
            print(json.dumps(dict(l0=l0, half=half, wall=round(time.time() - t0, 1))), flush=True)
        for R in RS: res[R][f'N_{l0}'] = np.array(2 * NH)
    for R in RS: np.savez(outs[R], **res[R])
print(json.dumps(dict(done=True, outs=[str(f) for f in outs.values()], wall=round(time.time() - t0, 1))), flush=True)
