#!/usr/bin/env python3
"""TAG fable-physicist (CR-19): GAUSSIAN-RESTART memory-depth test for per-unit k3/k4 (lead's k3/k4 page question, 09-24 23:2x).
Question: how many layers back is the per-unit pre-activation k3/k4 at L12-15 born? A deterministic construction that starts at layer l0
(a late fourth-order carrier, or an exact backward contraction over the last L-l0 layers) can at best know the state at l0 to second order
and compute everything after it exactly. That ceiling is sampled here: z^{l0} ~ N(truth mean, truth covariance) (covhunter's 32M mcall
files, an independent draw from the scoring truth), then the TRUE network forward (relu, x @ W[l]) to L15, per-unit k3/k4 at every later
layer from raw moments of (z - ref) in two independent halves A/B (noise floor). l0 = 0 is the positive control: z0 = x @ W[0], x ~ N(0,I)
is the exact network (it must match the 32M truth within the halves' noise). Network convention as mc_truth_mps.py / mc_allayers_mps.py.
MPS float32 GEMMs, float64 CPU accumulation per batch, FIFO MPS lock (fable_engineer_v1/mps_lock.py); launch behind machine_gate.sh.
usage: gauss_restart_mc.py <rank> <l0 list, e.g. 0,2,5,8,10,11,12,13> <samples per half in millions> [control samples per half, M]
out: carry/gauss_restart_r<rank>_h<M>M_l<l0s>.npz  keys  {m,v,k3,k4}_{A,B}_<l0>  (16 x 1024 float64, zero at layers <= l0), N_<l0>"""
import sys, json, time, os
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[2]
sys.path.insert(0, str(P2 / 'fable_engineer_v1'))
import numpy as np, torch, mps_lock
rank = int(sys.argv[1]); L0S = [int(x) for x in sys.argv[2].split(',')]; NH = int(float(sys.argv[3]) * 1e6)
NH0 = int(float(sys.argv[4]) * 1e6) if len(sys.argv) > 4 else NH
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load(STORE / Path(man['path']).parent / 'weights.npy')
MC = np.load(P2 / 'fable_covhunter_v1' / f'mcall_rank{rank:04d}_P32M.npz')
out_path = HERE / f"gauss_restart_r{rank}_h{sys.argv[3]}M_l{sys.argv[2].replace(',', '-')}.npz"
assert not out_path.exists(), out_path                      # never overwrite: version instead
BATCH, n = 131072, 1024
dev = torch.device('mps')
print(json.dumps(dict(tag='CR-19 gauss_restart', rank=rank, l0s=L0S, per_half=NH, control_per_half=NH0, batch=BATCH,
                      cov_source=f'mcall_rank{rank:04d}_P32M (seed law SEED_XOR default; scoring truth is the indep32M draw)')), flush=True)
with mps_lock.hold(f'gauss_restart r{rank} l0={sys.argv[2]} (pid {os.getpid()})'):
    wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
    res = {}
    t0 = time.time()
    for l0 in L0S:
        if l0 == 0:
            mean = np.zeros(n); F = None                             # exact network: z0 = x @ W[0]
        else:
            mean = np.asarray(MC[f'mean_pre_{l0}'], dtype=np.float64); C = np.asarray(MC[f'cov_pre_{l0}'], dtype=np.float64)
            C = (C + C.T) / 2; ev, U = np.linalg.eigh(C); neg = float(-ev.min()) if ev.min() < 0 else 0.0
            F = torch.from_numpy((U * np.sqrt(np.clip(ev, 0, None))).T.astype(np.float32)).to(dev)   # z = m + eps @ F, F^T F = C
            print(json.dumps(dict(l0=l0, cov_min_eig=float(ev.min()), cov_max_eig=float(ev.max()), clipped=neg)), flush=True)
        mg = torch.from_numpy(mean.astype(np.float32)).to(dev)
        nh = NH0 if l0 == 0 else NH
        ref = {}
        for half, seed in (('A', 1000 + 17 * l0 + rank), ('B', 5000 + 17 * l0 + rank)):
            torch.manual_seed(seed)
            S = np.zeros((16, 4, n)); done = 0
            while done < nh:
                b = min(BATCH, nh - done)
                eps = torch.randn((b, n), device=dev)
                z = eps @ wg[0] if l0 == 0 else eps @ F + mg
                for l in range(l0 + 1, 16):
                    z = torch.clamp(z, min=0.0) @ wg[l]
                    if l not in ref: ref[l] = z.mean(0).clone()
                    d = z - ref[l]; d2 = d * d
                    S[l] += torch.stack((d.sum(0), d2.sum(0), (d2 * d).sum(0), (d2 * d2).sum(0))).cpu().numpy().astype(np.float64)
                done += b
            r0 = {l: ref[l].cpu().numpy().astype(np.float64) for l in ref}
            m = np.zeros((16, n)); v = np.zeros((16, n)); k3 = np.zeros((16, n)); k4 = np.zeros((16, n))
            for l in range(l0 + 1, 16):
                m1, m2, m3, m4 = (S[l] / nh)
                v[l] = m2 - m1 ** 2; k3[l] = m3 - 3 * m1 * m2 + 2 * m1 ** 3
                k4[l] = m4 - 4 * m1 * m3 - 3 * m2 ** 2 + 12 * m1 ** 2 * m2 - 6 * m1 ** 4; m[l] = m1 + r0[l]
            for key, arr in (('m', m), ('v', v), ('k3', k3), ('k4', k4)): res[f'{key}_{half}_{l0}'] = arr
            print(json.dumps(dict(l0=l0, half=half, samples=nh, wall=round(time.time() - t0, 1))), flush=True)
        res[f'N_{l0}'] = np.array(2 * nh)
    np.savez(out_path, **res)
print(json.dumps(dict(done=True, out=str(out_path), wall=round(time.time() - t0, 1))), flush=True)
