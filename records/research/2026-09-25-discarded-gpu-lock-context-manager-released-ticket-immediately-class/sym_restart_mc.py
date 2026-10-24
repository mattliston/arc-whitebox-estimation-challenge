#!/usr/bin/env python3
"""TAG fable-analyst (AN-63): WHICH ORDER of the content born at layer l carries its contribution to per-unit k4 at L13/L14?
The physicist's Gaussian restart at pre-activation l (z_l ~ N(true mean, true cov), covhunter's 32M mcall; l = 0 exact x @ W[0]),
y = relu(z_l), then the TRUE network forward (row convention z_k = relu(z_{k-1}) @ W[k]) in two versions on COMMON random numbers:
  full  y as is                          (= the physicist's restart(l): births at l and later)
  even  E[y] + s (y - E[y]), s = +-1 per sample (Rademacher): every EVEN-order cumulant of the born content kept (covariance and the whole
        fourth-order tensor), every ODD-order one (third order, fifth, ...) removed; later layers act on it exactly.
full - even = what the born odd-order (mainly third-order) content contributes downstream (low noise: common random numbers); compare with
the births-at-l increment restart(l) - restart(l+1) from the physicist's grid. Per-unit k4 (and k3) at L13/L14 per half A/B.
MPS float32 GEMMs, float64 CPU accumulation, FIFO MPS lock; launch behind machine_gate.sh.
usage: sym_restart_mc.py <rank> <birth layers> <samples per half, millions> <tag>   -> sym_restart_r<rank>_<tag>.npz"""
import sys, json, time, os, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
sys.path.insert(0, str(P2 / 'fable_engineer_v1'))
import numpy as np, torch, mps_lock
rank = int(sys.argv[1]); BL = [int(x) for x in sys.argv[2].split(',')]; NH = int(float(sys.argv[3]) * 1e6); TAG = sys.argv[4]
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load(STORE / Path(man['path']).parent / 'weights.npy')
MC = np.load(P2 / 'fable_covhunter_v1' / f'mcall_rank{rank:04d}_P32M.npz')
out = HERE / f'sym_restart_r{rank}_{TAG}.npz'; assert not out.exists(), out
BATCH, n, TG = 65536, 1024, (13, 14)
dev = torch.device('mps'); Phi = lambda x: 0.5 * (1 + np.vectorize(math.erf)(x / math.sqrt(2)))
res = {}
print(json.dumps(dict(tag='AN-63 sym_restart', rank=rank, births=BL, per_half=NH)), flush=True)
with mps_lock.hold(f'sym_restart r{rank} l={sys.argv[2]} (pid {os.getpid()})'):
    wg = [torch.from_numpy(np.ascontiguousarray(W[l], dtype=np.float32)).to(dev) for l in range(16)]
    t0 = time.time()
    for l in BL:
        if l == 0: mean = np.zeros(n); F = W[0].astype(np.float64)
        else:
            mean = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); C = (C + C.T) / 2
            ev, U = np.linalg.eigh(C); F = (U * np.sqrt(np.clip(ev, 0, None))).T
        sd = np.sqrt(np.clip((F * F).sum(0), 1e-30, None)); al = mean / sd
        Ey = mean * Phi(al) + sd * np.exp(-al ** 2 / 2) / math.sqrt(2 * math.pi)
        Fg = torch.from_numpy(F.astype(np.float32)).to(dev); mg = torch.from_numpy(mean.astype(np.float32)).to(dev)
        Eg = torch.from_numpy(Ey.astype(np.float32)).to(dev)
        for half, seed in (('A', 11000 + 17 * l + rank), ('B', 15000 + 17 * l + rank)):
            torch.manual_seed(seed); S = {v: {L: np.zeros((4, n)) for L in TG} for v in ('full', 'even')}; ref = {}; done = 0
            while done < NH:
                b = min(BATCH, NH - done)
                z = torch.randn((b, n), device=dev) @ Fg + mg
                y = torch.clamp(z, min=0.0)
                s = (torch.randint(0, 2, (b, 1), device=dev).to(torch.float32) * 2 - 1)
                for v, yy in (('full', y), ('even', Eg + s * (y - Eg))):
                    zz = yy
                    for k in range(l + 1, 15):
                        zz = zz @ wg[k]
                        if k in TG:
                            key = (v, k)
                            if key not in ref: ref[key] = zz.mean(0).clone()
                            d = zz - ref[key]; d2 = d * d
                            S[v][k] += torch.stack((d.sum(0), d2.sum(0), (d2 * d).sum(0), (d2 * d2).sum(0))).cpu().numpy().astype(np.float64)
                        if k < 14: zz = torch.clamp(zz, min=0.0)
                done += b
            for v in S:
                for L in TG:
                    if L <= l: continue
                    m1, m2, m3, m4 = S[v][L] / NH
                    res[f'k4_{v}_{half}_{l}_{L}'] = m4 - 4 * m1 * m3 - 3 * m2 ** 2 + 12 * m1 ** 2 * m2 - 6 * m1 ** 4
                    res[f'k3_{v}_{half}_{l}_{L}'] = m3 - 3 * m1 * m2 + 2 * m1 ** 3
            print(json.dumps(dict(l=l, half=half, wall=round(time.time() - t0, 1))), flush=True)
    np.savez(out, **res)
print(json.dumps(dict(done=True, out=str(out), wall=round(time.time() - t0, 1))), flush=True)
