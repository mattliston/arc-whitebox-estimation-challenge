#!/usr/bin/env python3
"""TAG g22-diff-card (T-124): a DIFFERENTIABLE Gaussian card in torch — the mean + full-covariance recursion of a bias-free ReLU MLP
under x ~ N(0, I), with the card's own Mehler/Hermite pair map (37 terms) for the off-diagonal post second moments and the exact
univariate raw2 on the diagonal. Milestone M1: reproduce the fnp Gaussian card's open-loop per-layer post means on one net.
  python diff_card.py m1 <rank>      -> per-layer relative error vs dbgdump_gaussopen_retro-<rank>.npz and vs retro_out_gauss_open_rank<rank>.npy
The recursion is a plain torch function (gauss_forward) so a compensator can be trained through it (M2, train_diff_card.py)."""
import json, math, os, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
NHERM = int(os.environ.get("DC_NHERM", "37"))          # Mehler terms (the card uses He_0..He_36)
VAR_FLOOR = 1e-30
TRANSPOSE = os.environ.get("DC_TRANSPOSE", "1") == "1"   # weights stored as (in, out): pre = post @ W
SQRT2PI = math.sqrt(2.0 * math.pi)


def load_weights(rank):
    man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
    return np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")        # (16, 1024, 1024), layer l maps post_{l-1} -> pre_l


def hermite_table(x, n):
    """He_0..He_{n-1} at x (probabilists'), shape (n, N)."""
    out = [torch.ones_like(x), x]
    for k in range(1, n - 1):
        out.append(x * out[-1] - k * out[-2])
    return torch.stack(out[:n], dim=0)


def relu_gauss_layer(mu, Sigma):
    """Post mean and post covariance of relu(z), z ~ N(mu, Sigma). Off-diagonal via the Mehler series, diagonal exact."""
    var = torch.clamp(torch.diagonal(Sigma), min=VAR_FLOOR); sigma = torch.sqrt(var); alpha = mu / sigma
    pdf = torch.exp(-0.5 * alpha * alpha) / SQRT2PI; cdf = 0.5 * (1.0 + torch.erf(alpha / math.sqrt(2.0)))
    m = sigma * pdf + mu * cdf                                                   # raw1
    raw2 = (mu * mu + var) * cdf + mu * sigma * pdf
    # c_n = E[relu(z) He_n(zhat)] : c_0 = m, c_1 = sigma*cdf, c_n = sigma*pdf*He_{n-2}(-alpha); Mehler weight 1/n!
    He = hermite_table(-alpha, NHERM - 2)                                        # He_0..He_{NHERM-3}
    coef = [m, sigma * cdf] + [sigma * pdf * He[k] for k in range(NHERM - 2)]
    rho = Sigma / (sigma[:, None] * sigma[None, :])
    rho = torch.clamp(rho, -1.0, 1.0)
    M11 = torch.zeros_like(Sigma); rpow = torch.ones_like(Sigma); fact = 1.0
    for n in range(NHERM):
        if n > 0:
            rpow = rpow * rho; fact *= n
        cn = coef[n] / math.sqrt(fact)
        M11 = M11 + torch.outer(cn, cn) * rpow
    C = M11 - torch.outer(m, m)
    d = raw2 - m * m                                                             # exact diagonal
    C = C - torch.diag(torch.diagonal(C)) + torch.diag(d)
    return m, C, (sigma, alpha, pdf, cdf, raw2)


def gauss_forward(W, compensate=None, dtype=torch.float64, device="cpu"):
    """W: (16, n, n) numpy. Returns the list of per-layer post means (and covariances) through the Gaussian recursion.
    compensate(l, m, C, aux) -> (m, C) is the hook for a trained compensator (None = the plain card)."""
    n = W.shape[1]
    m = torch.zeros(n, dtype=dtype, device=device); C = torch.eye(n, dtype=dtype, device=device)
    means = []; covs = []
    for l in range(W.shape[0]):
        Wl = torch.as_tensor(np.asarray(W[l], dtype=np.float64), dtype=dtype, device=device)
        if TRANSPOSE:
            mu = Wl.T @ m; Sigma = Wl.T @ C @ Wl                                  # row-vector convention: pre = post @ W
        else:
            mu = Wl @ m; Sigma = Wl @ C @ Wl.T
        m, C, aux = relu_gauss_layer(mu, Sigma)
        if compensate is not None:
            m, C = compensate(l, m, C, aux)
        means.append(m); covs.append(C)
    return means, covs


if __name__ == "__main__":
    mode, rank = sys.argv[1], int(sys.argv[2])
    W = load_weights(rank)
    t0 = __import__("time").time()
    with torch.no_grad():
        means, covs = gauss_forward(W)
    print(f"torch Gaussian card forward: {__import__('time').time() - t0:.1f}s (float64 cpu, {NHERM} Mehler terms)")
    z = np.load(HERE / f"dbgdump_gaussopen_retro-{rank}.npz")
    truth = np.load(HERE / "truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    for l in range(16):
        mt = means[l].numpy(); key = f"post_mean_{l}"
        if key in z.files:
            mc = z[key].astype(np.float64); rel = np.sqrt(np.mean((mt - mc) ** 2) / np.mean(mc ** 2))
            vt = np.diag(covs[l].numpy()); vc = z[f"post_var_{l}"].astype(np.float64); relv = np.sqrt(np.mean((vt - vc) ** 2) / np.mean(vc ** 2))
            print(f"L{l:2d}: torch vs fnp card mean rel {rel:.2e}  var rel {relv:.2e} | torch vs truth mse {np.mean((mt - truth[l]) ** 2):.2e}  fnp vs truth {np.mean((mc - truth[l]) ** 2):.2e}")
        else:
            print(f"L{l:2d}: (no dump) torch vs truth mse {np.mean((mt - truth[l]) ** 2):.2e}")
    out = P2 / "public_g20_emission_retrofit_v1" / f"retro_out_gauss_open_rank{rank:04d}.npy"
    if out.exists():
        pred = np.load(out).astype(np.float64).reshape(-1)[:1024]; mt = means[-1].numpy()   # retro_out holds [mean, var]
        print(f"FINAL: torch vs fnp prediction rel {np.sqrt(np.mean((mt - pred) ** 2) / np.mean(pred ** 2)):.2e}; torch final mse vs truth {np.mean((mt - truth[-1]) ** 2):.3e}; fnp {np.mean((pred - truth[-1]) ** 2):.3e}")
