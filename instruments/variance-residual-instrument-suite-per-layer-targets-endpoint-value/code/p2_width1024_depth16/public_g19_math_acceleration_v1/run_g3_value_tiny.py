#!/usr/bin/env python3
"""TAG g3-value-tiny: does the supplied G3-Gram close the closure's
residual? (the value gate in miniature, perturbative world)
Rt = truth-post-cov - closure(truth pre-stats); supply candidates:
G3hat (my transported T-hat Gram), G3exact (truth H3 projection),
each minus M3 (the closure's own order-3 content by gating).
Usage: SA=.. SM=.. SMU=.. run_g3_value_tiny.py 21 [layers]
"""
import os, sys, math, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "public_g19_attribution_v1"))
from run_word_closure import MI, IDX3, r, n
from run_chaos_ceiling import he_features, multi_indices
from run_vresid_field import engine_post_cov

_args = [a for a in sys.argv[1:] if a.isdigit()]
SEED = int(_args[0]) if _args else 21
LAYERS = [int(x) for x in _args[1:]] or [4, 6]
N, CH = 2_000_000, 200_000

def main():
    rng = np.random.default_rng(SEED)
    A0 = rng.standard_normal((n, r)) * float(os.environ.get("SA", "0.3"))
    M0 = rng.standard_normal((n, r, r)) * float(os.environ.get("SM", "0.25"))
    M0 = 0.5 * (M0 + np.transpose(M0, (0, 2, 1)))
    m0 = rng.standard_normal(n) * float(os.environ.get("SMU", "0.1"))
    Ws = [rng.standard_normal((n, n)).astype(np.float32) * np.float32(np.sqrt(2.0 / n)) for _ in range(8)]
    g = rng.standard_normal((N, r)).astype(np.float32)
    Z = (m0[None, :].astype(np.float32) + g @ A0.T.astype(np.float32)
         + 0.5 * (np.einsum("na,iab,nb->ni", g, M0.astype(np.float32), g, optimize=True)
                  - np.einsum("iaa->i", M0)[None, :].astype(np.float32)))
    mi = multi_indices(r, 4)
    fact = np.array([np.sqrt(np.prod([math.factorial(int(o)) for o in a])) for a in mi], dtype=np.float32)
    orders = mi.sum(1); i3 = np.where(orders == 3)[0]
    G = np.zeros((len(i3), len(i3)))
    for s in range(0, N, CH):
        ft = he_features(g[s:s + CH], mi[i3], fact[i3])
        G += (ft.T @ ft).astype(np.float64)
    G /= N
    out = {}
    for l in range(1, max(LAYERS) + 1):
        if l in LAYERS:
            zm = Z.mean(0).astype(np.float64); zc = (Z - zm).astype(np.float32)
            C = (zc.T @ zc).astype(np.float64) / N
            v = np.diag(C).copy()
            k3 = ((zc ** 3).mean(0)).astype(np.float64)
            k4 = ((zc ** 4).mean(0)).astype(np.float64) - 3 * v ** 2
            K21 = ((zc * zc).T @ zc).astype(np.float64) / N
            z2 = (zc * zc).astype(np.float32)
            K22 = (z2.T @ z2).astype(np.float64) / N - np.outer(v, v) - 2 * C ** 2
            K31 = ((zc * z2).T @ zc).astype(np.float64) / N - 3 * v[:, None] * C
            ph = np.maximum(Z, 0.0)
            pm = ph.mean(0).astype(np.float64); pc = (ph - pm).astype(np.float32)
            Tpost = (pc.T @ pc).astype(np.float64) / N
            z0 = np.zeros_like
            Mfull = engine_post_cov(zm, v, k3, k4, C, K21, K22, K31)
            Mno3 = engine_post_cov(zm, v, z0(k3), k4, C, z0(K21), K22, K31)
            M3 = Mfull - Mno3
            Rt = Tpost - Mfull
            # the closure's own Mehler t=3 copula term (surgical target)
            from run_tensor_recursion_v4 import columns
            c1cols, c2cols, sdv, pdfv, cdfv = columns(zm, v)
            rhoz = np.clip(C / np.outer(np.sqrt(v), np.sqrt(v)),
                           -0.9999, 0.9999)
            T3meh = (rhoz ** 3 / 6.0) * np.outer(c1cols[3], c1cols[3])
            # exact G3 of the POST activations (projection)
            b = np.zeros((len(i3), n))
            for s in range(0, N, CH):
                ft = he_features(g[s:s + CH], mi[i3], fact[i3])
                b += (ft.T @ ph[s:s + CH]).astype(np.float64)
            b /= N
            cf3 = np.linalg.solve(G, b)
            G3ex = cf3.T @ cf3
            # my supplied G3: T-hat of the POST at layer l =
            # pre-coef of layer l+1 un-mixed... use carried pre at l:
            # the Gram that feeds Cov(post) is the POST T-emission;
            # approximate with the carried pre at l+1 pulled back? use
            # the saved carried T-hat at preL(l+1) mixed-forward
            # equivalence: G3(post) = Gram of post-emission = the
            # carried preL(l+1) T-hat back-rotated... simplest честно:
            # recompute post T-hat = (W^T)^-1 That(l+1)? W not orth.
            # Instead load That(pre l) and apply ONE emission+no-mix
            # step is what transport did before mixing; rerun that
            # piece here: load pre-l T-hat and state, emit:
            from run_g3_transport import state_at
            from run_g3_exact_emission import exact_T
            from run_g3_birth_words import cf_scalars
            from run_word_closure import tens_to_coef as t2c
            if l == 1:
                mv, A, Mt = m0, A0, M0
                Tc = np.zeros((n, len(IDX3)))
            else:
                pre = np.load(HERE / f"ceil_coef_seed{SEED}_L{l - 1}.npy").astype(np.float64) @ Ws[l - 2].astype(np.float64)
                mv, A, Mt = state_at(pre)
                Tc = np.load(HERE / f"g3that_seed{SEED}_preL{l}.npy")
            postT = np.zeros((n, len(IDX3)))
            for c in range(n):
                lam, U = np.linalg.eigh(Mt[c])
                at = U.T @ A[c]
                pz, p0, p1, p2 = cf_scalars(float(mv[c]), lam, at)
                Tex, d1 = exact_T(float(mv[c]), A[c], Mt[c])
                postT[c] = pz * Tc[c] + t2c(Tex / 6.0, IDX3)
            G3hat = postT @ postT.T
            Wp = Ws[l].astype(np.float64) if l < 8 else None
            def push(M):
                return np.einsum("ac,ab,bc->c", Wp, M, Wp, optimize=True)
            nR = np.linalg.norm(Rt)
            pR = push(Rt)
            rec = {}
            for tag, Gx in (("G3hat", G3hat), ("G3exact", G3ex),
                            ("swap_hat", G3hat), ("swap_ex", G3ex)):
                D = (Gx - T3meh) if tag.startswith("swap") else (Gx - M3)
                rec[f"{tag}_corr"] = round(float(np.dot(D.ravel(), Rt.ravel()) / max(np.linalg.norm(D) * nR, 1e-30)), 3)
                rec[f"{tag}_frob_closure"] = round(1 - float(np.linalg.norm(Rt - D) / nR), 3)
                rec[f"{tag}_pushedv_closure"] = round(1 - float(np.sqrt(np.mean(push(Rt - D) ** 2)) / np.sqrt(np.mean(pR ** 2))), 3)
            out[l] = rec
            print(f"L{l}", json.dumps(rec), flush=True)
        Z = np.maximum(Z, 0.0) @ Ws[l - 1]
    (HERE / f"G3_VALUE_seed{SEED}.json").write_text(json.dumps(out, indent=1) + "\n")

if __name__ == "__main__":
    main()
