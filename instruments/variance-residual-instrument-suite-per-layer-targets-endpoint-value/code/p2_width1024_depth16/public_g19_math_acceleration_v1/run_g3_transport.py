#!/usr/bin/env python3
"""TAG g3-transport: transported-birth accumulation vs exact carried T.

The G3-supply architecture's composition test (tiny world, exact
references): T-hat(l+1)_c = sum_i W_ic [pz_i(l) T-hat(l)_i +
birth_i(l)], T-hat(1)=0, with pz = E[phi'] (CF-exact) and birth from
the certified G3-1 words. Graded per layer against the exact carried
T (pre-coef order-3 rows from the ceiling saves). PASS at the
perturbative regime => the deployable G3-Gram supply (identify ->
birth -> D-transport -> Gram) is verified end-to-end in miniature.
Usage: SA=.45 SM=.04 SMU=.15 run_g3_transport.py 21
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_word_closure import MI, IDX3, FACT, MULT, tens_to_coef, r, n
from run_g3_birth_words import birth_words, cf_scalars


def coef_to_dense(coef, idx):
    T = np.zeros((r, r, r))
    for j, f in enumerate(idx):
        a = MI[f]
        ids = []
        for d in range(r):
            ids += [d] * int(a[d])
        val = coef[j] / (MULT[f] * FACT[f])
        import itertools as it
        for pm in set(it.permutations(ids)):
            T[pm] = val
    return T

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 21
DEPTH = 8


def state_at(pre):
    mv = pre[0, :]
    A = pre[1:1 + r, :].T
    Mt = np.zeros((n, r, r))
    for f in range(1 + r, 1 + r + r * (r + 1) // 2):
        ids = np.where(MI[f] > 0)[0]
        row = pre[f, :]
        if len(ids) == 1:
            Mt[:, ids[0], ids[0]] = row * math.sqrt(2)
        else:
            Mt[:, ids[0], ids[1]] = row
            Mt[:, ids[1], ids[0]] = row
    return mv, A, Mt


def main():
    rng = np.random.default_rng(SEED)
    A0 = rng.standard_normal((n, r)) * float(os.environ.get("SA", "0.3"))
    M0 = rng.standard_normal((n, r, r)) * float(os.environ.get("SM", "0.25"))
    M0 = 0.5 * (M0 + np.transpose(M0, (0, 2, 1)))
    m0 = rng.standard_normal(n) * float(os.environ.get("SMU", "0.1"))
    Ws = [rng.standard_normal((n, n)) * np.sqrt(2.0 / n)
          for _ in range(DEPTH)]
    That = np.zeros((n, len(IDX3)))       # carried model T (coef space)
    out = {}
    for l in range(1, 7):
        if l == 1:
            mv, A, Mt = m0, A0, M0
        else:
            pre = np.load(HERE / f"ceil_coef_seed{SEED}_L{l - 1}.npy"
                          ).astype(np.float64) @ Ws[l - 2]
            mv, A, Mt = state_at(pre)
            y_true = pre[IDX3, :].T           # exact carried T coefs
            num = np.einsum("ck,ck->c", That, y_true)
            r2 = 1 - ((That - y_true) ** 2).sum(1) \
                / np.maximum((y_true ** 2).sum(1), 1e-30)
            beta = num / np.maximum((That ** 2).sum(1), 1e-30)
            out[l] = {"R2_med": round(float(np.median(r2)), 4),
                      "R2_p10": round(float(np.percentile(r2, 10)), 4),
                      "beta_med": round(float(np.median(beta)), 4)}
            print(f"preL{l}", json.dumps(out[l]), flush=True)
        # emission: passthrough + birth, then mix
        from run_g3_exact_emission import exact_T
        post = np.zeros_like(That)
        for c in range(n):
            lam, U = np.linalg.eigh(Mt[c])
            at = U.T @ A[c]
            pz, p0, p1, p2 = cf_scalars(float(mv[c]), lam, at)
            Tex, d1 = exact_T(float(mv[c]), A[c], Mt[c])
            post[c] = pz * That[c] + tens_to_coef(Tex / 6.0, IDX3)
        cfl = np.load(HERE / f"ceil_coef_seed{SEED}_L{l}.npy"
                      ).astype(np.float64)
        ye = cfl[IDX3, :].T
        r2e = 1 - ((post - ye) ** 2).sum(1) \
            / np.maximum((ye ** 2).sum(1), 1e-30)
        print(f"  emission L{l}: R2_med {np.median(r2e):.4f} "
              f"p10 {np.percentile(r2e, 10):.4f}", flush=True)
        That = Ws[l - 1].T @ post
        if "--save" in sys.argv:
            np.save(HERE / f"g3that_seed{SEED}_preL{l + 1}.npy", That)
    (HERE / f"G3_TRANSPORT_seed{SEED}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
