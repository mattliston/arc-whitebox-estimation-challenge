#!/usr/bin/env python3
"""Mechanistic per-unit features from the card's internal state, as a callable for the round-3 row generator.

The card (public_g20_emission_retrofit_v1/estimator_retro.py) records its internal state per layer when
RETRO_DUMP_FEAT=1 is set BEFORE the module is executed; at the end of predict_with_native_tape it writes
<RETRO_DUMP_FEAT_PATH>_<mlp.name>.npz (default prefix "featdump") and clears the dict.  The features at layer_index l
are read BEFORE any forcing/table at layer l, so on the two-table host they describe the state produced by the
tables applied at layers < l.  Layer_index 15 (the final read) has no dump; layer_index 1 has no previous-layer
post quantities (those columns are 0).

mechfeat_from_run(feat, weights) -> float32 (16, 1024, F=31):
    feat    = the npz (np.load(...)) or a dict with the dump's keys;
    weights = list/array of the 16 weight matrices W[l] (1024, 1024), column i = incoming weights of unit i
              (the same convention as norm(W[l], axis=0) = the dump's w_in_norm_l).
    row l (1..14) = layer_index l; rows 0 and 15 are zero (see LAYER_MASK).
Column list: FEATURE_NAMES (exactly the families of ridge_study.py: F0 univariate part [alpha, g3, v, log w_in],
Fa CP3 [cp3_diag, age0..age3, older, star_diag, log1p(1e6 cp3_energy), K21 rownorm, K21 colnorm], Fb CP4 [cp4_d, cp4 K31/K22
rownorm, pre_k4, cp4 inherited/birth/dropped d of layer l-1, K22 rownorm, K31 rownorm], Fc pair/weights [w_in, gate_prev@W,
gate_prev@W^2, gate_prev^2@W^2], Fd pushed previous-layer cumulants [W^3'pk3, W^4'pk4, W^2'pv, W^3'(pk3*gate_prev)]).
raw20 (the tape) is NOT included: the row generator already has it.
"""
import numpy as np

FEATURE_NAMES = (
    ["alpha", "g3", "v", "log_w_in"]
    + ["cp3_diag", "cp3_age0", "cp3_age1", "cp3_age2", "cp3_age3", "cp3_older", "star_diag", "log1p_cp3_energy", "K21_rownorm", "K21_colnorm"]
    + ["cp4_d", "cp4_K31_rownorm", "cp4_K22_rownorm", "pre_k4", "cp4_inherited_d_prev", "cp4_k4_birth_prev", "cp4_dropped_d_prev", "K22_rownorm", "K31_rownorm"]
    + ["w_in", "gate_prev_W", "gate_prev_W2", "gate_prev2_W2"]
    + ["W3_pk3_prev", "W4_pk4_prev", "W2_pv_prev", "W3_pk3gate_prev"]
)
F = len(FEATURE_NAMES)
LAYER_MASK = np.array([False] + [True] * 14 + [False])   # rows with features


def _get(feat, key, default=None):
    files = feat.files if hasattr(feat, "files") else feat.keys()
    if key in files:
        return np.asarray(feat[key], dtype=float)
    return default


def mechfeat_from_run(feat, weights):
    out = np.zeros((16, 1024, F), dtype=np.float32)
    for l in range(1, 15):
        if _get(feat, f"cp3_diag_{l}") is None:
            continue
        m = _get(feat, f"pre_m_{l}"); v = _get(feat, f"pre_v_{l}")
        k3 = _get(feat, f"k3_carried_{l}")
        if k3 is None:   # dumps written before round 3 have no k3_carried key: the carried k3 is the CP3 read + the star's one-step diagonal
            k3 = _get(feat, f"cp3_diag_{l}") + _get(feat, f"star_diag_{l}")
        Wl = np.asarray(weights[l], dtype=float); win = np.linalg.norm(Wl, axis=0)
        gate_prev = _get(feat, f"gate_{l - 1}", np.ones(1024))
        blocks = _get(feat, f"cp3_diag_blocks_{l}"); nb = l
        age = np.zeros((4, 1024))
        for a in range(4):
            b = nb - 1 - a
            if b >= 0:
                age[a] = blocks[b]
        older = blocks[:max(nb - 4, 0)].sum(0) if nb > 4 else np.zeros(1024)
        z = np.zeros(1024)
        pk3 = _get(feat, f"post_k3_target_{l - 1}", z); pk4 = _get(feat, f"post_k4_target_{l - 1}", z); pv = _get(feat, f"post_var_{l - 1}", z)
        cols = [
            m / np.sqrt(v), k3 / v ** 1.5, v, np.log(win),
            _get(feat, f"cp3_diag_{l}"), age[0], age[1], age[2], age[3], older, _get(feat, f"star_diag_{l}"),
            np.log1p(_get(feat, f"cp3_energy_{l}") * 1e6), _get(feat, f"K21_rownorm_{l}"), _get(feat, f"K21_colnorm_{l}"),
            _get(feat, f"cp4_d_{l}"), _get(feat, f"cp4_K31_rownorm_{l}"), _get(feat, f"cp4_K22_rownorm_{l}"), _get(feat, f"pre_k4_{l}"),
            _get(feat, f"cp4_inherited_d_{l - 1}", z), _get(feat, f"cp4_k4_birth_{l - 1}", z), _get(feat, f"cp4_dropped_d_{l - 1}", z),
            _get(feat, f"K22_rownorm_{l}"), _get(feat, f"K31_rownorm_{l}"),
            win, gate_prev @ Wl, gate_prev @ (Wl ** 2), (gate_prev ** 2) @ (Wl ** 2),
            (Wl ** 3).T @ pk3, (Wl ** 4).T @ pk4, (Wl ** 2).T @ pv, (Wl ** 3).T @ (pk3 * gate_prev),
        ]
        out[l] = np.column_stack(cols).astype(np.float32)
    assert np.isfinite(out).all()
    return out


if __name__ == "__main__":
    # CERTIFICATE (net 35): (1) a fresh dump on the ORIGINAL host reproduces mechfeat_retro-35.npz key by key (bit-for-bit
    # expected: same code path, the round-3 gates inert); (2) the feature matrix built from both is identical; (3) the new
    # k3_carried key equals the G26 card_k3 row l-1 and w_in_norm equals norm(W[l], axis=0); (4) the two-table dump exists and
    # its host MSE (from the screen log) is reported against the additive-table reference 4.123e-8.
    import json, sys
    from pathlib import Path
    HERE = Path(__file__).resolve().parent; P2 = HERE.parent
    STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
    man = json.loads((STORE / "manifest.json").read_text())
    e = man["roles"]["observer_fit"]["row_manifests"][35]
    W = np.load((STORE / e["path"]).parent / "weights.npy", mmap_mode="r")
    ref = np.load(HERE / "mechfeat_retro-35.npz")
    fresh_path = HERE / (sys.argv[1] if len(sys.argv) > 1 else "cert_orig_retro-35.npz")
    fresh = np.load(fresh_path)
    worst = 0.0; nbit = 0; nkeys = 0
    for k in ref.files:
        a, b = ref[k], fresh[k]; nkeys += 1
        if np.array_equal(a, b):
            nbit += 1; continue
        rel = float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) + 1e-300)); worst = max(worst, rel)
    print(f"(1) keys {nkeys}: bit-identical {nbit}; worst max-abs rel deviation among the rest {worst:.3e}; extra keys in the fresh dump: {sorted(set(fresh.files) - set(ref.files))[:3]} ...")
    Xr = mechfeat_from_run(ref, W); Xf = mechfeat_from_run(fresh, W)
    print(f"(2) feature matrix {Xf.shape}: max |Xf - Xr| / max|Xr| per column worst = {float(np.max(np.abs(Xf - Xr) / (np.abs(Xr).max(axis=(0, 1)) + 1e-300))):.3e}; array_equal {np.array_equal(Xf, Xr)}")
    G = np.load(P2 / "public_g26_k3_corrector_features_v1/features_rank0035.npz")
    r3 = max(float(np.linalg.norm(fresh[f"k3_carried_{l}"] - G["card_k3"][l - 1]) / np.linalg.norm(G["card_k3"][l - 1])) for l in range(1, 15))
    rw = max(float(np.max(np.abs(fresh[f"w_in_norm_{l}"] - np.linalg.norm(np.asarray(W[l], dtype=float), axis=0)))) for l in range(1, 15))
    print(f"(3) k3_carried vs G26 card_k3 (row l-1): worst rel {r3:.3e}; w_in_norm vs norm(W[l],axis=0): max abs {rw:.3e}")
    tt = HERE / "cert_twotable_retro-35.npz"
    if tt.exists():
        T = np.load(tt); Xt = mechfeat_from_run(T, W)
        print(f"(4) two-table dump: {len(T.files)} keys, matrix {Xt.shape} finite {np.isfinite(Xt).all()}; k3_carried L8 rms ratio two-table/original = {float(np.sqrt((T['k3_carried_8']**2).mean() / (fresh['k3_carried_8']**2).mean())):.4f} (expected ≈ 1/s_l products > 1)")
