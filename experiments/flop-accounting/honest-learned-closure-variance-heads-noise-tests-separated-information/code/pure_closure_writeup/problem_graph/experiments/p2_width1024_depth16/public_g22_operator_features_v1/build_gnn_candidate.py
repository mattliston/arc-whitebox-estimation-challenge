#!/usr/bin/env python3
"""TAG g22-build-gnn-candidate (T-114j): the deployed D11 card (estimator_cap.py) + the LEARNED CLOSURE OVER THE WEIGHT
GRAPH run closed-loop inside the recursion: at every layer the 31 mechanistic features are computed in flopscope from the
card's live objects (_mech_fnp), the GNN step (_gnn_step) produces the per-neuron mean correction, and the correction is
carried into the recursion (mean + the consistent covariance C <- C - (m d^T + d m^T) - d d^T). The output rows are the
carried corrected means; the open-loop patch is removed. Both fnp functions are copied verbatim from the certified
instrument (estimator_retro.py, T-114c/T-114i); the GNN payload is embedded as a literal (numpy-free).
Usage: build_gnn_candidate.py <payload.npz> <out_candidate.py>"""
import hashlib, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_d11_util_halving_v1/estimator_cap.py"
RETRO = P2 / "public_g20_emission_retrofit_v1/estimator_retro.py"
PAYLOAD = Path(sys.argv[1]); OUT = Path(sys.argv[2])


def replace_once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:90])
    return s.replace(old, new, 1)


def extract(src, name):
    """the full text of a top-level function from the instrument (up to the next top-level def / assignment)"""
    m = re.search(rf"^def {name}\(.*?(?=^\S)", src, flags=re.S | re.M)
    assert m, name
    return m.group(0)


def main():
    s = BASE.read_text(encoding="utf-8"); r = RETRO.read_text(encoding="utf-8")
    mech_fn = extract(r, "_mech_fnp"); gnn_fn = extract(r, "_gnn_step")
    # the candidate's GNN step reads the embedded literal, not an env file
    gnn_fn = replace_once(gnn_fn, '''        _z = _np_retro.load(_os.environ["RETRO_GNN_PAYLOAD"]); _GNN["P"] = {k: fnp.asarray(_z[k], dtype=fnp.float32) for k in _z.files}''',
                          '''        _GNN["P"] = {k: fnp.asarray(v, dtype=fnp.float32) for k, v in _GNN_PAYLOAD.items()}''')
    assert "_os.environ" not in gnn_fn and "_np_retro" not in gnn_fn and "_np_retro" not in mech_fn and "_os.environ" not in mech_fn
    z = np.load(PAYLOAD)
    literal = "_GNN_PAYLOAD = {\n" + "".join(f"    {k!r}: {np.asarray(z[k], dtype=np.float32).tolist()!r},\n" for k in z.files) + "}\n"
    import os as _os_b
    final_push = _os_b.environ.get("BUILD_FINAL_PUSH") == "1"   # T-115d: the payload was trained with the pushed final-layer plane columns
    vhead_mix = float(_os_b.environ.get("BUILD_VHEAD_MIX", "0.0"))   # T-116o variance-head mix; T-130: DEFAULT 0 - the instrument applies the head only under RETRO_CL_VHEAD=1, and every lean payload (HV lineage) carries one; a 0.5 default silently applied it in cand_gnn_hcap256/hnq/lean2 (the "packaging divergence")
    block = ("\n\n# ---- T-114j: the learned closure over the weight graph (closed-loop, fnp, certified against the instrument) ----\n"
             "_MECH = {}\n_GNN = {}\n" + f"_RETRO_FINAL_PUSH = {final_push!r}   # T-115d pushed plane columns at the final read\n" + f"_VHEAD_MIX = {vhead_mix!r}   # T-116o\n" + literal + "\n\n" + mech_fn + "\n\n" + gnn_fn + "\n")
    s = replace_once(s, "\n\ndef _predict_corrected(mlp: MLP):\n", block + "\ndef _predict_corrected(mlp: MLP):\n")
    if _os_b.environ.get("BUILD_F32_COV") == "1":   # T-138: bake the float32 covariance push (F32_COV=1: same accuracy, -9% util)
        s = replace_once(s, '        if _os.environ.get("F32_COV") == "1":\n', '        if True:   # T-138 baked F32_COV\n')
    if _os_b.environ.get("BUILD_F32_PUSH34") == "1":   # T-138: float32 cubic/quartic pushes (cast directed/diagonal at the 4 push sites)
        _anchor = "    directed = _off_diagonal(star_h, diagonal_context)\n"
        assert s.count(_anchor) == 4, s.count(_anchor)
        s = s.replace(_anchor, _anchor + "    directed = fnp.asarray(directed, dtype=fnp.float32); diagonal = fnp.asarray(diagonal, dtype=fnp.float32)   # T-138 baked F32_PUSH34\n")
    cp4cap = _os_b.environ.get("BUILD_CP4_CAP")   # T-118c: bake a cheaper fourth-order cap into the candidate (no env at deploy)
    if cp4cap:
        s = replace_once(s, "_CP4_CAP = 2048\n", f"_CP4_CAP = {int(cp4cap)}   # T-118c baked cap\n")
    cp3cap = _os_b.environ.get("BUILD_CP3_CAP")   # T-118g: bake a lower third-order cap
    if cp3cap:
        s = replace_once(s, '_CP3_CAP = int(_os.environ.get("CP3_CAP", "16384"))', f'_CP3_CAP = {int(cp3cap)}   # T-118g baked cap')
    if _os_b.environ.get("BUILD_K21_ZERO") == "1":   # T-121: the pure Gaussian card (only if the payload was co-adapted to it)
        s = replace_once(s, "        pre_k21 = cp_pre_k21 + cubic_pre_k21\n", "        pre_k21 = fnp.zeros((width, width), dtype=fnp.float64)   # T-121 no third order\n")
    if _os_b.environ.get("BUILD_K21_DIAGONLY") == "1":   # T-120: the diagonal card (only if the payload was co-adapted to it)
        s = replace_once(s, "        pre_k21 = cp_pre_k21 + cubic_pre_k21\n", "        pre_k21 = fnp.diag(fnp.diag(fnp.asarray(cp_pre_k21 + cubic_pre_k21, dtype=fnp.float64)))   # T-120 K21 diag-only\n")
    if _os_b.environ.get("BUILD_NO_CUBIC") == "1":   # T-118h: skip the cubic star push (only if the payload was co-adapted to it)
        s = replace_once(s, "        cubic_pre_k21 = _push_cubic(\n            cubic, weight_context, diagonal_context, fusion_context\n        )\n",
                         "        cubic_pre_k21 = fnp.zeros((width, width), dtype=fnp.float64)   # T-118h no cubic star push\n")
    noq = _os_b.environ.get("BUILD_NO_QUARTIC") == "1"   # T-118f: skip the dense quartic push (only if the payload was co-adapted to it)
    if noq:
        s = replace_once(s, "        pre_k4, pre_k31, pre_k22 = _assemble_quartic_provider(\n            _push_quartic(\n",
                         "        pre_k4, pre_k31, pre_k22 = _assemble_quartic_provider(\n            tuple(fnp.zeros_like(fnp.asarray(_pp, dtype=fnp.float64)) for _pp in cp4_pre_provider_planes) if True else _push_quartic(\n")
    # CP4 reset: stash the pieces for the next layer's features (unconditional; copies)
    s = replace_once(s, "    stored_birth = fnp.asarray(birth_lambda, dtype=fnp.float64)\n",
                     "    stored_birth = fnp.asarray(birth_lambda, dtype=fnp.float64)\n"
                     "    _MECH[\"cp4_inherited_d\"] = fnp.array(inherited_d, dtype=fnp.float64); _MECH[\"cp4_k4_birth\"] = fnp.array(stored_birth, dtype=fnp.float64); _MECH[\"cp4_dropped_d\"] = fnp.zeros((int(k4_target.shape[0]),), dtype=fnp.float64)\n")
    s = replace_once(s, "    dropped_planes = _cp4_read_planes(\n", "    dropped_planes = _cp4_read_planes(\n")   # anchor exists
    # after dropped_planes is computed, stash its diagonal: find the assignment block end (the next line starting with 4 spaces and 'retained')
    m = re.search(r"    dropped_planes = _cp4_read_planes\(.*?\n    \)\n", s, flags=re.S)
    assert m
    s = s[:m.end()] + "    _MECH[\"cp4_dropped_d\"] = fnp.array(dropped_planes[0], dtype=fnp.float64)\n" + s[m.end():]
    # interior layers: features -> step -> feedback -> stash
    s = replace_once(s, "            native_sigma_rows.append(sigma)\n\n        gate = gaussian_raw[0]\n",
                     "            native_sigma_rows.append(sigma)\n"
                     "            # T-114j closed loop: fnp features, GNN step, correction carried into mean + covariance\n"
                     "            _mech = _mech_fnp(layer_index, pre_mean, pre_variance, pre_k21, pre_k4, pre_k22, pre_k31, fnp.diag(cp_pre_k21), fnp.diag(cubic_pre_k21),\n"
                     "                              cp4_pre_provider_planes[0], cp4_pre_provider_planes[1], cp4_pre_provider_planes[2], cp_lambda, pushed_p, pushed_q, weight32, width)\n"
                     "            _rowin = fnp.concatenate((fnp.asarray(native_raw_rows[-1], dtype=fnp.float32), fnp.asarray(_mech, dtype=fnp.float32)), axis=1)\n"
                     "            _GNN[\"W_next\"] = fnp.asarray(mlp.weights[layer_index + 1], dtype=fnp.float32) if layer_index + 1 < len(mlp.weights) else None   # T-116o\n"
            "            _dl = fnp.asarray(_gnn_step(layer_index, _rowin, sigma, weight32), dtype=fnp.float64)\n"
                     "            _mc = fnp.asarray(mean, dtype=fnp.float64)\n"
                     "            covariance = fnp.asarray(covariance, dtype=fnp.float64) - fnp.outer(_mc, _dl) - fnp.outer(_dl, _mc) - fnp.outer(_dl, _dl)\n"
                     "            mean = fnp.asarray(_mc + _dl, dtype=fnp.float64)\n"
                     "            rows[-1] = mean\n"
                     "            _MECH[\"gate\"] = fnp.array(gaussian_raw[0], dtype=fnp.float64); _MECH[\"post_k3\"] = fnp.array(fnp.diag(k21_target), dtype=fnp.float64)\n"
                     "            _MECH[\"post_k4\"] = fnp.array(k4_target, dtype=fnp.float64); _MECH[\"post_var\"] = fnp.array(repaired_variance, dtype=fnp.float64)\n\n"
                     "        gate = gaussian_raw[0]\n")
    # final layer: split the reads and apply the step on the final row
    s = replace_once(s, """            final_k3 = (
                _cp_k3(cp_lambda, pushed_p, pushed_q, cp3_context[3])
                + _push_cubic_diagonal(
                    cubic, weight_context, diagonal_context, fusion_context
                )
            )
            final_k4 = (
                _push_quartic_diagonal(
                    quartic,
                    pairing_alpha,
                    weight_context,
                    pre_covariance,
                    pre_variance,
                    diagonal_context,
                    fusion_context,
                )
                + _cp4_read_diagonal(cp4_lambda, pushed_u)
            )
""", """            _fk3_cp = _cp_k3(cp_lambda, pushed_p, pushed_q, cp3_context[3])
            _fk3_star = _push_cubic_diagonal(cubic, weight_context, diagonal_context, fusion_context)
            final_k3 = _fk3_cp + _fk3_star
            _fk4_star = _push_quartic_diagonal(
                quartic,
                pairing_alpha,
                weight_context,
                pre_covariance,
                pre_variance,
                diagonal_context,
                fusion_context,
            )
            _fk4_cp4 = _cp4_read_diagonal(cp4_lambda, pushed_u)
            final_k4 = _fk4_star + _fk4_cp4
""")
    s = replace_once(s, "                native_sigma_rows.append(sigma)\n            break\n",
                     "                native_sigma_rows.append(sigma)\n"
                     "                _mechf = _mech_fnp(layer_index, pre_mean, pre_variance, final_k3, final_k4, None, None, _fk3_cp, _fk3_star, _fk4_cp4, None, None, None, None, None, weight32, width, final=True)\n"
                     "                _rowin = fnp.concatenate((fnp.asarray(native_raw_rows[-1], dtype=fnp.float32), fnp.asarray(_mechf, dtype=fnp.float32)), axis=1)\n"
                     "                _dl = fnp.asarray(_gnn_step(layer_index, _rowin, sigma, weight32), dtype=fnp.float64)\n"
                     "                rows[-1] = fnp.asarray(fnp.asarray(final_mean, dtype=fnp.float64) + _dl, dtype=fnp.float64)\n"
                     "            break\n")
    # output = the carried corrected rows; no open-loop patch; reset the state per prediction
    s = replace_once(s, """    host, native_raw, native_sigma = _predict(mlp, return_native_tape=True)
    correction = transport_correction(
        native_raw, native_sigma, mlp.weights, _TRANSPORT_PAYLOAD
    )
    output = fnp.asarray(
        fnp.maximum(host + correction, 0.0), dtype=fnp.float32
    )
""", """    _MECH.clear(); _GNN.pop("e", None)
    host, native_raw, native_sigma = _predict(mlp, return_native_tape=True)
    output = fnp.asarray(fnp.maximum(host, 0.0), dtype=fnp.float32)   # T-114j: closed-loop rows already corrected
""")
    s = s.replace('"""FB-EC4-Q24 with a target-free native raw20/sigma instrumentation seam.',
                  '"""T-114j candidate: FB-EC4-Q24 (deployed D11 lineage) + the learned closure over the weight graph run closed-loop\n'
                  '(per-layer mean corrections from a message-passing model on the card\'s live state and the weights, carried into the\n'
                  'recursion with the consistent covariance). Numpy-free; the fnp functions are certified against the instrument.\n\n'
                  'Base docstring follows.\nFB-EC4-Q24 with a target-free native raw20/sigma instrumentation seam.', 1)
    # T-116o: the variance head's correction at the pre side (before the ReLU-map coefficients), interior (8 spaces) and final (12 spaces)
    for ind in ("        ", "            "):
        anchor = (ind + "(\n" + ind + "    coefficients, gaussian_raw, sigma, alpha, pdf,\n" + ind + "    sigma2, sigma3,\n" + ind + ") = _q38_coefficients(\n" + ind + "    pre_mean,\n" + ind + "    pre_variance,\n")
        assert s.count(anchor) == 1, (len(ind), s.count(anchor))
        vblock = (ind + "if _GNN.get(\"dv_next_layer\") == layer_index:   # T-116o variance head (+ T-116q bilinear channel)\n"
                  + ind + "    _v0 = fnp.asarray(pre_variance, dtype=fnp.float64); _dvn = fnp.asarray(_GNN[\"dv_next\"], dtype=fnp.float64)\n"
                  + ind + "    if \"dv_bil_q\" in _GNN: _dvn = _dvn + fnp.asarray(_GNN[\"dv_bil_q\"], dtype=fnp.float64) / (_v0 + 1e-12) + float(_GNN[\"dv_bil_b\"])\n"
                  + ind + "    _dvp = _VHEAD_MIX * _dvn\n"
                  + ind + "    pre_variance = _v0 * (1.0 + _dvp); pre_covariance = fnp.asarray(pre_covariance, dtype=fnp.float64) + fnp.diag(_v0 * _dvp)\n")
        s = s.replace(anchor, vblock + anchor, 1)
    s = replace_once(s, "    _MECH.clear(); _GNN.pop(\"e\", None)\n", "    _MECH.clear(); _GNN.pop(\"e\", None); _GNN.pop(\"dv_next\", None); _GNN.pop(\"dv_next_layer\", None); _GNN.pop(\"W_next\", None)\n")
    compile(s, str(OUT), "exec")
    OUT.write_text(s, encoding="utf-8")
    print("candidate", OUT, "size", OUT.stat().st_size, "sha256", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16], "| base", hashlib.sha256(BASE.read_bytes()).hexdigest()[:16], "| payload", hashlib.sha256(PAYLOAD.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
