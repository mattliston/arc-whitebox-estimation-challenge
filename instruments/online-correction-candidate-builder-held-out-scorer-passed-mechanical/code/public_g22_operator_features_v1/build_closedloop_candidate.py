#!/usr/bin/env python3
"""TAG g22-build-closedloop-candidate (T-107): the deployed D11 card (estimator_cap.py) + the CLOSED-LOOP operator:
the P2R0F2J recurrence is stepped online inside the layer loop and its per-layer mean correction is carried INTO the
recursion (mean + the consistent covariance C <- C - (m d^T + d m^T) - d d^T); the output rows are the carried
corrected means; the open-loop output patch is removed. The refit payload replaces the deployed literal via the
sha-pinned splicer (public_deploy_composition_prep_v1/splice_payload.py). Numpy-free: every op is flopscope.
Usage: build_closedloop_candidate.py <REFIT_MODEL.pt> <out_candidate.py>"""
import hashlib, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_d11_util_halving_v1/estimator_cap.py"
SPLICER = P2 / "public_deploy_composition_prep_v1/splice_payload.py"
MODEL = Path(sys.argv[1]); OUT = Path(sys.argv[2])
PREPAY = OUT.with_name(OUT.stem + "_prepay.py")


def replace_once(s, old, new):
    assert s.count(old) == 1, (s.count(old), old[:80])
    return s.replace(old, new, 1)


STEP = '''
_CL = {}


def _cl_step(layer_index, raw_row, sigma_row, layer_weights):
    """T-107 CLOSED LOOP: one layer of the P2R0F2J recurrence (transport_correction stepped online) -> this layer's
    per-unit mean correction. State persists across layers of one prediction; reset below FIRST_CORRECTED_LAYER."""
    if "params" not in _CL:
        _CL["params"] = materialize_parameters(_TRANSPORT_PAYLOAD)
    P = _CL["params"]; width = int(raw_row.shape[0]); channels = int(P["source_bias"].shape[1])
    if layer_index < FIRST_CORRECTED_LAYER:
        _CL["states"] = fnp.zeros((width, channels), dtype=fnp.float32)
        return fnp.zeros((width,), dtype=fnp.float32)
    raw = fnp.asarray(raw_row, dtype=fnp.float32); sig = fnp.asarray(sigma_row, dtype=fnp.float32)
    if layer_index == FIRST_CORRECTED_LAYER:
        messages = fnp.zeros((width, channels), dtype=fnp.float32); energies = fnp.zeros((width, channels), dtype=fnp.float32)
    else:
        states = _CL["states"]
        messages = fnp.transpose(fnp.transpose(states) @ layer_weights)
        energies = fnp.transpose(fnp.transpose(states * states) @ (layer_weights * layer_weights))
        phi = raw[:, 5, None]; messages = phi * messages; energies = (phi * phi) * energies
    tail = fnp.clip((raw[:, 1:] - P["centers"][layer_index]) / P["scales"][layer_index], -8.0, 8.0)
    rms = P["layer_rms"][layer_index]; rms2 = rms * rms
    message_rms = fnp.clip(messages / rms, -8.0, 8.0)
    message_sigma = fnp.clip(messages / fnp.maximum(sig[:, None] * rms, 2.0**-30), -8.0, 8.0)
    log_energy = fnp.clip(fnp.log1p(fnp.maximum(energies / rms2, 0.0)), 0.0, 8.0)
    coherence = fnp.clip(messages / fnp.sqrt(energies + COHERENCE_FLOOR * rms2), -8.0, 8.0)
    features = fnp.concatenate((raw[:, :1], tail, message_rms, message_sigma, log_energy, coherence), axis=1)
    hidden = _silu(features @ fnp.transpose(P["trunk0_weight"]) + P["trunk0_bias"])
    hidden = _silu(hidden @ fnp.transpose(P["trunk2_weight"]) + P["trunk2_bias"])
    source = (sig * rms)[:, None] * (hidden @ P["source_weight"][layer_index] + P["source_bias"][layer_index])
    dynamic_gain = 1.0 + 0.25 * fnp.tanh(hidden @ P["gate_weight"][layer_index] + P["gate_bias"][layer_index] + P["transport_delta"][layer_index])
    mixing = fnp.eye(channels, dtype=fnp.float32) + 0.12 * fnp.tanh(P["mix_delta"][layer_index])
    _CL["states"] = (dynamic_gain * messages) @ mixing + source
    return _CL["states"][:, 0]


def _predict_corrected(mlp: MLP):
'''


def main():
    s = BASE.read_text(encoding="utf-8")
    # (1) the online operator step
    s = replace_once(s, "\n\ndef _predict_corrected(mlp: MLP):\n", STEP)
    # (2) interior layers: after the tape row, step the operator and carry its correction into mean + covariance
    s = replace_once(s, "            native_sigma_rows.append(sigma)\n\n        gate = gaussian_raw[0]\n",
                     "            native_sigma_rows.append(sigma)\n"
                     "            # T-107 closed loop: the operator's mean correction carried into the recursion\n"
                     "            _dl = fnp.asarray(_cl_step(layer_index, native_raw_rows[-1], sigma, weight32), dtype=fnp.float64)\n"
                     "            _mc = fnp.asarray(mean, dtype=fnp.float64)\n"
                     "            covariance = fnp.asarray(covariance, dtype=fnp.float64) - fnp.outer(_mc, _dl) - fnp.outer(_dl, _mc) - fnp.outer(_dl, _dl)\n"
                     "            mean = fnp.asarray(_mc + _dl, dtype=fnp.float64)\n"
                     "            rows[-1] = mean\n\n"
                     "        gate = gaussian_raw[0]\n")
    # (3) the final layer: its correction goes onto the final row
    s = replace_once(s, "                native_sigma_rows.append(sigma)\n            break\n",
                     "                native_sigma_rows.append(sigma)\n"
                     "                _dl = fnp.asarray(_cl_step(layer_index, native_raw_rows[-1], sigma, weight32), dtype=fnp.float64)\n"
                     "                rows[-1] = fnp.asarray(fnp.asarray(final_mean, dtype=fnp.float64) + _dl, dtype=fnp.float64)\n"
                     "            break\n")
    # (4) the output is the carried corrected rows; no open-loop patch
    s = replace_once(s, """    host, native_raw, native_sigma = _predict(mlp, return_native_tape=True)
    correction = transport_correction(
        native_raw, native_sigma, mlp.weights, _TRANSPORT_PAYLOAD
    )
    output = fnp.asarray(
        fnp.maximum(host + correction, 0.0), dtype=fnp.float32
    )
""", """    _CL.clear()
    host, native_raw, native_sigma = _predict(mlp, return_native_tape=True)
    output = fnp.asarray(fnp.maximum(host, 0.0), dtype=fnp.float32)   # T-107: closed-loop rows already corrected
""")
    s = s.replace('"""FB-EC4-Q24 with a target-free native raw20/sigma instrumentation seam.',
                  '"""T-107 candidate: FB-EC4-Q24 (deployed D11 lineage) + the CLOSED-LOOP P2R0F2J operator (per-layer mean\n'
                  'corrections carried into the recursion with the consistent covariance; refit payload spliced).\n\n'
                  'Base docstring follows.\nFB-EC4-Q24 with a target-free native raw20/sigma instrumentation seam.', 1)
    compile(s, str(PREPAY), "exec")
    PREPAY.write_text(s, encoding="utf-8")
    r = subprocess.run([sys.executable, str(SPLICER), "--model", str(MODEL), "--estimator", str(PREPAY), "--output", str(OUT)],
                       capture_output=True, text=True)
    print(r.stdout[-800:], r.stderr[-800:])
    assert r.returncode == 0 and OUT.exists(), "splice failed"
    print("candidate", OUT, "sha256", hashlib.sha256(OUT.read_bytes()).hexdigest()[:16], "| base", hashlib.sha256(BASE.read_bytes()).hexdigest()[:16], "| model", hashlib.sha256(MODEL.read_bytes()).hexdigest()[:16])


if __name__ == "__main__":
    main()
