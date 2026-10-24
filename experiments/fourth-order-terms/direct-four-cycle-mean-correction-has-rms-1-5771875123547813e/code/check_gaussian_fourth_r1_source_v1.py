"""INERT fixed rank35 Gaussian-fourth source-information test; explicit --run.

Unchanged official-bank P+Q capture is byte-qualified and sealed BEFORE a
small deterministic algebra certificate and separately metered FP64 proxy.
No truth/error labels, fitting, other rank, candidate intervention or retry.
Only copies, IO and assertion diagnostics use NumPy on wide arrays.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import marshal
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
OUT = HERE / "gaussian_fourth_r1_source_test_v1"
GATE_ENV = HERE.parent / "data_gate/.venv"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ROW = STORE / "rows/observer_fit/0035/row_manifest.json"
WEIGHTS = ROW.parent / "weights.npy"
REFERENCE = HERE / "response_rows_panel_v1/bank/call_00_rank35_output.npy"
BANK = "candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py"
HELPER = "check_response_rows_adjusted_v1.py"
PROTOCOL = "GAUSSIAN_FOURTH_R1_SOURCE_TEST_PROTOCOL_V1.md"
NOTE = "GAUSSIAN_FOURTH_R1_SAME_SOURCE_PROXY_V1.md"
PINS = {
    BANK: "3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757",
    HELPER: "8e99890bdc7742164a53303a23691e9f1cc54909c9cccc9cf64fc362eb459304",
    PROTOCOL: "65549c82badee889cd62b029ea103b01e281bdb8e8e8b1247016be4fb3c7e41c",
    NOTE: "871eac23613fb22e50c53ba71115a7fd31bb418b8e39e4811d25cccaa0c5169e",
}
WEIGHTS_SHA = "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2"
ROW_SHA = "5067b577620cb6d5dc97c61b62f10e8fb3f98880b99fe9320f1ef8a8ca567809"
MANIFEST_SHA = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
REFERENCE_SHA = "43f66a6fbb1cd0a362a8d2b1251235322a53994650be2da3c0262c53ad26a5b5"
MLP_SEED = 2758000291903877946
SCHEMA = "gaussian_fourth_r1_source_information_v1"
BLAS_ENV = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
BUDGET, PROXY_BUDGET, RSS_LIMIT = 2**41, 20_000_000_000, 8 * 1024**3
KEYS = ("C0", "gate12", "pdf12", "sigma12", "M", "alive12", "W13",
        "alpha13", "pdf13", "sigma13", "pre_v13")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def exclusive_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_helper():
    path = HERE / HELPER
    require(hashlib.sha256(path.read_bytes()).hexdigest() == PINS[HELPER], "Helper pin changed")
    spec = importlib.util.spec_from_file_location("gaussian_r1_inert_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.digest(path) == PINS[HELPER], "Helper changed during import")
    return module


def provenance(h):
    sources = {name: h.binding(HERE / name) for name in (*PINS, Path(__file__).name)}
    for name, sha in PINS.items():
        require(sources[name]["sha256"] == sha, f"Frozen source changed: {name}")
    files = {"manifest": h.binding(STORE / "manifest.json"), "row": h.binding(ROW),
             "weights": h.binding(WEIGHTS), "bank_parity_output": h.binding(REFERENCE)}
    for key, sha in (("manifest", MANIFEST_SHA), ("row", ROW_SHA),
                     ("weights", WEIGHTS_SHA), ("bank_parity_output", REFERENCE_SHA)):
        require(files[key]["sha256"] == sha, f"Fixture/reference changed: {key}")
    manifest = h.load_json(STORE / "manifest.json")["roles"]["observer_fit"]["row_manifests"]
    entry = manifest[35] if isinstance(manifest, list) else manifest["35"]
    require((STORE / entry["path"]).resolve() == ROW.resolve() and entry["sha256"] == ROW_SHA,
            "Rank35 manifest ownership mismatch")
    row = h.load_json(ROW)
    require(row["role"] == "observer_fit" and row["role_rank"] == 35
            and int(row["mlp_seed"]) == MLP_SEED
            and row["files"]["weights.npy"]["sha256"] == WEIGHTS_SHA,
            "Wrong rank35 seed or authenticated weight declaration")
    return sources, files


def save_npz(np, h, name, values):
    with (OUT / name).open("xb") as stream:
        np.savez(stream, **values)
    return h.binding(OUT / name)


def function_hash(function):
    return hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()


def capture_bank(np, flops, h, mlp, inputs, artifacts):
    spec = importlib.util.spec_from_file_location("gaussian_r1_fixed_bank", HERE / BANK)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    import_start = time.perf_counter()
    spec.loader.exec_module(module)
    import_wall = time.perf_counter() - import_start
    require(h.digest(HERE / BANK) == PINS[BANK], "Bank changed during import")
    names = ("nonlinear", "_cov_compact", "_nonlinear_first_gaussian", "nonlinear_scalar",
             "gnn_step", "final_delta")
    original = {name: getattr(module, name) for name in names}
    code_hashes = {name: function_hash(fn) for name, fn in original.items()}
    literal_hashes = {"P": h.canonical_digest(module._GNN_LITERAL),
                      "Q": h.canonical_digest(module._FINAL_LITERAL)}
    events, pair_layers, cov_layers, head_layers = [], [], [], []
    counts = {"first": 0, "scalar": 0, "final": 0}

    def copied(key, value):
        require(key not in inputs, f"Duplicate capture field {key}")
        inputs[key] = np.asarray(value).copy()

    def cov(w, C, indices, alive):
        layer = len(cov_layers) + 1
        require(layer <= 14 and head_layers == list(range(layer)), "Covariance call order changed")
        cov_layers.append(layer)
        events.append(["covariance", layer])
        if layer == 13:
            copied("M", C)
            copied("alive12", alive)
            copied("W13", w)
        return original["_cov_compact"](w, C, indices, alive)

    def nonlinear(mu, C, K21, k4, k22_override=None, metric="identity"):
        layer = len(pair_layers) + 1
        require(cov_layers[-1:] == [layer] and layer <= 14, "Pair-map call order changed")
        pair_layers.append(layer)
        events.append(["nonlinear", layer])
        if layer == 12:
            copied("C0", C)  # BEFORE the original map or the current P hook.
        value = original["nonlinear"](mu, C, K21, k4, k22_override, metric)
        aux = value[4]
        if layer == 12:
            for saved, key in (("gate12", "gate"), ("pdf12", "pdf"), ("sigma12", "sigma")):
                copied(saved, aux[key])
        if layer == 13:
            for saved, key in (("alpha13", "alpha"), ("pdf13", "pdf"),
                               ("sigma13", "sigma"), ("pre_v13", "pre_v")):
                copied(saved, aux[key])
        return value  # The original tuple and every array are returned unchanged.

    def first(*args, **kwargs):
        counts["first"] += 1
        require(counts["first"] == 1 and not events, "First-Gaussian route changed")
        events.append(["first_gaussian", 0])
        return original["_nonlinear_first_gaussian"](*args, **kwargs)

    def scalar(*args, **kwargs):
        counts["scalar"] += 1
        require(counts["scalar"] == 1 and head_layers == list(range(15)), "Final scalar route changed")
        events.append(["scalar_nonlinear", 15])
        return original["nonlinear_scalar"](*args, **kwargs)

    def step(layer, x, sigma, W, state, payload):
        require(int(layer) == len(head_layers) and payload is module._GNN_PARAMS,
                "Prefix P layer/payload changed")
        head_layers.append(int(layer))
        events.append(["P", int(layer)])
        return original["gnn_step"](layer, x, sigma, W, state, payload)

    def final(x, sigma, payload):
        counts["final"] += 1
        require(counts["scalar"] == counts["final"] == 1 and payload is module._FINAL_PARAMS,
                "Final Q route/payload changed")
        events.append(["Q", 15])
        return original["final_delta"](x, sigma, payload)

    wrappers = dict(nonlinear=nonlinear, _cov_compact=cov, _nonlinear_first_gaussian=first,
                    nonlinear_scalar=scalar, gnn_step=step, final_delta=final)
    for name, fn in wrappers.items():
        setattr(module, name, fn)
    began_utc, t0 = h.utc(), time.perf_counter()
    context = None
    try:
        with flops.BudgetContext(flop_budget=BUDGET, wall_time_limit_s=120, quiet=True) as context:
            served = module.Estimator().predict(mlp, BUDGET)
        wall = time.perf_counter() - t0
    finally:
        for name, fn in original.items():
            setattr(module, name, fn)
        require(all(getattr(module, name) is fn and function_hash(fn) == code_hashes[name]
                    for name, fn in original.items()), "Captured functions were not restored unchanged")
        sys.modules.pop(spec.name, None)
    sealing_start = time.perf_counter()
    output = np.asarray(served).copy()
    h.assert_output(np, output)
    artifacts["bank_output.npy"] = h.save_npy(np, OUT / "bank_output.npy", output)
    bill = dict(began_utc=began_utc, ended_utc=h.utc(), context_wall_s=wall,
                flops=int(context.flops_used), native_residual_s=float(context.residual_wall_time_s),
                peak_rss_bytes=h.peak_rss_bytes(), import_wall_s=import_wall,
                first_call_initialization_included=True, production_runtime_qualification=False)
    h.write_json(OUT / "bank_bill.json", bill)
    artifacts["bank_bill.json"] = h.binding(OUT / "bank_bill.json")
    require(0 < bill["flops"] <= BUDGET and wall < 120 and bill["peak_rss_bytes"] < RSS_LIMIT
            and math.isfinite(bill["native_residual_s"]), "Native acquisition resource gate failed")
    require(pair_layers == cov_layers == list(range(1, 15)) and head_layers == list(range(15))
            and counts == dict(first=1, scalar=1, final=1), "Full bank routing did not qualify")
    expected_events = [["first_gaussian", 0], ["P", 0]]
    for layer in range(1, 15):
        expected_events += [["covariance", layer], ["nonlinear", layer], ["P", layer]]
    expected_events += [["scalar_nonlinear", 15], ["Q", 15]]
    require(events == expected_events, "Full causal event ordering differs")
    require(literal_hashes == {"P": h.canonical_digest(module._GNN_LITERAL),
                               "Q": h.canonical_digest(module._FINAL_LITERAL)}, "Bank P/Q literals changed")
    require(set(inputs) == set(KEYS), "Unexpected/missing causal field; no K21/error targets are allowed")
    for key, value in inputs.items():
        shape = (1024, 1024) if key in ("C0", "M", "W13") else (1024,)
        require(value.shape == shape and bool(np.isfinite(value).all()), f"Invalid input {key}")
        if key == "alive12":
            require(value.dtype == np.dtype("bool"), "Alive mask changed dtype")
        else:
            require(value.dtype == np.dtype("float32" if key == "W13" else "float64"),
                    f"Unexpected captured dtype: {key}")
    require(bool((inputs["sigma12"] > 0).all()) and bool((inputs["sigma13"] > 0).all())
            and bool((inputs["pre_v13"] != 0).all()), "Degenerate Gaussian response units")
    require(inputs["W13"].tobytes() == np.asarray(mlp.weights[13]).tobytes(), "W13 ownership changed")
    reference = np.load(REFERENCE, allow_pickle=False)
    h.assert_output(np, reference)
    require(output.tobytes(order="C") == reference.tobytes(order="C"), "All16 served bank bytes differ")
    artifacts["causal_inputs.npz"] = save_npz(np, h, "causal_inputs.npz", inputs)
    for value in inputs.values():
        value.setflags(write=False)
    return dict(bank_bill=bill, head_events=events, original_function_code_sha256=code_hashes,
                functions_restored=True, payload_literal_sha256=literal_hashes,
                all16_served_byte_parity=True, reference_sha256=REFERENCE_SHA,
                post_capture_io_and_parity_wall_s=time.perf_counter() - sealing_start,
                post_capture_io_peak_rss_bytes=h.peak_rss_bytes(),
                captured_fields={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k, v in inputs.items()})


def qtrace(xp, matrix):
    return xp.add(xp.square(xp.trace(matrix)), xp.multiply(2.0, xp.sum(xp.square(matrix))))


def norm(xp, array):
    return xp.sqrt(xp.sum(xp.square(array)))


def proxy_arithmetic(xp, inputs):
    """All proxy arithmetic, including scalar bookkeeping; xp=FNP for width1024.

    The only other caller uses NumPy for the deterministic width4 certificate.
    Scalar conversion below is solely a failure guard or receipt serialization.
    """
    x = {key: xp.asarray(inputs[key], dtype=xp.float64) for key in KEYS}
    C = xp.multiply(xp.add(x["C0"], x["C0"].T), 0.5)
    M, W = x["M"], x["W13"]
    dA = xp.multiply(xp.divide(x["pdf12"], x["sigma12"]), x["alive12"])
    gA = xp.multiply(x["gate12"], x["alive12"])
    n = int(C.shape[0])
    u = xp.divide(xp.ones(n, dtype=xp.float64), xp.sqrt(xp.asarray(n, dtype=xp.float64)))
    norms = []
    for _ in range(16):
        z = C @ u
        denominator = norm(xp, z)
        require(math.isfinite(float(denominator)) and float(denominator) > 0,
                "Degenerate/nonfinite fixed power norm; no fallback")
        norms.append(float(denominator))
        u = xp.divide(z, denominator)
    Cu = C @ u
    theta = xp.sum(xp.multiply(u, Cu))
    lam = xp.multiply(12.0, theta)
    rawV = xp.multiply(xp.multiply(dA, u)[:, None], xp.multiply(C, gA[None, :]))
    V = xp.multiply(xp.add(rawV, rawV.T), 0.5)
    v, h = xp.diag(C), xp.multiply(dA, gA)
    vh = xp.multiply(v, h)
    term1 = xp.sum(xp.multiply(vh, C @ vh))
    term2 = xp.sum(xp.multiply(xp.multiply(v, xp.square(dA)), xp.square(C) @ xp.square(gA)))
    term3 = xp.sum(xp.multiply(xp.multiply(xp.multiply(xp.square(C), C), h[:, None]), h[None, :]))
    TF = xp.multiply(4.0, xp.add(xp.add(term1, term2), term3))
    DM, QV = qtrace(xp, M), qtrace(xp, V)
    require(math.isfinite(float(DM)) and float(DM) != 0, "Degenerate bank-Q denominator")
    alphaF = xp.divide(xp.multiply(3.0, TF), DM)
    alpha1 = xp.divide(xp.multiply(lam, QV), DM)
    # Exactly THREE dense width1024 products. All other @ are matrix-vector.
    B = C @ xp.multiply(gA[:, None], W)
    T = xp.multiply(xp.multiply(dA[:, None], W), B)
    Y = C @ T
    F = xp.multiply(12.0, xp.sum(xp.multiply(T, Y), axis=0))
    F1 = xp.multiply(lam, xp.square(u @ T))
    q = xp.sum(xp.multiply(W, M @ W), axis=0)
    R0 = xp.multiply(alphaF, xp.square(q))
    R1 = xp.add(xp.multiply(xp.subtract(alphaF, alpha1), xp.square(q)), F1)
    omega = xp.divide(xp.multiply(xp.subtract(xp.square(x["alpha13"]), 1.0), x["pdf13"]),
                      xp.multiply(24.0, xp.multiply(x["pre_v13"], x["sigma13"])))
    error0 = xp.multiply(omega, xp.subtract(F, R0))
    error1 = xp.multiply(omega, xp.subtract(F, R1))
    correction = xp.multiply(omega, xp.subtract(R1, R0))
    E0, E1 = xp.mean(xp.square(error0)), xp.mean(xp.square(error1))
    require(math.isfinite(float(E0)) and float(E0) > 0, "Degenerate/nonfinite E0; no fallback")
    error_norm, correction_norm = norm(xp, error0), norm(xp, correction)
    cosine_den = xp.multiply(error_norm, correction_norm)
    cosine = None if float(cosine_den) == 0 else xp.divide(xp.sum(xp.multiply(error0, correction)), cosine_den)
    Msym = xp.multiply(xp.add(M, M.T), 0.5)
    mode_trace = xp.divide(xp.multiply(lam, QV), 3.0)
    mode_plane_trace = xp.sum(xp.multiply(xp.divide(lam, 3.0),
        xp.add(xp.multiply(xp.diag(V)[:, None], xp.diag(V)[None, :]), xp.multiply(2.0, xp.square(V)))))
    scalars = dict(E0=E0, E1=E1, ratio=xp.divide(E1, E0), theta=theta, lambda_mode=lam,
        source_partial_trace=TF, mode_partial_trace=mode_trace, mode_plane_trace=mode_plane_trace,
        alpha_same_source=alphaF, alpha_mode_overlap=alpha1, bank_Q=DM,
        bank_Q_minus_symmetric_Q=xp.subtract(DM, qtrace(xp, Msym)),
        explicit_asymmetry_defect=xp.multiply(0.5, xp.sum(xp.square(xp.subtract(M, M.T)))),
        power_residual=norm(xp, xp.subtract(Cu, xp.multiply(theta, u))),
        unit_norm=xp.sum(xp.square(u)), error_norm=error_norm, correction_norm=correction_norm,
        q_minus_original_pre_v_rms=xp.sqrt(xp.mean(xp.square(xp.subtract(q, x["pre_v13"])))),
        q_minus_original_pre_v_max_abs=xp.max(xp.abs(xp.subtract(q, x["pre_v13"]))))
    arrays = dict(F=F, mode_F=F1, radial_F=R0, radial_plus_mode_F=R1, omega=omega, q=q,
                  original_pre_v=x["pre_v13"], u=u, V=V)
    return scalars, arrays, norms, cosine


def small_certificate(np):
    """Deterministic width4 algebra only; independent coordinate polarization."""
    positive = np.array([[1.4, .2, -.1, .05], [.2, 1.1, .12, -.08],
                         [-.1, .12, .9, .06], [.05, -.08, .06, 1.3]], dtype=np.float64)
    signed = positive.copy()
    signed[0, 0] = -2.0
    carried = np.array([[1.2, .12, .04, -.06], [.12, .8, -.03, .02],
                        [.04, -.03, 1.1, .05], [-.06, .02, .05, .7]], dtype=np.float64)
    asym = carried.copy()
    asym[0, 1] += .07
    W = np.array([[(((i + 1) * (j + 2)) % 7 - 3) / 5 for j in range(6)] for i in range(4)], dtype=np.float64)
    cases = (("positive_full", positive, carried, [1, 1, 1, 1]),
             ("positive_partial_asymmetric", positive, asym, [1, 0, 1, 1]),
             ("signed_partial_asymmetric", signed, asym, [1, 0, 1, 1]))
    receipts = []
    for name, C, M, mask in cases:
        inputs = dict(C0=C, M=M, W13=W, alive12=np.array(mask, dtype=bool),
            gate12=np.array([.25, .65, .4, .9]), pdf12=np.array([.17, .31, .23, .11]),
            sigma12=np.array([1.1, .8, 1.3, .9]), alpha13=np.array([-1.2, .3, 1.5, .2, -.4, .7]),
            pdf13=np.array([.19, .38, .13, .39, .37, .31]), sigma13=np.ones(6),
            pre_v13=np.array([.7, 1.2, .8, 1.1, .9, 1.3]))
        s, a, _, _ = proxy_arithmetic(np, inputs)
        dA = inputs["pdf12"] / inputs["sigma12"] * mask
        gA = inputs["gate12"] * mask
        def polynomial(w):
            t = dA * w * (C @ (gA * w))
            return 12 * float(t @ C @ t)
        eye = np.eye(4)
        diagonals = [polynomial(e) for e in eye]
        pair_values = list(diagonals)
        trace = sum(diagonals)
        for i in range(4):
            for j in range(i + 1, 4):
                kij = (polynomial(eye[i] + eye[j]) + polynomial(eye[i] - eye[j])
                        - 2 * diagonals[i] - 2 * diagonals[j]) / 12
                trace += 2 * kij
                pair_values.extend([kij, kij])
        trace_error = abs(float(s["source_partial_trace"]) - trace) / max(sum(map(abs, pair_values)), 1e-300)
        Vquery = np.sum(W * (a["V"] @ W), axis=0)
        T = dA[:, None] * W * (C @ (gA[:, None] * W))
        ut = a["u"] @ T
        query_error = float(np.linalg.norm(Vquery - ut) / max(np.linalg.norm(Vquery), np.linalg.norm(ut), 1e-300))
        Fdirect = np.array([polynomial(W[:, j]) for j in range(W.shape[1])])
        ray_error = float(np.linalg.norm(a["F"] - Fdirect) / max(np.linalg.norm(Fdirect), 1e-300))
        trace_mode_error = abs(float(s["mode_partial_trace"] - s["mode_plane_trace"])) / max(abs(float(s["mode_partial_trace"])), 1e-300)
        defect_error = abs(float(s["bank_Q_minus_symmetric_Q"] - s["explicit_asymmetry_defect"])) / max(abs(float(s["bank_Q"])), 1e-300)
        errors = dict(trace=trace_error, V_query=query_error, exact_F_ray=ray_error,
                      mode_trace=trace_mode_error, asymmetry_defect=defect_error)
        passed = all(math.isfinite(v) and v <= 1e-10 for v in errors.values())
        receipts.append(dict(case=name, relative_errors=errors, passed=passed))
    return dict(width=4, deterministic=True, random_draws=0, target_access=False,
                unmetered_small_numpy_only=True, cases=receipts, passed=all(r["passed"] for r in receipts))


def run(h, state, inputs, artifacts):
    require(Path(sys.prefix).resolve() == GATE_ENV.resolve() and "anaconda3" not in sys.executable,
            "Use designated data_gate/.venv; never AP/Torch")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch this sole worker at nice>=10")
    env = {key: os.environ.get(key) for key in BLAS_ENV}
    require(env == dict.fromkeys(BLAS_ENV, "1"), "Set all three BLAS variables to1 BEFORE imports")
    require("numpy" not in sys.modules and "torch" not in sys.modules,
            "This must be a fresh numerical process")
    swap = subprocess.run(["/usr/sbin/sysctl", "vm.swapusage"], text=True, capture_output=True, check=True)
    sources, fixtures = provenance(h)
    state.update(source_bindings=sources, fixture_bindings=fixtures, blas_env=env,
                 swap_before=swap.stdout.strip(), pid=os.getpid(), executable=sys.executable,
                 nice=os.getpriority(os.PRIO_PROCESS, 0), begun_utc=h.utc())
    versions = {k: importlib.metadata.version(k) for k in ("flopscope", "whestbench", "numpy", "threadpoolctl")}
    require(versions["flopscope"] == "0.12.0" and versions["whestbench"] == "0.16.0", "Wrong native versions")
    state["versions"] = versions
    h.write_json(OUT / "started.json", state)
    artifacts["started.json"] = h.binding(OUT / "started.json")
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info, threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    flops.configure(symmetry_warnings=False)
    with threadpool_limits(limits=1):
        pools = threadpool_info()
        require(all(p.get("num_threads") == 1 for p in pools if p.get("user_api") == "blas"), "BLAS thread gate failed")
        state["threadpools"] = pools
        state["thread_attestation"] = "Launch variables plus available threadpoolctl inventory; empty inventory cannot attest Accelerate threads."
        t0 = time.perf_counter()
        weights = np.load(WEIGHTS, mmap_mode="r", allow_pickle=False)
        require(weights.shape == (16, 1024, 1024) and weights.dtype == np.dtype("float32")
                and bool(np.isfinite(weights).all()), "Invalid Phase2 weight fixture")
        mlp = MLP(width=1024, depth=16, weights=list(weights), seed=int(derive_estimator_seed(MLP_SEED)),
                  name="fixed-gaussian-fourth-source-rank35")
        state["loading_wall_s"] = time.perf_counter() - t0
        state["loading_peak_rss_bytes"] = h.peak_rss_bytes()
        h.check_bindings(sources)
        h.check_bindings(fixtures)
        state["stage"] = "unchanged_bank_capture"
        capture = capture_bank(np, flops, h, mlp, inputs, artifacts)
        t0 = time.perf_counter()
        h.check_bindings(sources)
        h.check_bindings(fixtures)
        seal = dict(schema=SCHEMA, sealed_utc=h.utc(), source_bindings=sources,
                    fixture_bindings=fixtures, artifacts=dict(artifacts), capture=capture,
                    truth_files_accessed=False, current_error_targets_accessed=False)
        h.write_json(OUT / "capture_sealed.json", seal)
        artifacts["capture_sealed.json"] = h.binding(OUT / "capture_sealed.json")
        state.update(capture=capture, capture_seal_sha256=artifacts["capture_sealed.json"]["sha256"],
                     sealing_wall_s=time.perf_counter() - t0,
                     sealing_peak_rss_bytes=h.peak_rss_bytes())
        del mlp, weights
        gc.collect()
        state["stage"] = "deterministic_small_certificate"
        t0 = time.perf_counter()
        certificate = small_certificate(np)
        certificate.update(wall_s=time.perf_counter() - t0, peak_rss_bytes=h.peak_rss_bytes(),
                           utc=h.utc(), source_bindings=sources, capture_seal_sha256=state["capture_seal_sha256"])
        h.write_json(OUT / "small_certificate.json", certificate)
        artifacts["small_certificate.json"] = h.binding(OUT / "small_certificate.json")
        require(certificate["passed"], "Deterministic algebra gate failed; wide proxy prohibited")
        h.check_bindings(sources)
        h.check_bindings(fixtures)
        h.check_bindings(artifacts)
        state["stage"] = "separately_metered_float64_proxy"
        began, t0 = h.utc(), time.perf_counter()
        with flops.BudgetContext(flop_budget=PROXY_BUDGET, wall_time_limit_s=60, quiet=True) as context:
            scalar_values, proxy_arrays, power_norms, cosine = proxy_arithmetic(fnp, inputs)
        wall = time.perf_counter() - t0
        scalars = {k: float(v) for k, v in scalar_values.items()}
        scalars["correction_residual_cosine"] = None if cosine is None else float(cosine)
        require(all(math.isfinite(v) for v in scalars.values() if v is not None)
                and len(power_norms) == 16 and all(math.isfinite(v) and v > 0 for v in power_norms),
                "Nonfinite proxy result")
        arrays = {k: np.asarray(v).copy() for k, v in proxy_arrays.items()}
        for key, value in arrays.items():
            expected = (1024, 1024) if key == "V" else (1024,)
            require(value.shape == expected and value.dtype == np.dtype("float64")
                    and bool(np.isfinite(value).all()), f"Invalid proxy output {key}")
        identity = dict(unit_norm_abs=abs(scalars["unit_norm"] - 1),
            mode_trace_relative=abs(scalars["mode_partial_trace"] - scalars["mode_plane_trace"])
                / max(abs(scalars["mode_partial_trace"]), 1e-300),
            asymmetry_relative=abs(scalars["bank_Q_minus_symmetric_Q"] - scalars["explicit_asymmetry_defect"])
                / max(abs(scalars["bank_Q"]), 1e-300))
        require(all(v <= 1e-10 for v in identity.values()), "Wide numerical identity gate failed")
        artifacts["proxy_values.npz"] = save_npz(np, h, "proxy_values.npz", arrays)
        bill = dict(began_utc=began, ended_utc=h.utc(), flops=int(context.flops_used),
                    context_wall_s=wall, residual_s=float(context.residual_wall_time_s),
                    peak_rss_bytes=h.peak_rss_bytes(), flop_cap=PROXY_BUDGET,
                    acquisition_flops=capture["bank_bill"]["flops"])
        h.write_json(OUT / "proxy_bill.json", bill)
        artifacts["proxy_bill.json"] = h.binding(OUT / "proxy_bill.json")
        require(0 < bill["flops"] <= PROXY_BUDGET and wall < 60 and bill["peak_rss_bytes"] < RSS_LIMIT
                and math.isfinite(bill["residual_s"]), "Proxy resource gate failed")
        ratio = scalars["ratio"]
        decision = ("CLOSE_FIXED_SUPPLIER" if scalars["E1"] >= scalars["E0"] else
                    "SOURCE_VALUE_AND_COST_REVIEW_ONLY" if ratio <= .5 else
                    "SMALL_SOURCE_EFFECT_NO_CARRIER_OR_FIT")
        state.update(proxy_scalars=scalars, power_normalization_norms=power_norms, identity_checks=identity,
                     proxy_bill=bill, scientific_decision=decision, information_gate_passed=ratio <= .5,
                     source_proxy_not_raw_score=True, no_candidate_or_fit_authorized=True)
    h.check_bindings(sources)
    h.check_bindings(fixtures)
    h.check_bindings(artifacts)
    state.update(completed=True, qualification_passed=True, hashes_unchanged=True,
                 completed_utc=h.utc(), stage="completed", peak_rss_bytes=h.peak_rss_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Execute only after full root and independent source review")
    args = parser.parse_args()
    if not args.run:
        print("INERT: no data opened or numeric modules imported. Explicit reviewed --run required.")
        return 0
    OUT.mkdir(exist_ok=False)  # Never overwrite any previous attempt or partial failure.
    state = dict(schema=SCHEMA, rank=35, completed=False, qualification_passed=False,
                 truth_files_accessed=False, current_error_targets_accessed=False, fits=0,
                 protocol_sha256=PINS[PROTOCOL], stage="preflight")
    inputs, artifacts, h = {}, {}, None
    t0 = time.perf_counter()
    try:
        h = load_helper()
        run(h, state, inputs, artifacts)
    except Exception as error:
        state.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        # Preserve whatever input-only observation was obtained on a failed call.
        if inputs and not (OUT / "causal_inputs.npz").exists():
            try:
                np = sys.modules["numpy"]
                artifacts["partial_causal_inputs.npz"] = save_npz(np, h, "partial_causal_inputs.npz", inputs)
            except Exception as preservation_error:
                state["partial_preservation_error"] = repr(preservation_error)
        exclusive_json(OUT / "failure.json", dict(state))
    state.update(artifacts=artifacts, whole_run_wall_s=time.perf_counter() - t0)
    if h is not None:
        state["terminal_utc"] = h.utc()
        state["peak_rss_bytes"] = h.peak_rss_bytes()
    result_sha = exclusive_json(OUT / "result.json", state)
    exclusive_json(OUT / "completion.json", dict(schema=SCHEMA, result_sha256=result_sha,
        qualification_passed=state["qualification_passed"], completed=state["completed"],
        protocol_sha256=PINS[PROTOCOL], truth_files_accessed=False))
    print(json.dumps(dict(qualification_passed=state["qualification_passed"],
                         decision=state.get("scientific_decision"), result_sha256=result_sha)), flush=True)
    return 0 if state["qualification_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
