"""Inert-by-default, target-free actual-bank teacher action capture.

PREPARED FOR REVIEW; execution is a separate root-authorized action.
Only ranks 35..58 are accepted. Rank 35 must qualify this exact source
before another rank can run with --qualified-panel. No Torch, MC, truth,
fit, or candidate code is imported. See the companion design document.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time


HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "rows_two_memory_teacher_action_v1"
DESIGN = "TWO_MEMORY_TEACHER_ACTION_CAPTURE_DESIGN_V1.md"
TEST_PROTOCOL = "NONLINEAR_PAIR_MEMORY_INFORMATION_PROTOCOL_V1.md"
TEST_PHYSICS = "pair_memory_action_physics_v1.py"
BANK = "candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py"
BANK_SHA = "3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757"
P_NAME = "payload_compact_feed_fullpair_v1.npz"
Q_NAME = "payload_fullpair_final_head_v1.npz"
EXPECTED = {
    BANK: BANK_SHA,
    P_NAME: "166b3c67ec4d27793d433e38e15086b95d052426fdf435a79067cc07a2a4dd7e",
    Q_NAME: "848420b459f08e3e39a0a83916db16d7a4400381fcf256a373cc0ac139c12c63",
    "compact_fullpair_final_reference_v1.py": "bf14d9508ec3d6c1bfedd313f3b357b27b046e1fccd472448a4239435cfd101c",
    "radial_compact_feed_fullpair_v1.py": "5e39ee96296de7f1b7af583ba8a78d8bc82ff1f85f5bd3b9f5439c540f88fa25",
    "radial_compact_radialfeed_v1.py": "4f1ea342dc84833c35cea41d3d5029d0beb0262484684e802f37dfedd5c3b866",
    "radial_binary_compact.py": "9759750857aab404da2774d336d809b0933a57fb4bdf1fe64c297fe7a1fce108",
    "radial_mehler.py": "828e7e7dac78437bc528ec58b1235ea96cae92b7bcb1e22dcedde977808b6495",
    "k3_radial_numpy.py": "13deca1212750672f6c9808140eb225074baea078aedc2b19a843cda733e45e7",
    "radial_cubic.py": "39ce1264317af3e88b90ba8b31de35cf75a61002eca9d40ab157697e2470ddf1",
    "radial_features.py": "6fda3eeb3dfca19fdf60bd44b4f427f8dbd016542af53ec1e721efafe43fd30e",
    "radial_gnn_numpy.py": "519abaee0601ea71366fe3a74bd07397064c31257b17ab31b26fca5a39802e29",
    "compact_scalar_final_v1.py": "e9b64b77db028fe88b9c5b1c3b5cfdebdeab7bb97780f41017349d82e48bc5f8",
    "compact_final_head_v1.py": "acaee5719a619518caa989f4122847033ace9f15526bbe283144dc1535c3702a",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execute(args):
    # Numerical imports and every array access are behind --execute.
    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules),
            "This capture must run in a fresh NumPy-only process")
    require("/anaconda3/" not in sys.executable,
            "Use the repository NumPy .venv interpreter, never the AP/Torch interpreter")
    source_names = list(EXPECTED) + [Path(__file__).name, DESIGN, TEST_PROTOCOL, TEST_PHYSICS]
    source_hashes = {name: sha(HERE / name) for name in source_names}
    for name, expected in EXPECTED.items():
        require(source_hashes[name] == expected, f"Frozen source/payload mismatch: {name}")
    if args.rank != 35:
        require(args.qualified_panel, "Only rank 35 is permitted before the explicit panel stage")
        first = json.loads((OUT_ROOT / "row_0035" / "result.json").read_text())
        require(first["passed"] and first["rank"] == 35 and not first["target_access"],
                "Rank 35 did not qualify")
        require(first["source_hashes"] == source_hashes,
                "Rank 35 qualified a different source or payload")

    tape = HERE / "rows_current_transport_shadow_v2" / f"row_{args.rank:04d}"
    old_final_dir = HERE / "rows_matrix_error_observer_v1" / f"row_{args.rank:04d}"
    input_paths = {f"prefix/{name}": tape / name for name in
                   ("weights.npy", "baseline.npy", "x.npy", "sigma.npy", "pre_cov.npy", "k21.npy")}
    input_paths.update({"final/final.npz": old_final_dir / "final.npz",
                        "final/result.json": old_final_dir / "result.json"})
    input_hashes = {name: sha(path) for name, path in input_paths.items()}
    old_receipt = json.loads(input_paths["final/result.json"].read_text())
    require(old_receipt["passed"] and old_receipt["target_access"] is False,
            "Existing final parity row was not qualified/target-free")
    for name in (P_NAME, Q_NAME):
        require(old_receipt["source_hashes"][name] == source_hashes[name],
                f"Existing parity row uses another payload: {name}")

    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info, threadpool_limits
    import compact_fullpair_final_reference_v1 as parent
    from radial_cubic import plane
    from radial_features import features
    from radial_gnn_numpy import gnn_step
    from compact_scalar_final_v1 import final_scalar_features
    from compact_final_head_v1 import final_input, final_delta

    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules),
            "Unexpected Torch import")
    W = np.load(input_paths["prefix/weights.npy"], allow_pickle=False)
    require(W.shape == (16, 1024, 1024) and W.dtype == np.float32 and np.isfinite(W).all(),
            "Not the fixed width-1024/depth-16 Phase 2 weight tensor")
    with np.load(HERE / P_NAME, allow_pickle=False) as z:
        P = {k: z[k] for k in z.files}
    with np.load(HERE / Q_NAME, allow_pickle=False) as z:
        Q = {k: z[k] for k in z.files}
    old = {name: np.load(input_paths[f"prefix/{name}.npy"], mmap_mode="r", allow_pickle=False)
           for name in ("baseline", "x", "sigma", "pre_cov", "k21")}
    with np.load(input_paths["final/final.npz"], allow_pickle=False) as z:
        old_final = {k: z[k] for k in ("prediction", "sigma", "host", "delta", "h")}

    out = OUT_ROOT / f"row_{args.rank:04d}"
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "started.json", dict(rank=args.rank, target_access=False,
               source_hashes=source_hashes, input_hashes=input_hashes,
               input_paths={k: str(p.relative_to(HERE)) for k, p in input_paths.items()},
               numpy_version=np.__version__, scipy_version=scipy.__version__,
               threads=2, fixed_unit_variance_floor=1e-10))
    started = time.perf_counter()
    state = np.zeros((1024, 64), np.float32)
    pre_vectors, post_receipts, prefix_parity = [], [], []
    final_saved = {}
    previous_short = None
    aux13 = None

    def save_npz(name, **arrays):
        for key, value in arrays.items():
            require(np.isfinite(value).all(), f"Non-finite output {name}:{key}")
        with (out / name).open("xb") as stream:
            np.savez(stream, **arrays)

    def cov_stats(C):
        d = np.diag(C)
        return dict(diagonal_min=float(d.min()), negative_diagonal_count=int((d < 0).sum()),
                    below_unit_floor_count=int((d < 1e-10).sum()))

    def capture_pre(l, previous_m, previous_C, mu, Cpre, K21, alive, radial):
        # This call is inserted before nonlinear(), not inside/after the head hook.
        nonlocal aux13
        require(l == len(pre_vectors), "Unexpected pre-state order")
        require(np.array_equal(Cpre.astype(np.float32), old["pre_cov"][l]) and
                np.array_equal(K21.astype(np.float32), old["k21"][l]),
                f"Existing prefix pre C/K21 byte parity failed at {l}")
        if 1 <= l <= 13:
            require(previous_short is not None and previous_short["layer"] == l - 1,
                    f"Missing coherent previous post state at {l}")
            require(np.array_equal(previous_m, previous_short["m"]) and
                    np.array_equal(np.diag(previous_C), previous_short["variance"]),
                    f"Previous m/C diagonal coherence failed at {l}")
        v = np.diag(Cpre).copy()
        sigma = np.sqrt(np.maximum(v, 1e-10))
        values = dict(mu=mu.copy(), variance=v, sigma=sigma, alpha=mu/sigma,
                      alive=alive.copy(), radial=np.array(radial))
        pre_vectors.append(values)
        if l == 13:
            # Keep the exact bank read. Never substitute reconstructed post D here.
            aux13 = dict(pre_C=Cpre.copy(), pre_K21_loss_only=K21.copy(),
                         pre_mu=mu.copy(), pre_alive=alive.copy(),
                         previous_m=previous_m.copy(),
                         previous_variance=np.diag(previous_C).copy())

    def capture_post(l, m, C, p, lam, star, b, alive, radial, map_target):
        nonlocal previous_short
        if l > 12:
            return
        require(l == len(post_receipts), "Unexpected post-state order")
        require(p.dtype == np.float32 and lam.dtype == np.float32 and star.dtype == np.float32,
                "Carrier dtype changed")
        require(np.isfinite(p).all() and np.isfinite(lam).all() and np.isfinite(star).all(),
                "Non-finite carried third state")
        require(np.count_nonzero(p[~alive]) == 0 and np.count_nonzero(b[~alive]) == 0,
                "Cold-support carrier invariant changed")
        # Interpret the actual stored float32 atoms as a physical symmetric tensor.
        # This diagnostic float64 read is NOT fed back into the bank.
        p64 = p.astype(np.float64)
        lam64 = lam.astype(np.float64)
        cp = (p64*p64) @ (p64*lam64).T
        st = star.astype(np.float64)
        decoded_star = (st.T + 2*np.diag(np.diag(st))) / 3
        B = C.copy()
        np.fill_diagonal(B, np.diag(C)*alive)
        trace = np.diag(B)[:, None]*b[None, :] + 2*B*b[:, None]
        physical_K21 = cp + decoded_star + trace

        indices = np.array([0, 1, 17, 23, 127, 128, 511, 1000, 1023])
        cp32 = plane(p[indices], lam)
        fp_delta = cp[np.ix_(indices, indices)] - cp32
        errors = []
        # Fixed direct entries independently check orientation and the star diagonal.
        for i, j in ((0, 0), (0, 1), (1, 0), (17, 23), (23, 17),
                     (127, 128), (511, 1000), (1023, 1023)):
            terms = lam64*p64[i]*p64[i]*p64[j]
            star_entry = st[j, i]/3 + (2*st[i, i]/3 if i == j else 0)
            trace_entry = B[i, i]*b[j] + 2*B[i, j]*b[i]
            direct = np.sum(terms, dtype=np.float64) + star_entry + trace_entry
            scale = max(1., float(np.sum(np.abs(terms))) + abs(star_entry) + abs(trace_entry))
            errors.append(float(abs(physical_K21[i, j] - direct)/scale))
        require(max(errors) <= 1e-12, f"Actual carrier entry certificate failed at {l}")
        variance = np.diag(C).copy()
        previous_short = dict(layer=l, m=m.copy(), variance=variance)
        save_npz(f"post_{l:02d}.npz", layer=np.array(l), m=m, variance=variance,
                 physical_K21=physical_K21, alive=alive, radial=np.array(radial))
        post_receipts.append(dict(layer=l, carrier_columns=int(p.shape[1]),
                     max_scaled_direct_entry_error=max(errors),
                     sampled_cp_float64_minus_float32_max_abs=float(np.max(np.abs(fp_delta))),
                     sampled_cp_float64_minus_float32_rms=float(np.sqrt(np.mean(fp_delta**2))),
                     actual_carrier_minus_internal_map_plane_rms=float(np.sqrt(np.mean((physical_K21-map_target)**2))),
                     covariance=cov_stats(C)))
        print(json.dumps(dict(rank=args.rank, captured_post_layer=l,
                              carrier_columns=int(p.shape[1]))), flush=True)

    def hook(l, m, C, K21, aux, w):
        nonlocal state, aux13
        x = features(l, m, C, K21, aux, w)
        same = (np.array_equal(x, old["x"][l]) and
                np.array_equal(aux["sigma"].astype(np.float32), old["sigma"][l]) and
                np.array_equal(aux["sigma"], pre_vectors[l]["sigma"]) and
                np.array_equal(aux["alpha"], pre_vectors[l]["alpha"]))
        require(same, f"Prefix feature/sigma/alpha parity failed at {l}")
        prefix_parity.append(bool(same))
        if l == 13:
            require(aux13 is not None, "Pre13 capture did not precede the head")
            # All requested aux are preserved, but post quantities are diagnostics,
            # not causal learner features. The pre K21 is a loss label only.
            for key in ("sigma", "alpha", "pdf", "gate", "gaussian_mean",
                        "pre_v", "pre_k3", "pre_k4", "post_v", "post_k3", "post_k4"):
                aux13[f"aux_{key}"] = aux[key].copy()
            save_npz("pre_13.npz", **aux13)
            aux13 = None
        delta, state = gnn_step(l, x, aux["sigma"], w, state, P)
        return delta

    def final_hook(l, m, aux, w):
        require(l == 15, "Final Q was not called at layer 15")
        x = final_scalar_features(l, m, aux, w)
        h = final_input(x, w, state, P)
        delta = final_delta(h, aux["sigma"], Q)
        final_saved.update(x=x.copy(), h=h.copy(), sigma=aux["sigma"].copy(),
                           alpha=aux["alpha"].copy(), host=m.copy(), delta=delta.copy())
        return delta

    # Add observation-only callbacks to the actual generated reference source.
    # Stripping these two additions must recover the frozen source byte-for-byte.
    source = parent.source
    edits = [
        ("        k4 = radial*vv*vv",
         "        _capture_pre(l, m, C, mu, Cpre, K21, alive, radial)\n        k4 = radial*vv*vv"),
        ("        previous_alive = alive",
         "        _capture_post(l, m, C, p, lam, star, b, alive, radial_new, target)\n        previous_alive = alive"),
    ]
    for old_text, new_text in edits:
        require(source.count(old_text) == 1, f"Reference instrumentation anchor changed: {old_text}")
        source = source.replace(old_text, new_text)
    recovered = source
    for old_text, new_text in reversed(edits):
        recovered = recovered.replace(new_text, old_text)
    require(recovered == parent.source, "Instrumentation changed bank arithmetic")
    env = dict(parent.env, _capture_pre=capture_pre, _capture_post=capture_post)
    exec(compile(source, "<two_memory_observation_only_reference_v1>", "exec"), env)
    try:
        with threadpool_limits(limits=2):
            prediction, _ = env["predict"](W, hook=hook, final_hook=final_hook,
                gate_floor=.01, cov32=True, metric="covariance")
            backend = threadpool_info()
        require(len(pre_vectors) == 15 and len(post_receipts) == 13 and
                len(prefix_parity) == 15, "Incomplete prefix/post capture")
        final_saved["prediction"] = np.maximum(prediction, 0)
        parity = {k: bool(np.array_equal(final_saved[k], old_final[k]))
                  for k in ("prediction", "sigma", "host", "delta", "h")}
        raw_prefix_parity = bool(np.array_equal(prediction[:15], old["baseline"][:15]))
        require(all(parity.values()) and raw_prefix_parity,
                f"Actual P0/final Q baseline parity failed: {parity}; prefix={raw_prefix_parity}")
        save_npz("pre_vectors_00_14.npz", **{k: np.stack([r[k] for r in pre_vectors])
                                            for k in pre_vectors[0]})
        save_npz("outputs_and_final_Q.npz", **final_saved,
                 prediction_unclipped=prediction,
                 prediction_served_float32=final_saved["prediction"].astype(np.float32))
        end_sources = {name: sha(HERE / name) for name in source_names}
        end_inputs = {name: sha(path) for name, path in input_paths.items()}
        require(end_sources == source_hashes and end_inputs == input_hashes,
                "A frozen source, payload, weight, or parity input changed during capture")
        require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules),
                "Unexpected Torch import during capture")
        artifacts = {p.name: dict(sha256=sha(p), bytes=p.stat().st_size)
                     for p in sorted(out.glob("*.npz"))}
        result = dict(passed=True, rank=args.rank, target_access=False,
              source_hashes=source_hashes, input_hashes=input_hashes,
              generated_reference_sha256=hashlib.sha256(parent.source.encode()).hexdigest(),
              instrumented_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
              source_restoration_byte_parity=True, prefix_feature_sigma_alpha_byte_parity=prefix_parity,
              prefix_unclipped_mean_byte_parity=raw_prefix_parity, actual_final_Q_byte_parity=parity,
              post_planes=post_receipts, artifacts=artifacts,
              pre_covariance=[dict(layer=l, diagonal_min=float(r["variance"].min()),
                  negative_diagonal_count=int((r["variance"] < 0).sum()),
                  below_unit_floor_count=int((r["variance"] < 1e-10).sum()))
                  for l, r in enumerate(pre_vectors)],
              wall_s=time.perf_counter()-started,
              peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
              backend=backend, torch_imported=False,
              interpretation="Target-free actual-bank action data; no student capacity, raw-score, or rollout claim")
        write_json(out / "result.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
    except BaseException as exc:
        write_json(out / "failure.json", dict(passed=False, rank=args.rank, target_access=False,
                   exception_type=type(exc).__name__, message=str(exc),
                   wall_s=time.perf_counter()-started, source_hashes=source_hashes))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rank", type=int, default=35, choices=range(35, 59))
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--execute", action="store_true", help="Run only after root source review and authorization")
    modes.add_argument("--dry-check", action="store_true", help="Inert plan only; no model/array imports or reads")
    ap.add_argument("--qualified-panel", action="store_true", help="Explicit later stage; requires unchanged passing rank-35 receipt")
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps(dict(execute=False, rank=args.rank, fixed_ranks=[35, 58],
              first_gate="Root review, then rank 35 only", output=str(OUT_ROOT / f"row_{args.rank:04d}"),
              planned_post_layers=[0, 12], planned_action_layer=13,
              weights_copied=False, target_access=False, model_imported=False,
              approximate_uncompressed_bytes_per_rank=126_000_000)))
        return
    execute(args)


if __name__ == "__main__":
    main()
