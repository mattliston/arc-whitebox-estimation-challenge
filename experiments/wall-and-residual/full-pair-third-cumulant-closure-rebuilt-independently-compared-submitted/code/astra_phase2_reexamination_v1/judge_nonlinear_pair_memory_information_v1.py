"""Inert-by-default judge of the fixed nonlinear historical-pair information test.

PREPARED FOR ROOT REVIEW; not executed by its author. This reads only the
specified deterministic bank captures after --execute-reviewed-judge. There
is no MC/private truth, optimizer, sweep, rollout, candidate or submission.
Every prediction feature is frozen before its current pre13 K21 is decoded;
the sole two-scalar TRAIN fit is frozen before any VALID action is decoded.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
CAPTURE_NAME = "capture_two_memory_teacher_action_replay_v2.py"
PARENT_CAPTURE_NAME = "capture_two_memory_teacher_action_v1.py"
RESOURCE_NOTE_NAME = "TWO_MEMORY_RESOURCE_REPLAY_V2.md"
PROTOCOL_NAME = "NONLINEAR_PAIR_MEMORY_INFORMATION_PROTOCOL_V1.md"
CAPTURE_PHYSICS_NAME = "pair_memory_action_physics_v1.py"
PHYSICS_NAME = "pair_memory_action_physics_v2.py"
RESPONSE_NOTE_NAME = "PAIR_MEMORY_RESPONSE_EXACTNESS_V2.md"
REPLAY_QUALIFICATION_NAME = "two_memory_capture_replay_v2_qualification.json"
CAPTURE_ROOT = HERE / "rows_two_memory_teacher_action_v2"
OUTPUT = HERE / "nonlinear_pair_memory_information_v1"
RANKS = tuple(range(35, 59))
VALID = (37, 39, 42, 48)
TRAIN = tuple(rank for rank in RANKS if rank not in VALID)
ARMS = ("history", "reset")
CONSUMERS = ("covariance", "next_mean")
N = 1024
RSS_LIMIT = 6 * 1024**3
PINNED = {
    CAPTURE_NAME: "82103b513f517cd555803260dd6f2c54aeeeb07a3a9e55254b12162baf646d7e",
    PARENT_CAPTURE_NAME: "4eb6a294651800c137614cbc1ad8a1bab99a206ad220d76e86f74b184056d608",
    RESOURCE_NOTE_NAME: "f32c9e8097d1845b4882930f9d45ae8aaab89499d9f4493f964123a763a05d4c",
    PROTOCOL_NAME: "6b8dc12bcadaa038eb24eb941a2656b83cd6a76947e11a43f0f6c61addd2acf9",
    CAPTURE_PHYSICS_NAME: "167127269de079238ad62fd5a3243d516f9f187147f410a5b3ee85071ae951a1",
    PHYSICS_NAME: "649d1523b2a0c764373b4a779231e74a2397368649d7e1f81c042401b31e6333",
    RESPONSE_NOTE_NAME: "915230d47b0c1cdd550c789e4434cf9c21ed5a63e9f5df5f378a4858d96f44d0",
    "certify_pair_memory_response_v2.py":
        "18f3d3ca2f01c22354c0af58d778d097280bb548c0dddcf647889a6d49151b3b",
    "qualify_two_memory_capture_replay_v2.py":
        "3b5e25198d342e760aea5827b5100e99630bdbdb89578f1adc05945b21597247",
    REPLAY_QUALIFICATION_NAME:
        "af8c2b0cb66b6730af9d93b5aa0e56efcc97ba3e825093c4926f4bbd3805f328",
    "pair_memory_response_certificate_v2.json":
        "846254792751b6b01bd658fcb477ea67605c6ed671685e2c3910c87d4b3e8e58",
    "pair_memory_response_certificate_v1.json":
        "4356ba6636cc6c8406229455ca3823d5a7d4d4d225084031525167fa7310692d",
    "certify_pair_memory_response_v1.py":
        "9c1b4208bd9961e7e7e4a29dcdeddc0b83766bdb6d2a00619390f2096a42f655",
    "TWO_MEMORY_TEACHER_ACTION_CAPTURE_DESIGN_V1.md":
        "5f5eedee44b71b3b3c451f7bc96c3f80322585e76728e728345ffacb6ac29504",
}
ARTIFACT_NAMES = tuple(f"post_{layer:02d}.npz" for layer in range(13)) + (
    "pre_13.npz", "pre_vectors_00_14.npz", "outputs_and_final_Q.npz")


class Inconclusive(RuntimeError):
    """The protocol does not define a valid numerical decision."""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    with Path(path).open("r") as stream:
        return json.load(stream)


def peak_rss():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def check_process():
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch with nice >=10")
    require("/anaconda3/" not in sys.executable,
            "Use the repository NumPy .venv interpreter, never AP/Anaconda")
    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules),
            "Torch must never be imported in this judge")
    require(peak_rss() <= RSS_LIMIT, "Judge peak RSS exceeded 6 GiB")


def checked_file(path, manifest, expected=None):
    path = Path(path).resolve()
    require(path.is_relative_to(HERE), f"Unexpected external input: {path}")
    require(path.is_file(), f"Missing frozen file: {path}")
    stat = path.stat()
    item = dict(sha256=sha(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if expected is not None:
        require(item["sha256"] == expected, f"Digest mismatch: {path.name}")
    key = str(path.relative_to(HERE))
    if key in manifest:
        require(item == manifest[key], f"Frozen file changed during preflight: {key}")
    manifest[key] = item
    return item


def recheck_manifest(manifest):
    for key, original in manifest.items():
        path = HERE / key
        require(path.is_file(), f"Frozen file disappeared: {key}")
        stat = path.stat()
        actual = dict(sha256=sha(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        require(actual == original, f"Frozen input/source changed: {key}")
    for rank in RANKS:
        require(not (CAPTURE_ROOT / f"row_{rank:04d}" / "failure.json").exists(),
                f"Capture acquired a failure receipt: {rank}")


def qualify_replay_receipt(manifest):
    checked_file(HERE / REPLAY_QUALIFICATION_NAME, manifest)
    receipt = read_json(HERE / REPLAY_QUALIFICATION_NAME)
    require(receipt["passed"] is True and receipt["ranks"] == [35, 36, 37]
            and receipt["arrays_read_for_numerical_parity_only"] is True
            and receipt["predictive_features_evaluated"] is False
            and receipt["current_action_loss_evaluated"] is False
            and 0 < receipt["peak_rss_bytes"] <= RSS_LIMIT,
            "The resource-only three-row replay qualification did not pass")
    expected_sources = {name: PINNED[name] for name in (
        PARENT_CAPTURE_NAME, CAPTURE_NAME, RESOURCE_NOTE_NAME,
        "qualify_two_memory_capture_replay_v2.py")}
    require(receipt["source_hashes"] == expected_sources,
            "Replay qualification source ownership mismatch")
    for name, expected in expected_sources.items():
        checked_file(HERE / name, manifest, expected)
    expected_files = {
        str(Path(f"rows_two_memory_teacher_action_v{version}") / f"row_{rank:04d}" / name)
        for version in (1, 2) for rank in (35, 36, 37)
        for name in (*ARTIFACT_NAMES, "result.json")}
    require(set(receipt["checked_files"]) == expected_files,
            "Replay qualification omitted or added an unexpected input artifact")
    for name, expected in receipt["checked_files"].items():
        checked_file(HERE / name, manifest, expected)
    expected_comparisons = {
        (f"post_{layer:02d}.npz", key) for layer in range(13)
        for key in ("layer", "m", "variance", "physical_K21", "alive", "radial")}
    expected_comparisons.update({("pre_13.npz", key) for key in (
        "pre_C", "pre_K21_loss_only", "pre_mu", "pre_alive", "previous_m", "previous_variance",
        "aux_sigma", "aux_alpha", "aux_pdf", "aux_gate", "aux_gaussian_mean", "aux_pre_v",
        "aux_pre_k3", "aux_pre_k4", "aux_post_v", "aux_post_k3", "aux_post_k4")})
    expected_comparisons.update({("pre_vectors_00_14.npz", key)
                                for key in ("mu", "variance", "sigma", "alpha", "alive", "radial")})
    expected_comparisons.update({("outputs_and_final_Q.npz", key) for key in (
        "x", "h", "sigma", "alpha", "host", "delta", "prediction",
        "prediction_unclipped", "prediction_served_float32")})
    require([row["rank"] for row in receipt["rows"]] == [35, 36, 37],
            "Replay qualification row list changed")
    for row in receipt["rows"]:
        rank = row["rank"]
        require(0 < row["v2_peak_rss_bytes"] <= RSS_LIMIT,
                "A v2 replay exceeded the fixed resource cap")
        for version in (1, 2):
            name = str(Path(f"rows_two_memory_teacher_action_v{version}") / f"row_{rank:04d}" / "result.json")
            require(row[f"v{version}_receipt_sha256"] == receipt["checked_files"][name],
                    "Replay qualification receipt binding failed")
        comparisons = row["comparisons"]
        require(len(comparisons) == len(expected_comparisons)
                and {(item["artifact"], item["array"]) for item in comparisons} == expected_comparisons,
                "Replay qualification did not check every fixed artifact/key exactly once")
        for item in comparisons:
            if item["artifact"].startswith("post_") and item["array"] == "physical_K21":
                require(0 <= item["relative_l2"] <= 1e-10,
                        "Blocked physical plane exceeds the fixed numerical-parity tolerance")
            else:
                require(item["byte_equal"] is True, "Original bank-array byte parity failed")


def prepare_provenance(manifest):
    # Hashing NPZ/NPY bytes in this routine never decodes any array.
    for name, expected in PINNED.items():
        checked_file(HERE / name, manifest, expected)
    checked_file(Path(__file__), manifest)
    import capture_two_memory_teacher_action_v1 as capture

    require(capture.HERE.resolve() == HERE
            and capture.OUT_ROOT.resolve() == HERE / "rows_two_memory_teacher_action_v1",
            "Frozen capture parent resolves to another directory")
    source_names = list(capture.EXPECTED) + [CAPTURE_NAME, capture.DESIGN,
                                            PROTOCOL_NAME, CAPTURE_PHYSICS_NAME,
                                            PARENT_CAPTURE_NAME, RESOURCE_NOTE_NAME]
    source_hashes = {}
    for name in source_names:
        item = checked_file(HERE / name, manifest, capture.EXPECTED.get(name))
        source_hashes[name] = item["sha256"]
    certificate = read_json(HERE / "pair_memory_response_certificate_v1.json")
    require(certificate["passed"] is True and certificate["target_access"] is False
            and certificate["torch_imported"] is False
            and certificate["wide_execution"] is False,
            "The exact-response certificate did not qualify")
    for name, expected in certificate["source_hashes"].items():
        checked_file(HERE / name, manifest, expected)
    checked_file(HERE / "pair_memory_response_certificate_v2.json", manifest)
    certificate_v2 = read_json(HERE / "pair_memory_response_certificate_v2.json")
    require(certificate_v2["passed"] is True and certificate_v2["target_access"] is False
            and certificate_v2["torch_imported"] is False and certificate_v2["wide_execution"] is False
            and certificate_v2["all_source_hashes_unchanged"] is True
            and certificate_v2["rng_seed"] == 930119 and certificate_v2["fixture_count"] == 10
            and certificate_v2["dimension"] == 7 and certificate_v2["fixed_tolerance"] == 2e-11
            and certificate_v2["worst_relative_error"] <= 2e-11
            and certificate_v2["worst_symmetric_v1_vs_v2_difference"] <= 2e-11
            and certificate_v2["asymmetric_v1_error_witness"] > 1e-10,
            "The directed-correlation/floor response certificate did not qualify")
    expected_certificate_sources = {
        "certify_pair_memory_response_v2.py", CAPTURE_PHYSICS_NAME, PHYSICS_NAME,
        "radial_mehler.py", "k3_radial_numpy.py", RESPONSE_NOTE_NAME, PROTOCOL_NAME}
    require(set(certificate_v2["source_hashes"]) == expected_certificate_sources,
            "Incomplete v2 response-certificate source inventory")
    for name, expected in certificate_v2["source_hashes"].items():
        checked_file(HERE / name, manifest, expected)
    qualify_replay_receipt(manifest)

    rows = {}
    for rank in RANKS:
        row_dir = CAPTURE_ROOT / f"row_{rank:04d}"
        require(not (row_dir / "failure.json").exists(), f"Failed capture: {rank}")
        checked_file(row_dir / "result.json", manifest)
        checked_file(row_dir / "started.json", manifest)
        receipt = read_json(row_dir / "result.json")
        started = read_json(row_dir / "started.json")
        require(receipt["passed"] is True and receipt["rank"] == rank
                and receipt["target_access"] is False and receipt["torch_imported"] is False,
                f"Unqualified/non-target-free capture: {rank}")
        require(started["rank"] == rank and started["target_access"] is False
                and started["threads"] == 2 and started["fixed_unit_variance_floor"] == 1e-10,
                f"Capture start contract changed: {rank}")
        require(receipt["source_hashes"] == source_hashes
                and started["source_hashes"] == source_hashes,
                f"Capture source/PQ/protocol ownership mismatch: {rank}")
        require(receipt["source_restoration_byte_parity"] is True
                and receipt["prefix_unclipped_mean_byte_parity"] is True
                and receipt["prefix_feature_sigma_alpha_byte_parity"] == [True] * 15,
                f"Prefix/observation parity failed: {rank}")
        final_parity = receipt["actual_final_Q_byte_parity"]
        require(set(final_parity) == {"prediction", "sigma", "host", "delta", "h"}
                and all(value is True for value in final_parity.values()),
                f"Actual final Q parity failed: {rank}")
        require(0 < receipt["peak_rss_bytes"] <= RSS_LIMIT,
                f"Capture peak RSS exceeds the fixed 6 GiB limit: {rank}")
        blas = [entry for entry in receipt["backend"] if entry.get("user_api") == "blas"]
        require(all(entry["num_threads"] == 2 for entry in blas),
                f"A reported capture BLAS backend violated the two-thread request: {rank}")
        require(len(receipt["post_planes"]) == 13
                and [entry["layer"] for entry in receipt["post_planes"]] == list(range(13))
                and len(receipt["pre_covariance"]) == 15,
                f"Incomplete captured chronology: {rank}")
        for entry in receipt["post_planes"]:
            require(0 <= entry["max_scaled_direct_entry_error"] <= 1e-12,
                    f"Actual represented-plane entry certificate failed: {rank}")
        require(receipt["input_hashes"] == started["input_hashes"],
                f"Capture input hashes differ within its receipts: {rank}")
        expected_paths = {
            f"prefix/{name}": HERE / "rows_current_transport_shadow_v2" / f"row_{rank:04d}" / name
            for name in ("weights.npy", "baseline.npy", "x.npy", "sigma.npy", "pre_cov.npy", "k21.npy")}
        expected_paths.update({
            f"final/{name}": HERE / "rows_matrix_error_observer_v1" / f"row_{rank:04d}" / name
            for name in ("final.npz", "result.json")})
        require(started["input_paths"] == {key: str(path.relative_to(HERE))
                                          for key, path in expected_paths.items()},
                f"Capture input paths differ from the fixed public-development row: {rank}")
        require(set(receipt["input_hashes"]) == set(expected_paths),
                f"Incomplete input digest inventory: {rank}")
        for key, path in expected_paths.items():
            checked_file(path, manifest, receipt["input_hashes"][key])
        require(set(receipt["artifacts"]) == set(ARTIFACT_NAMES),
                f"Unexpected/incomplete capture artifact inventory: {rank}")
        require({path.name for path in row_dir.glob("*.npz")} == set(ARTIFACT_NAMES),
                f"Undeclared captured array artifact: {rank}")
        for name in ARTIFACT_NAMES:
            item = checked_file(row_dir / name, manifest, receipt["artifacts"][name]["sha256"])
            require(item["bytes"] == receipt["artifacts"][name]["bytes"],
                    f"Capture artifact size mismatch: {rank}:{name}")
        rows[rank] = dict(directory=row_dir, weights=expected_paths["prefix/weights.npy"],
                          receipt=receipt)
        check_process()
    return source_hashes, rows


def finite_array(np, value, shape, name, dtype=None):
    value = np.asarray(value)
    require(value.shape == shape, f"Wrong shape for {name}: {value.shape}")
    if dtype is not None:
        require(value.dtype == np.dtype(dtype), f"Wrong dtype for {name}: {value.dtype}")
    require(np.isfinite(value).all(), f"Nonfinite predictor/response input: {name}")
    return value


def rms(np, value):
    result = float(np.sqrt(np.mean(value * value)))
    require(np.isfinite(result), "Nonfinite RMS; no normalization repair is allowed")
    return result


def array_sha(np, value):
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def build_causal_features(np, physics, rank, row, output):
    started = time.perf_counter()
    row_dir = row["directory"]
    weight = np.load(row["weights"], mmap_mode="r", allow_pickle=False)
    finite_array(np, weight, (16, N, N), "weights", np.float32)
    with np.load(row_dir / "pre_vectors_00_14.npz", allow_pickle=False) as data:
        # These are lower-order input units/support only. No query K21 is in this file.
        pre_sigma = finite_array(np, data["sigma"], (15, N), "pre sigma", np.float64)
        pre_variance = finite_array(np, data["variance"], (15, N), "pre variance", np.float64)
        pre_alive = finite_array(np, data["alive"], (15, N), "pre alive", np.bool_)
    require(np.all(pre_sigma > 0)
            and np.array_equal(pre_sigma, np.sqrt(np.maximum(pre_variance, 1e-10))),
            "Pre sigma ownership/floor mismatch")
    history = np.zeros((N, N), np.float64)
    layers = []
    previous_sigma = None
    last_mean = None
    last_variance = None
    last_d = None
    reset = None
    for layer in range(13):
        with np.load(row_dir / f"post_{layer:02d}.npz", allow_pickle=False) as data:
            require(int(data["layer"]) == layer, "Post history layer mismatch")
            variance = finite_array(np, data["variance"], (N,), "post variance", np.float64)
            mean = finite_array(np, data["m"], (N,), "post mean", np.float64)
            physical = finite_array(np, data["physical_K21"], (N, N),
                                    "actual prior represented K21", np.float64)
            finite_array(np, data["alive"], (N,), "post alive", np.bool_)
        sigma = np.sqrt(np.maximum(variance, 1e-10))
        d = physical / (sigma[:, None]**2 * sigma[None, :])
        finite_array(np, d, (N, N), "standardized prior D")
        source = physics.memory_source(d)
        source_normalized = physics.normalized(source)
        if layer == 0:
            u = np.zeros((N, N), np.float64)
        else:
            v = previous_sigma[:, None] * weight[layer].astype(np.float64) / pre_sigma[layer][None, :]
            u = v.T @ history @ v
            finite_array(np, u, (N, N), "transported prior H")
        history = np.tanh(.5 * physics.normalized(u) + .5 * source_normalized)
        reset = np.tanh(.5 * source_normalized)
        finite_array(np, history, (N, N), "causal H")
        finite_array(np, reset, (N, N), "reset H")
        layers.append(dict(layer=layer, variance_min=float(variance.min()),
                           negative_variance_count=int((variance < 0).sum()),
                           variance_floor_count=int((variance < 1e-10).sum()),
                           d_rms=rms(np, d), source_rms=rms(np, source),
                           source_divisor=max(rms(np, source), 1e-3),
                           u_rms=rms(np, u), u_divisor=max(rms(np, u), 1e-3),
                           history_rms=rms(np, history), reset_rms=rms(np, reset),
                           history_max_abs=float(np.max(np.abs(history)))))
        previous_sigma = sigma
        last_mean, last_variance, last_d = mean, variance, d
        check_process()
    v = previous_sigma[:, None] * weight[13].astype(np.float64) / pre_sigma[13][None, :]
    k = physics.star_action(last_d, v)
    finite_array(np, k, (N, N), "full query star K")
    mask = pre_alive[13, :, None] & pre_alive[13, None, :]
    np.fill_diagonal(mask, False)
    masked_k = mask * k
    f0 = physics.normalized(masked_k)
    feature = {}
    query_audit = {}
    for arm, memory in (("history", history), ("reset", reset)):
        u = v.T @ memory @ v
        j = k @ u  # All summation directions retained; mask only after this full product.
        finite_array(np, u, (N, N), f"{arm} query U")
        finite_array(np, j, (N, N), f"{arm} full KU")
        masked_j = mask * j
        f1 = physics.normalized(masked_j)
        finite_array(np, f1, (N, N), f"{arm} F1")
        require(np.count_nonzero(f1[~mask]) == 0, "A feature escaped the fixed live domain")
        feature[arm] = f1
        query_audit[arm] = dict(u_rms=rms(np, u), j_full_rms=rms(np, j),
                               j_masked_rms=rms(np, masked_j),
                               f1_divisor=max(rms(np, masked_j), 1e-3),
                               f1_rms=rms(np, f1), f1_sha256=array_sha(np, f1))
    require(np.count_nonzero(f0[~mask]) == 0, "Common feature escaped the fixed live domain")
    frozen = dict(rank=rank, split="train" if rank in TRAIN else "valid",
                  query="post12_to_pre13", prior_layers=list(range(13)),
                  feature_frozen_utc=utc_now(), current_pre13_K21_decoded=False,
                  current_pre13_diagonal_used_by_predictor=False,
                  prior_post_diagonals_used_in_B=True,
                  predictor_inputs=["weights", "pre_sigma_00_13", "pre_alive_13",
                                    "post_00_12_variance", "post_00_12_actual_physical_K21"],
                  unused_loaded_metadata=["post_00_12_mean", "pre_sigma_14", "pre_variance_14"],
                  active_count=int(pre_alive[13].sum()), live_offdiagonal_entries=int(mask.sum()),
                  k_full_rms=rms(np, k), k_masked_rms=rms(np, masked_k),
                  f0_divisor=max(rms(np, masked_k), 1e-3), f0_rms=rms(np, f0),
                  f0_sha256=array_sha(np, f0), k_sha256=array_sha(np, k),
                  arms=query_audit, layers=layers,
                  wall_seconds=time.perf_counter() - started, peak_rss_bytes=peak_rss())
    marker = output / f"features_frozen_{rank:04d}.json"
    write_json(marker, frozen)
    # This function has never opened pre_13.npz. Return features before its label is decoded.
    return dict(weight=weight, k=k, mask=mask, f0=f0, f1=feature,
                pre_sigma=pre_sigma, pre_variance=pre_variance, pre_alive=pre_alive,
                last_mean=last_mean, last_variance=last_variance,
                frozen=frozen, frozen_path=marker, frozen_sha256=sha(marker))


def sufficient_statistics(np, x0, x1, target):
    gram = np.array([[np.mean(x0*x0), np.mean(x0*x1)],
                     [np.mean(x0*x1), np.mean(x1*x1)]], dtype=np.float64)
    rhs = np.array([np.mean(x0*target), np.mean(x1*target)], dtype=np.float64)
    energy = float(np.mean(target*target))
    require(np.isfinite(gram).all() and np.isfinite(rhs).all() and np.isfinite(energy),
            "Nonfinite sufficient statistics")
    return dict(gram=gram.tolist(), rhs=rhs.tolist(), target_energy=energy,
                coordinate_count=int(target.size))


def collect_one(np, physics, rank, row, output, progress):
    started = time.perf_counter()
    features = build_causal_features(np, physics, rank, row, output)
    require(sha(features["frozen_path"]) == features["frozen_sha256"],
            "Causal feature-freeze marker changed before label decoding")
    require(features["frozen"]["current_pre13_K21_decoded"] is False,
            "Invalid causal feature-freeze audit")
    progress["features_frozen_ranks"].append(rank)
    with np.load(row["directory"] / "pre_13.npz", allow_pickle=False) as data:
        covariance = finite_array(np, data["pre_C"], (N, N), "pre13 C", np.float64)
        sigma = finite_array(np, data["aux_sigma"], (N,), "pre13 sigma", np.float64)
        alpha = finite_array(np, data["aux_alpha"], (N,), "pre13 alpha", np.float64)
        pdf = finite_array(np, data["aux_pdf"], (N,), "pre13 pdf", np.float64)
        gate = finite_array(np, data["aux_gate"], (N,), "pre13 gate", np.float64)
        alive = finite_array(np, data["pre_alive"], (N,), "pre13 alive", np.bool_)
        mu = finite_array(np, data["pre_mu"], (N,), "pre13 mu", np.float64)
        previous_mean = finite_array(np, data["previous_m"], (N,), "previous m", np.float64)
        previous_variance = finite_array(np, data["previous_variance"], (N,),
                                         "previous variance", np.float64)
        require(np.array_equal(sigma, features["pre_sigma"][13])
                and np.array_equal(np.diag(covariance), features["pre_variance"][13])
                and np.array_equal(alive, features["pre_alive"][13])
                and np.array_equal(previous_mean, features["last_mean"])
                and np.array_equal(previous_variance, features["last_variance"])
                and np.array_equal(alpha, mu/sigma),
                "Captured query/past boundary coherence failed")
        require(np.all(np.diag(covariance) > 0) and np.all(sigma > 0)
                and np.all(pdf >= 0) and np.all((gate >= 0) & (gate <= 1)),
                "Invalid exact-response inputs; no covariance repair is allowed")
        # FIRST query action-target decode. Everything predictive already exists and is hashed.
        progress["action_targets_decoded_ranks"].append(rank)
        target_decoded_utc = utc_now()
        physical_target = finite_array(np, data["pre_K21_loss_only"], (N, N),
                                       "pre13 loss-only actual bank K21", np.float64)
    target = features["mask"] * (
        physical_target/(sigma[:, None]**2*sigma[None, :]) - features["k"])
    finite_array(np, target, (N, N), "masked standardized action residual")
    require(np.count_nonzero(target[~features["mask"]]) == 0,
            "Current query diagonal/cold pairs entered the loss")
    with np.load(row["directory"] / "pre_vectors_00_14.npz", allow_pickle=False) as data:
        next_alpha = finite_array(np, data["alpha"], (15, N), "pre alpha", np.float64)[14]
        next_mu = finite_array(np, data["mu"], (15, N), "pre mu", np.float64)[14]
    next_sigma = features["pre_sigma"][14]
    require(np.all(next_sigma > 0) and np.array_equal(next_alpha, next_mu/next_sigma),
            "Next consumer sigma/alpha ownership mismatch")
    # Identical Gaussian pdf arithmetic to the frozen bank; these values are LOSS ONLY.
    next_pdf = np.exp(-next_alpha*next_alpha/2)/np.sqrt(2*np.pi)
    finite_array(np, next_pdf, (N,), "next Gaussian pdf")
    next_weight = features["weight"][14].astype(np.float64)

    def covariance_response(delta):
        result = physics.full_cubic_covariance_response(delta, covariance, sigma, alpha, pdf, gate)
        return finite_array(np, result, (N, N), "covariance response")

    def next_response(delta_c):
        result = physics.next_gaussian_mean_response(delta_c, next_weight, next_sigma, next_pdf)
        return finite_array(np, result, (N,), "next leading-mean response")

    common_covariance = covariance_response(features["f0"])
    common_next = next_response(common_covariance)
    target_covariance = covariance_response(target)
    target_next = next_response(target_covariance)
    arms = {}
    for arm in ARMS:
        extra_covariance = covariance_response(features["f1"][arm])
        extra_next = next_response(extra_covariance)
        arms[arm] = dict(
            covariance=sufficient_statistics(np, common_covariance, extra_covariance, target_covariance),
            next_mean=sufficient_statistics(np, common_next, extra_next, target_next))
    for consumer in CONSUMERS:
        require(arms["history"][consumer]["target_energy"] == arms["reset"][consumer]["target_energy"],
                "Arms did not receive identical action targets/consumers")
    record = dict(rank=rank, split="train" if rank in TRAIN else "valid", arms=arms,
                  feature_freeze_sha256=features["frozen_sha256"],
                  feature_freeze_utc=features["frozen"]["feature_frozen_utc"],
                  action_target_decoded_utc=target_decoded_utc,
                  action_target_kind="Noiseless actual-bank offdiagonal pre13 K21 only",
                  mc_truth_accessed=False, private_truth_accessed=False,
                  current_query_diagonal_used_by_predictor=False,
                  current_query_diagonal_used_by_loss=False,
                  covariance_response="Directed-rho v2 r=0..2 cubic action, original variance denominator",
                  next_response="Frozen pre14 Gaussian pdf/(2*sigma), loss-only; not full Jacobian",
                  standardized_target_rms=rms(np, target),
                  pre13_original_variance_min=float(np.diag(covariance).min()),
                  pre13_covariance_antisymmetry_relative_l2=float(
                      np.linalg.norm(covariance-covariance.T)/np.linalg.norm(covariance)),
                  pre13_covariance_antisymmetry_max_abs=float(np.max(np.abs(covariance-covariance.T))),
                  covariance_or_response_symmetrized=False,
                  pre13_sigma_floor_count=int((np.diag(covariance) < 1e-10).sum()),
                  pre14_sigma_floor_count=int((features["pre_variance"][14] < 1e-10).sum()),
                  peak_rss_bytes=peak_rss(), wall_seconds=time.perf_counter()-started)
    check_process()
    write_json(output / f"rank_{rank:04d}.json", record)
    print(json.dumps(dict(event="row_sufficient_statistics_saved", rank=rank, split=record["split"],
                          wall_seconds=record["wall_seconds"], peak_rss_bytes=record["peak_rss_bytes"])),
          flush=True)
    del features, physical_target, target, common_covariance, target_covariance, extra_covariance
    gc.collect()
    return record


def fit_once(np, training):
    require(set(training) == set(TRAIN), "The sole fit requires exactly the frozen twenty TRAIN ranks")
    energies = {consumer: float(np.mean([
        training[rank]["arms"]["history"][consumer]["target_energy"] for rank in TRAIN]))
        for consumer in CONSUMERS}
    if any(not np.isfinite(value) or value <= 0 for value in energies.values()):
        raise Inconclusive("Zero/nonfinite TRAIN normalization energy")
    fitted = {}
    for arm in ARMS:
        gram = np.zeros((2, 2), np.float64)
        rhs = np.zeros(2, np.float64)
        for consumer in CONSUMERS:
            gram += np.mean([np.asarray(training[rank]["arms"][arm][consumer]["gram"])
                             for rank in TRAIN], axis=0) / energies[consumer]
            rhs += np.mean([np.asarray(training[rank]["arms"][arm][consumer]["rhs"])
                            for rank in TRAIN], axis=0) / energies[consumer]
        ridge = 1e-6 * float(np.trace(gram)) / 2
        if not np.isfinite(gram).all() or not np.isfinite(rhs).all() or not np.isfinite(ridge) or ridge <= 0:
            raise Inconclusive("Invalid normalized normal equations/ridge")
        regularized = gram + np.eye(2)*ridge
        try:
            beta = np.linalg.solve(regularized, rhs)
        except np.linalg.LinAlgError as exc:
            raise Inconclusive("The sole prescribed ridge solve failed") from exc
        if not np.isfinite(beta).all():
            raise Inconclusive("The sole prescribed ridge solve returned nonfinite coefficients")
        eigenvalues = np.linalg.eigvalsh(gram)
        condition = float(np.linalg.cond(gram))
        fitted[arm] = dict(beta=beta.tolist(), normalized_gram=gram.tolist(),
                           normalized_rhs=rhs.tolist(), ridge=ridge,
                           gram_eigenvalues=eigenvalues.tolist(),
                           gram_condition=condition if np.isfinite(condition) else None,
                           regularized_condition=float(np.linalg.cond(regularized)),
                           solve_residual_norm=float(np.linalg.norm(regularized@beta-rhs)))
    return dict(train_ranks=list(TRAIN), valid_ranks=list(VALID),
                train_normalization_energies=energies, arms=fitted,
                solve_count_per_arm=1, ridge_rule="1e-6*trace(normalized_Gram)/2",
                feature_count=2, frozen_utc=utc_now(), valid_action_targets_decoded=False,
                optimizer_or_checkpoint_search=False, mc_truth_accessed=False)


def residual_energy(np, statistics, beta):
    # Wider accumulation reduces cancellation without clipping/repairing negative energies.
    b = np.asarray(beta, dtype=np.longdouble)
    gram = np.asarray(statistics["gram"], dtype=np.longdouble)
    rhs = np.asarray(statistics["rhs"], dtype=np.longdouble)
    energy = np.longdouble(statistics["target_energy"]) - 2*np.dot(b, rhs) + b@gram@b
    result = float(energy)
    if not np.isfinite(result) or result < 0:
        raise Inconclusive("Nonfinite/negative residual squared energy from sufficient statistics")
    return result


def judge_once(np, validation, fit):
    require(set(validation) == set(VALID), "Judge requires exactly the four frozen VALID ranks")
    per_rank = {}
    for rank in VALID:
        row = validation[rank]
        values = {}
        for consumer in CONSUMERS:
            target_energy = row["arms"]["history"][consumer]["target_energy"]
            if not np.isfinite(target_energy) or target_energy <= 0:
                raise Inconclusive("Zero/nonfinite held-out zero-correction response energy")
            values[consumer] = dict(zero_correction_mse=target_energy)
            for arm in ARMS:
                mse = residual_energy(np, row["arms"][arm][consumer], fit["arms"][arm]["beta"])
                values[consumer][arm] = dict(mse=mse, ratio_to_zero=mse/target_energy)
        reset_next = values["next_mean"]["reset"]["mse"]
        if reset_next <= 0:
            raise Inconclusive("Zero RESET held-out next-response denominator")
        values["history_next_ratio_to_reset"] = values["next_mean"]["history"]["mse"]/reset_next
        values["history_next_beats_reset"] = bool(values["next_mean"]["history"]["mse"] < reset_next)
        per_rank[str(rank)] = values
    pooled = {}
    for consumer in CONSUMERS:
        zero = float(np.mean([per_rank[str(rank)][consumer]["zero_correction_mse"] for rank in VALID]))
        pooled[consumer] = dict(zero_correction_mse=zero)
        for arm in ARMS:
            mse = float(np.mean([per_rank[str(rank)][consumer][arm]["mse"] for rank in VALID]))
            pooled[consumer][arm] = dict(mse=mse, ratio_to_zero=mse/zero)
    ratio_to_reset = pooled["next_mean"]["history"]["mse"] / pooled["next_mean"]["reset"]["mse"]
    wins = sum(per_rank[str(rank)]["history_next_beats_reset"] for rank in VALID)
    gates = dict(
        pooled_covariance_ratio_le_050=pooled["covariance"]["history"]["ratio_to_zero"] <= .50,
        pooled_next_mean_ratio_le_050=pooled["next_mean"]["history"]["ratio_to_zero"] <= .50,
        pooled_next_ratio_to_reset_le_080=ratio_to_reset <= .80,
        next_wins_at_least_three_of_four=wins >= 3)
    return dict(passed=bool(all(gates.values())),
                verdict="PASS_FIXED_INFORMATION_TEST" if all(gates.values()) else "FAIL_FIXED_INFORMATION_TEST",
                gates=gates, pooled=pooled, per_rank=per_rank,
                pooled_history_next_ratio_to_reset=ratio_to_reset, next_wins=wins,
                fixed_thresholds=dict(covariance_to_zero=.50, next_mean_to_zero=.50,
                                      next_mean_to_reset=.80, wins=3),
                interpretation="Conditional offdiagonal bank-action capacity only; not raw accuracy or endogenous rollout")


def execute():
    check_process()
    # exist_ok=False and all subsequent writes use exclusive creation; nothing is overwritten.
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = {}
    progress = dict(features_frozen_ranks=[], action_targets_decoded_ranks=[],
                    coefficients_frozen=False, current_stage="provenance_preflight")
    try:
        require(not any(k == "numpy" or k.startswith("numpy.") for k in sys.modules),
                "Start fresh so the two-thread environment precedes NumPy initialization")
        thread_environment = {name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
        for name in thread_environment:
            os.environ[name] = "2"
        source_hashes, rows = prepare_provenance(manifest)
        import numpy as np
        from threadpoolctl import threadpool_info, threadpool_limits
        import pair_memory_action_physics_v2 as physics

        check_process()
        require(Path(physics.__file__).resolve() == HERE / PHYSICS_NAME,
                "Imported another physics implementation")
        write_json(OUTPUT / "started.json", dict(
            utc_started=utc_now(), interpreter=sys.executable, python_version=sys.version,
            numpy_version=np.__version__, train_ranks=list(TRAIN), valid_ranks=list(VALID),
            fixed_query="post12_to_pre13", source_hashes=source_hashes,
            judge_source_sha256=sha(Path(__file__)), capture_count=len(rows),
            capture_physics_version=CAPTURE_PHYSICS_NAME, judge_physics_version=PHYSICS_NAME,
            additive_response_note=RESPONSE_NOTE_NAME,
            action_target_access=False, mc_truth_accessed=False, private_truth_accessed=False,
            nice=os.getpriority(os.PRIO_PROCESS, 0), blas_threads_requested=2,
            previous_thread_environment=thread_environment,
            thread_environment={name: os.environ[name] for name in thread_environment},
            memory_limit_bytes=RSS_LIMIT, no_overwrite=True))
        write_json(OUTPUT / "provenance_preflight.json", dict(
            passed=True, files=manifest, count=len(manifest),
            all_capture_receipts_passed=True, all_capture_peak_rss_le_6GiB=True,
            capture_effective_blas_threads_independently_certified={
                str(rank): bool([entry for entry in rows[rank]["receipt"]["backend"]
                                 if entry.get("user_api") == "blas"]) for rank in RANKS},
            array_decoding_during_preflight=False, utc_finished=utc_now()))
        with threadpool_limits(limits=2, user_api="blas"):
            backend = threadpool_info()
            blas = [entry for entry in backend if entry.get("user_api") == "blas"]
            require(all(entry["num_threads"] == 2 for entry in blas),
                    "A reported judge BLAS backend violated the two-thread request")
            training = {}
            progress["current_stage"] = "train_sufficient_statistics"
            for rank in TRAIN:
                training[rank] = collect_one(np, physics, rank, rows[rank], OUTPUT, progress)
            progress["current_stage"] = "sole_train_fit"
            require(not set(progress["action_targets_decoded_ranks"]) & set(VALID),
                    "A held-out action was decoded before coefficient freezing")
            fit = fit_once(np, training)
            fit["train_statistics_sha256"] = {
                str(rank): sha(OUTPUT / f"rank_{rank:04d}.json") for rank in TRAIN}
            fit_path = OUTPUT / "coefficients_frozen.json"
            write_json(fit_path, fit)
            fit_sha = sha(fit_path)
            progress["coefficients_frozen"] = True
            progress["current_stage"] = "valid_sufficient_statistics"
            validation = {}
            for rank in VALID:
                require(sha(fit_path) == fit_sha, "Frozen coefficients changed before held-out action decoding")
                validation[rank] = collect_one(np, physics, rank, rows[rank], OUTPUT, progress)
            progress["current_stage"] = "single_heldout_decision"
            decision = judge_once(np, validation, fit)
            require(sha(fit_path) == fit_sha, "Frozen coefficients changed during held-out judgment")
            backend_end = threadpool_info()
        check_process()
        progress["current_stage"] = "provenance_postflight"
        recheck_manifest(manifest)
        require(progress["features_frozen_ranks"] == list(TRAIN) + list(VALID)
                and progress["action_targets_decoded_ranks"] == list(TRAIN) + list(VALID),
                "The fixed chronological rank inventory changed")
        output_hashes = {path.name: sha(path) for path in sorted(OUTPUT.glob("*.json"))}
        decision.update(dict(
            schema="nonlinear-pair-memory-information-judge-v1", qualified=True,
            source_hashes=source_hashes, judge_source_sha256=sha(Path(__file__)),
            input_manifest_sha256=sha(OUTPUT / "provenance_preflight.json"),
            all_frozen_inputs_and_sources_unchanged=True,
            coefficients_sha256=fit_sha, fit=fit, output_sha256=output_hashes,
            capture_peak_rss_bytes={str(rank): rows[rank]["receipt"]["peak_rss_bytes"] for rank in RANKS},
            chronology=progress, all_features_frozen_before_own_query_target=True,
            coefficients_frozen_before_all_valid_query_targets=True,
            action_targets_accessed=True, mc_truth_accessed=False, private_truth_accessed=False,
            torch_imported=False, backend_start=backend, backend_end=backend_end,
            blas_threads_requested=2,
            effective_blas_threads_independently_certified=bool(blas),
            thread_qualification="Environment request plus threadpool_limits; an empty backend list does not certify effective count",
            peak_rss_bytes=peak_rss(), wall_seconds=time.perf_counter()-started,
            utc_finished=utc_now(), no_adaptive_fit_or_feature_selection=True))
        write_json(OUTPUT / "result.json", decision)
        print(json.dumps(dict(verdict=decision["verdict"], gates=decision["gates"],
                              pooled=decision["pooled"], wins=decision["next_wins"],
                              wall_seconds=decision["wall_seconds"]), allow_nan=False), flush=True)
    except BaseException as exc:
        failure = dict(
            qualified=False, passed=False,
            verdict="INCONCLUSIVE_FIXED_INFORMATION_TEST" if isinstance(exc, Inconclusive)
                    else "FAILED_QUALIFICATION_OR_EXECUTION",
            exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
            chronology=progress, source_input_hashes_observed=manifest,
            action_targets_accessed=bool(progress["action_targets_decoded_ranks"]),
            mc_truth_accessed=False, private_truth_accessed=False,
            torch_imported=any(k == "torch" or k.startswith("torch.") for k in sys.modules),
            peak_rss_bytes=peak_rss(), wall_seconds=time.perf_counter()-started,
            utc_failed=utc_now(), partial_artifacts_preserved=True)
        write_json(OUTPUT / "failure.json", failure)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute-reviewed-judge", action="store_true",
                        help="Explicit root-reviewed execution of the one fixed source; no other modes fit")
    args = parser.parse_args()
    if not args.execute_reviewed_judge:
        print(json.dumps(dict(execute=False, train_ranks=list(TRAIN), valid_ranks=list(VALID),
                              query="post12_to_pre13", output=str(OUTPUT),
                              numerical_imports=False, input_or_target_array_reads=False,
                              fit_count=0, required_authorization="Root source review, then explicit flag")))
        return
    execute()


if __name__ == "__main__":
    main()
