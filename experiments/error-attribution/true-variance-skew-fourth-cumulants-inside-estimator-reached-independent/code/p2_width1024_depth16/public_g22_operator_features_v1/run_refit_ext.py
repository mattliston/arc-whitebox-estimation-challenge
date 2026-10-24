#!/usr/bin/env python3
"""TAG g22-refit-k4panel: refit the learned operator (F2J) on the
k4-RESCALED host, with the FROZEN 160/40 J-native transfer protocol
(public_deploy_composition_prep_v1/run_joint_refit.py spelling: same J
source by sha, 25,600 parameters, model_seed 1753362478, shuffle stream
40, six epochs, fixed final checkpoint, same valid-40 roster by blake2s of
mlp_id). Rows: rows_k4panel/ (gen_rows_k4panel.py: estimator_retro.py with
RETRO_FORCE=k4 + the frozen panel-mean table k4_scale_panel8.json). The
fit runs on MPS when possible (model constructed and seeded on CPU so the
protocol's initialization law is unchanged, then moved; load_batch
monkeypatched to move tensors; scoring runs on CPU because the protocol's
score() uses float64). Outputs (never overwritten): REFIT_K4PANEL_MODEL.pt,
REFIT_K4PANEL_RESULT.json. Then scores r35/r100 on the INDEPENDENT truths
(mps_truth_rank*_indep4M/_indep32M) in teammate A's frame:
deployed = clamp(host + corrector(tape), 0), noise-corrected final-layer MSE.
Usage: run_refit_k4panel.py [cpu]"""
from dataclasses import dataclass
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
J_SOURCE = HERE / "jext/run_experiment_ext.py"   # G22 extended-width copy of the pinned J source
J_SOURCE_SHA256 = None                              # recorded at run time (new operator source, not the pinned one)
LOCAL_FEATURES = int(os.environ.get("G22_LOCAL_FEATURES", "20"))
SMOKE_ZERO = int(os.environ.get("G22_SMOKE_ZERO_FEATURES", "0"))   # smoke tests only: append K zero features
TAG = os.environ.get("G22_TAG", "EXT")
ROWS = Path(os.environ.get("G22_ROWS_DIR", str(HERE / "rows_k4panel")))
HOST_SOURCE = Path(os.environ.get("G22_HOST_SOURCE", str(P2 / "public_g20_emission_retrofit_v1/estimator_retro.py")))
TABLE = Path(os.environ.get("G22_K4_TABLE", str(HERE / "k4_scale_panel8.json")))
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
TARGETS = P2 / "data_gate/sealed_role_target_bundle/observer_fit.parquet"
G19 = P2 / "public_g19_attribution_v1"
MODEL = HERE / f"REFIT_{TAG}_MODEL.pt"
RESULT = HERE / f"REFIT_{TAG}_RESULT.json"
ROSTER = tuple(range(35, 235))
EXPECTED_VALID_GLOBAL = (
    37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101, 122, 130,
    132, 135, 139, 145, 146, 149, 150, 151, 155, 159, 163, 169, 173, 175,
    181, 192, 198, 199, 204, 219, 224, 227, 233, 234,
)
MODEL_SEED = 1_753_362_478
SHUFFLE_STREAM_ID = 40
# frames from teammate A's tables (independent 4M draw): deployed FULL card
DEPLOYED_FULL_INDEP4M = {35: 2.7038e-8, 100: 2.7787e-8}
DEPLOYED_HOST_INDEP4M = {35: 8.622e-8, 100: 7.696e-8}


def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()


def load_module(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m


def nc_mse(pred_final, truth):
    r = truth["mc_post"][15].astype(np.float64) - pred_final.astype(np.float64)
    return (float(np.sum(r * r)) - float(np.sum(truth["mc_se"][15].astype(np.float64) ** 2))) / 1024.0


def memory_state():
    """(swap_used_fraction, free_fraction) from sysctl vm.swapusage and memory_pressure."""
    import re, subprocess
    sw = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"total = ([0-9.]+)M\s+used = ([0-9.]+)M", sw)
    swap_frac = float(m.group(2)) / max(float(m.group(1)), 1.0) if m else 0.0
    mp = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    f = re.search(r"free percentage: ([0-9.]+)%", mp)
    free_frac = float(f.group(1)) / 100.0 if f else 1.0
    return swap_frac, free_frac


def other_mps_jobs():
    """known MPS engines from other teammates (torch on Metal): count them."""
    import subprocess
    ps = subprocess.run(["ps", "-axo", "command"], capture_output=True, text=True).stdout
    pats = ("run_tensor_recursion", "run_chaos", "run_tgram", "mc_truth_mps", "run_g3_", "oracle.py")
    return sum(1 for line in ps.splitlines() if any(p in line for p in pats) and "python" in line)


def mps_gate(log):
    """Amended guardrail (coordinator, 2026-09-01 late): the swap figure on this box is stale
    (macOS never drains it), so gate the MPS fit on memory_pressure free >= 30% and on no other
    MPS job running. Wait up to 60 min for that; otherwise fall back to the protocol's CPU fit."""
    # 2026-09-01 22:40 finding: the frozen J source (sha-pinned) computes its feature normalization in
    # float64 and feeds Double tensors to its Linear layers; MPS has no float64, so the protocol-exact
    # fit cannot run on MPS without editing the pinned source. The fit is minutes on CPU (the deployed
    # payload's own device). MPS is therefore skipped by design; the memory numbers are still logged.
    swap_frac, free_frac = memory_state()
    log.append({"t_min": 0, "swap_used_frac_stale": round(swap_frac, 3), "free_frac": round(free_frac, 3),
                "other_mps_jobs": other_mps_jobs(), "mps_ok": False,
                "reason": "frozen J source is float64-bound (Double inputs to nn.Linear); MPS unsupported without editing the pinned file"})
    return False
    waited = 0
    while True:
        swap_frac, free_frac = memory_state()
        others = other_mps_jobs()
        ok = free_frac >= 0.30 and others == 0
        log.append({"t_min": waited, "swap_used_frac_stale": round(swap_frac, 3), "free_frac": round(free_frac, 3),
                    "other_mps_jobs": others, "mps_ok": ok})
        if ok:
            return True
        if waited >= 60:
            return False
        time.sleep(60); waited += 1


def main():
    want_cpu = len(sys.argv) > 1 and sys.argv[1] == "cpu"
    gate_log = []
    if not want_cpu:
        want_cpu = not mps_gate(gate_log)
        print(json.dumps({"mps_gate": gate_log[-1], "use_mps": not want_cpu}), flush=True)
    j_sha = sha(J_SOURCE)
    host_sha = sha(HOST_SOURCE)
    table_sha = sha(TABLE)
    if RESULT.exists() or MODEL.exists():
        raise FileExistsError("refit output exists; version, never overwrite")
    records = []
    for rank in ROSTER:
        d = ROWS / f"row_{rank:04d}"
        receipt = json.loads((d / "result.json").read_text())
        assert receipt["status"] == "complete" and int(receipt["rank"]) == rank
        assert int(receipt.get("targets_read", 0)) == 0
        if receipt.get("k4_table_sha256_prefix") not in ("n/a (literal in host)", table_sha[:16]):
            raise AssertionError((rank, receipt.get("k4_table_sha256_prefix")))
        if "host_sha256_prefix" in receipt:
            assert receipt["host_sha256_prefix"] == host_sha[:16], rank
        for name in ("host.npy", "raw20.npy", "sigma.npy"):
            assert (d / name).exists(), (rank, name)
        records.append(receipt)
    keyed = []
    for local, receipt in enumerate(records):
        digest = hashlib.blake2s(
            f"p2r0f2j-q24-native-v1:{int(receipt['mlp_id'])}".encode("ascii"),
            digest_size=16).digest()
        keyed.append((digest, local))
    valid = np.asarray(sorted(local for _, local in sorted(keyed)[:40]), dtype=np.int64)
    train = np.asarray(sorted(set(range(200)) - set(valid.tolist())), dtype=np.int64)
    assert tuple(ROSTER[int(l)] for l in valid) == EXPECTED_VALID_GLOBAL and len(train) == 160

    table = pq.read_table(TARGETS, columns=["mlp_id", "all_layer_means"])
    ids = table.column("mlp_id").combine_chunks().to_numpy(zero_copy_only=False)
    values = table.column("all_layer_means").combine_chunks()
    storage = getattr(values, "storage", values)
    while not pa.types.is_primitive(storage.type):
        storage = storage.flatten()
    targets = storage.to_numpy(zero_copy_only=False).reshape(688, 16, 1024)
    expected_ids = np.asarray([int(r["mlp_id"]) for r in records], dtype=np.int32)
    assert np.array_equal(ids[35:235], expected_ids)
    targets = np.ascontiguousarray(targets[35:235], dtype=np.float32)
    hosts = np.ascontiguousarray(np.stack([
        np.load(ROWS / f"row_{rank:04d}/host.npy", allow_pickle=False) for rank in ROSTER
    ]), dtype=np.float32)
    assert hosts.shape == (200, 16, 1024) and np.isfinite(hosts).all()
    error = np.ascontiguousarray(targets - hosts, dtype=np.float32)

    @dataclass(frozen=True)
    class JoinedRow:
        raw_locals: np.ndarray
        sigma: np.ndarray
        weights_path: Path

    class JointDataset:
        def __init__(self):
            self.mlp_ids = np.asarray([int(r["mlp_id"]) for r in records], dtype=np.int64)

        def load(self, local):
            rank = ROSTER[int(local)]
            d = ROWS / f"row_{rank:04d}"
            raw = np.load(d / "raw20.npy", allow_pickle=False)
            if (d / "mech.npy").exists():
                mech = np.load(d / "mech.npy", allow_pickle=False).astype(raw.dtype)
                if os.environ.get("G22_ZERO_MECH") == "1": mech = np.zeros_like(mech)   # control arm
                raw = np.concatenate([raw, mech], axis=2)
            if SMOKE_ZERO:
                raw = np.concatenate([raw, np.zeros(raw.shape[:2] + (SMOKE_ZERO,), raw.dtype)], axis=2)
            assert raw.shape == (16, 1024, LOCAL_FEATURES), (rank, raw.shape, LOCAL_FEATURES)
            return JoinedRow(
                raw_locals=raw,
                sigma=np.load(d / "sigma.npy", mmap_mode="r", allow_pickle=False),
                weights_path=STORE / f"rows/observer_fit/{rank:04d}/weights.npy",
            )

    _ds_tmp = JointDataset()
    def dataset_rows_raw(local):
        return _ds_tmp.load(local).raw_locals
    sums = np.zeros((200, 16, LOCAL_FEATURES - 1), dtype=np.float64)
    sums2 = np.zeros_like(sums)
    for local, rank in enumerate(ROSTER):
        raw = dataset_rows_raw(local)
        tail = np.asarray(raw[:, :, 1:], dtype=np.float64)
        sums[local] = tail.sum(axis=1, dtype=np.float64)
        sums2[local] = np.square(tail).sum(axis=1, dtype=np.float64)

    os.environ["G22_LOCAL_FEATURES"] = str(LOCAL_FEATURES)
    j = load_module("ext_refit_j", J_SOURCE)
    try:
        j.configure()
    except RuntimeError:
        pass
    dataset = JointDataset()
    centers, scales, layer_rms = j.moments_for(train, error, sums, sums2)
    torch.manual_seed(MODEL_SEED)
    model = j.TransportEnergyMultiStateResidual(centers, scales, layer_rms)   # CPU init = protocol law
    n_params = j.parameter_count(model)   # 25,600 at width 20; grows with the tape width

    device_used = "cpu"
    orig_load_batch = j.load_batch
    started = time.perf_counter()
    if not want_cpu and torch.backends.mps.is_available():
        try:
            dev = torch.device("mps")

            def load_batch_mps(dataset_, ranks_, error_):
                return tuple(t.to(dev) for t in orig_load_batch(dataset_, ranks_, error_))

            j.load_batch = load_batch_mps
            torch.set_default_device(dev)
            # MPS has no float64: the protocol registers centers/scales/layer_rms as float64 buffers ->
            # cast the model to float32 for the MPS fit (parameters are float32 already; deviation noted)
            model.float().to(dev)
            trace, _rss = j.fit_fold(SHUFFLE_STREAM_ID, model, dataset, train, error)
            device_used = "mps"
        except Exception as exc:  # fall back to the protocol's own CPU path IN A FRESH PROCESS
            # (a failed MPS attempt can leave torch._dynamo partially imported -> circular-import
            # crash if the CPU fit is retried in the same process; observed 2026-09-01 22:25)
            print(json.dumps({"mps_fallback": repr(exc)[:400], "action": "re-exec with cpu"}), flush=True)
            import subprocess
            torch.set_default_device("cpu"); j.load_batch = orig_load_batch
            r = subprocess.run([sys.executable, __file__, "cpu"], env=dict(os.environ, G22_MPS_FAILED=repr(exc)[:200]))
            raise SystemExit(r.returncode)
        finally:
            torch.set_default_device("cpu")
            j.load_batch = orig_load_batch
            model.to("cpu")
    else:
        trace, _rss = j.fit_fold(SHUFFLE_STREAM_ID, model, dataset, train, error)
    training_seconds = time.perf_counter() - started
    autonomous = j.score(model, dataset, valid, error)

    artifact = {
        "schema": "k4panel-refit-model-v1",
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "model_seed": MODEL_SEED, "shuffle_stream_id": SHUFFLE_STREAM_ID,
        "j_source_sha256": j_sha, "j_source": str(J_SOURCE), "local_features": LOCAL_FEATURES, "n_params": n_params, "smoke_zero_features": SMOKE_ZERO, "mech_zeroed_control": os.environ.get("G22_ZERO_MECH") == "1", "host_source_sha256": host_sha,
        "k4_table_sha256": table_sha, "device_used": device_used,
        "rows_source": str(ROWS),
    }
    tmp = MODEL.with_name(f".{MODEL.name}.{os.getpid()}.tmp")
    torch.save(artifact, tmp)
    tmp.replace(MODEL)

    # deployed-frame scoring vs the sealed targets (all-200 in-sample / valid-40)
    depth, width = 16, 1024
    sse_dep = np.zeros(depth); sse_host = np.zeros(depth)
    final_sq = np.zeros(200); clamp_hits = 0; finite = True
    corrections = {}
    model.eval()
    with torch.no_grad():
        for offset in range(0, 200, 2):
            group = np.arange(offset, min(offset + 2, 200), dtype=np.int64)
            raw, sigma, weights, _t = j.load_batch(dataset, group, error)
            pred = model(raw, sigma, weights, exposure=1.0).double().cpu().numpy()
            finite = finite and bool(np.isfinite(pred).all())
            hg = hosts[group].astype(np.float64); tg = targets[group].astype(np.float64)
            unclamped = hg + pred
            deployed = np.maximum(unclamped, 0.0)
            clamp_hits += int((unclamped < 0.0).sum())
            sse_dep += np.square(deployed - tg).sum(axis=(0, 2))
            sse_host += np.square(hg - tg).sum(axis=(0, 2))
            final_sq[group] = np.square(deployed[:, -1] - tg[:, -1]).mean(axis=1)
            for gi, local in enumerate(group):
                if ROSTER[int(local)] in (35, 100):
                    corrections[ROSTER[int(local)]] = pred[gi]
    all200_layer = sse_dep / (200 * width); host_layer = sse_host / (200 * width)
    host_valid40_final = float(np.square(error[valid].astype(np.float64))[:, -1].mean())

    # independent-truth scoring for r35 / r100 (teammate A's frame)
    indep = {}
    for rank in (35, 100):
        host = np.load(ROWS / f"row_{rank:04d}/host.npy").astype(np.float64)
        dep_final = np.maximum(host[15] + corrections[rank][15], 0.0)
        row = {}
        for frame in ("indep4M", "indep32M"):
            tr = np.load(G19 / f"mps_truth_rank{rank:04d}_{frame}.npz")
            row[frame] = {"refit_f2j_on": nc_mse(dep_final, tr), "host_f2j_off": nc_mse(host[15], tr)}
        row["deployed_full_card_indep4M_ref"] = DEPLOYED_FULL_INDEP4M[rank]
        row["deployed_host_indep4M_ref"] = DEPLOYED_HOST_INDEP4M[rank]
        row["cut_vs_deployed_full_indep4M_pct"] = round(100 * (1 - row["indep4M"]["refit_f2j_on"] / DEPLOYED_FULL_INDEP4M[rank]), 1)
        row["in_training_split"] = "valid40" if int(np.where(np.asarray(ROSTER) == rank)[0][0]) in set(valid.tolist()) else "train160"
        indep[str(rank)] = row

    result = {
        "schema": "k4panel-refit-deployed-v1",
        "protocol": {"law": "frozen 160/40 J-native transfer protocol", "model_seed": MODEL_SEED,
                     "shuffle_stream_id": SHUFFLE_STREAM_ID, "epochs": 6, "parameters": n_params,
                     "j_source_sha256": j_sha, "j_source": "jext/run_experiment_ext.py (extended-width copy of the pinned J)", "local_features": LOCAL_FEATURES, "n_params": n_params, "epochs_override": os.environ.get("G22_EPOCHS_OVERRIDE_SMOKE_ONLY"), "rows": f"{ROWS.name} ranks 35..234",
                     "host_source_sha256": host_sha, "k4_table_sha256": table_sha, "device_used": device_used, "memory_gate_log": gate_log, "mps_failure_note": os.environ.get("G22_MPS_FAILED")},
        "training": {"valid40_corrected_final_mse_protocol_unclamped": float(autonomous["final_mse"]),
                     "valid40_host_final_mse": host_valid40_final,
                     "j_removal_fraction_valid40": 1.0 - float(autonomous["final_mse"]) / host_valid40_final,
                     "finite_full_rollout": bool(autonomous["finite_full_rollout"]),
                     "training_trace_final": trace[-1], "training_seconds": training_seconds,
                     "model_sha256": sha(MODEL)},
        "deployed_frame_sealed_targets": {
            "valid40_final_mse": float(final_sq[valid].mean()),
            "train160_final_mse_in_sample": float(final_sq[train].mean()),
            "all200_final_mse_IN_SAMPLE": float(all200_layer[-1]),
            "all200_host_final_mse": float(host_layer[-1]),
            "all200_deployed_layer_mse": all200_layer.tolist(),
            "clamp_engaged_elements_of_3276800": clamp_hits, "finite": finite},
        "independent_truth_scoring": indep,
        "reference_prior_refit_floor": {"joint_refit_valid40": 3.133856737168449e-08,
                                        "base_refit_valid40": 3.133419038664782e-08,
                                        "deployed_baseline_all200_in_sample": 2.958e-8},
    }
    tmp2 = RESULT.with_name(f".{RESULT.name}.{os.getpid()}.tmp")
    tmp2.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    tmp2.replace(RESULT)
    print(json.dumps({"device": device_used,
                      "valid40_protocol": result["training"]["valid40_corrected_final_mse_protocol_unclamped"],
                      "valid40_host": host_valid40_final,
                      "valid40_deployed": result["deployed_frame_sealed_targets"]["valid40_final_mse"],
                      "all200_deployed_IN_SAMPLE": result["deployed_frame_sealed_targets"]["all200_final_mse_IN_SAMPLE"],
                      "indep": indep}, indent=1), flush=True)


if __name__ == "__main__":
    main()
