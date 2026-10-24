"""Additive, target-free provenance audit of the already-running joint fit.

Never imports an estimator, Torch, MPS, or a target array. Weight files are
stream-hashed in 2 MiB chunks. Only this new audit directory is written.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np


HERE = Path(__file__).resolve().parent
FIT = HERE / "joint_prefix_final_fit_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = HERE / "joint_prefix_final_provenance_v1"
CHUNK_BYTES = 2 * 1024 * 1024
MEMORY_LIMIT = 2_000_000_000
ENTRYPOINTS = {
    "numpy_inference": "joint_prefix_final_numpy_v1.py",
    "torch_card_loader": "joint_prefix_final_data_v1.py",
    "fixed_trainer": "train_joint_prefix_final_v1.py",
}
# These three source-text edges are not Python imports. They were separately
# inspected in the read-only audit; inspect.getsource edges are already
# covered by the imported module containing the relevant function.
SOURCE_TEXT_EDGES = {
    "current_transport_shadow_v3.py": ("current_transport_shadow_v1.py",),
    "centered_torch_numerics_v3.py": ("centered_torch_numerics_v1.py",),
    "centered_bank_autodiff_v3.py": ("current_transport_shadow_v1.py",),
}
EXPECTED_LOCAL_CLOSURE = {
    "centered_bank_autodiff_v1.py", "centered_bank_autodiff_v3.py",
    "centered_torch_numerics_v1.py", "centered_torch_numerics_v3.py",
    "compact_final_head_v1.py", "compact_fullpair_final_reference_v1.py",
    "compact_scalar_final_v1.py", "current_transport_shadow_v1.py",
    "current_transport_shadow_v3.py", "full_carrier_autodiff_v1.py",
    "full_carrier_bank_autodiff_v1.py", "gaussian_cdf_quadrature_v1.py",
    "joint_prefix_final_data_v1.py", "joint_prefix_final_numpy_v1.py",
    "k3_radial_numpy.py", "radial_binary_compact.py",
    "radial_compact_feed_fullpair_v1.py", "radial_compact_radialfeed_v1.py",
    "radial_cubic.py", "radial_features.py", "radial_gnn_numpy.py",
    "radial_mehler.py", "tensor_clamp_min_v1.py", "train_joint_prefix_final_v1.py",
}
PKEYS = ("msg_w", "msg_b", "enc0_w", "enc0_b", "enc2_w", "enc2_b",
         "enc4_w", "enc4_b", "head", "head_b", "gain")
QKEYS = ("w0", "b0", "w2", "b2", "w4", "b4")


def utc():
    return datetime.now(timezone.utc).isoformat()


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def hash_file(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_BYTES):
            digest.update(chunk)
            length += len(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (
        after.st_size, after.st_mtime_ns, after.st_ino
    ), f"File changed during hash: {path}"
    assert length == before.st_size
    assert peak_rss() < MEMORY_LIMIT
    return {
        "sha256": digest.hexdigest(), "bytes": length,
        "mtime_ns": before.st_mtime_ns,
        "mtime_utc": datetime.fromtimestamp(before.st_mtime, timezone.utc).isoformat(),
    }


def dependencies(entry):
    pending, seen, edges, external = [entry], set(), set(), set()
    while pending:
        filename = pending.pop()
        if filename in seen:
            continue
        seen.add(filename)
        source = (HERE / filename).read_text()
        tree = ast.parse(source, filename=filename)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, (filename, "unexpected relative import")
                names = [node.module] if node.module else []
            else:
                names = []
            for module in names:
                target = module.replace(".", "/") + ".py"
                if (HERE / target).is_file():
                    pending.append(target)
                    edges.add((filename, target, "import"))
                else:
                    external.add(module)
        for target in SOURCE_TEXT_EDGES.get(filename, ()):
            assert target in source, (filename, target)
            pending.append(target)
            edges.add((filename, target, "explicit_source_text"))
    return {
        "entrypoint": entry, "local_files": sorted(seen),
        "local_edges": [dict(source=a, target=b, kind=c) for a, b, c in sorted(edges)],
        "external_module_names_not_imported_by_auditor": sorted(external),
    }


def literal_assignments(path):
    result = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("PKEYS", "QKEYS", "VALID"):
                    result[target.id] = ast.literal_eval(node.value)
    return result


def parameter_audit(meta):
    pfile, qfile = HERE / "payload_compact_feed_fullpair_v1.npz", HERE / "payload_fullpair_final_head_v1.npz"
    with np.load(pfile, allow_pickle=False) as src:
        p = {key: src[key].copy() for key in PKEYS}
    with np.load(qfile, allow_pickle=False) as src:
        q = {key: src[key].copy() for key in QKEYS}
    values = {**{"P_" + k: v for k, v in p.items()}, **{"Q_" + k: v for k, v in q.items()}}
    allocated = sum(value.size for value in values.values())
    assert allocated == meta["parameter_count"] == 78097
    gradient_json = HERE / "joint_prefix_final_gradients_v1.json"
    gradient_npz = HERE / "joint_prefix_final_gradients_v1.npz"
    certificate = json.loads(gradient_json.read_text())
    assert certificate["passed"] and not certificate["target_access"]
    assert hash_file(gradient_npz)["sha256"] == certificate["output_sha256"]
    expected_zero = {key: 0 for key in values}
    expected_zero.update(P_enc0_w=256, P_head=128, P_head_b=2, P_gain=2, Q_w0=1664)
    records = []
    with np.load(gradient_npz, allow_pickle=False) as src:
        for rank in (35, 100):
            for device in ("cpu", "mps"):
                actual_zero, group_norms = {}, {}
                for key, value in values.items():
                    grad = src[f"gradient_{device}_{rank}_{key}"]
                    assert grad.shape == value.shape and np.isfinite(grad).all()
                    actual_zero[key] = int(np.count_nonzero(grad == 0))
                    group_norms[key] = float(np.linalg.norm(grad))
                    assert group_norms[key] > 0, (rank, device, key, "disconnected group")
                assert actual_zero == expected_zero, (rank, device, actual_zero)
                records.append(dict(rank=rank, device=device, zero_entries_by_group=actual_zero,
                                    nonzero_entries=allocated - sum(actual_zero.values()),
                                    all_groups_have_nonzero_vjp=True))
    return {
        "allocated_parameter_count": allocated,
        "structural_inactive_count": sum(expected_zero.values()),
        "active_upper_bound_and_observed_nonzero_point_vjp_count": allocated - sum(expected_zero.values()),
        "structural_inactive_entries": [
            dict(group="P_head/P_head_b/P_gain", rows=[0, 15], count=132,
                 reason="Prefix GNN skips layer 0; actual Q owns layer 15."),
            dict(group="P_enc0_w", columns=[1, 2], count=256,
                 reason="Frozen centering makes these constant input columns zero."),
            dict(group="Q_w0", columns=[0, 1, 2, 13, 14, 15, 22, 23, 24, 25, 26, 27, 28], count=1664,
                 reason="Three centered constant columns and ten explicitly masked columns."),
        ],
        "point_certificate_checks": records,
        "gradient_receipt_sha256": hash_file(gradient_json)["sha256"],
        "gradient_arrays_sha256": certificate["output_sha256"],
        "qualification": "Every intended P/Q tensor has a live gradient, but 78,097 is allocated count. Point VJPs do not establish global smoothness or accuracy gain.",
    }


def main():
    assert "torch" not in sys.modules
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert not OUT.exists(), "Preserve every existing audit; use a new version."
    OUT.mkdir()
    started = time.time()
    result = dict(passed=False, completed=False, utc_started=utc(), target_arrays_opened=0,
                  torch_imported=False, estimator_imported=False, weight_arrays_decoded=0,
                  stream_chunk_bytes=CHUNK_BYTES, weight_records=[])

    def record(item):
        item = {"utc": utc(), "wall_seconds": time.time() - started, **item}
        with (OUT / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(item) + "\n")
        print(json.dumps(item), flush=True)

    try:
        meta_path, store_path = FIT / "manifest.json", STORE / "manifest.json"
        meta_hash = hash_file(meta_path)
        meta = json.loads(meta_path.read_text())
        assert not meta["validation_targets_accessed"]
        fit_time = datetime.fromisoformat(meta["utc_started"])
        fit_start_ns = int(fit_time.timestamp() * 1_000_000_000)
        store_hash = hash_file(store_path)
        assert store_hash["sha256"] == meta["manifest_sha256"], "Store manifest differs from fit's authenticated manifest"
        manifest = json.loads(store_path.read_text())
        train, valid = meta["train_ranks"], meta["validation_ranks"]
        assert len(train) == 160 and len(valid) == 40 and not set(train) & set(valid)
        assert sorted(train + valid) == list(range(35, 235))
        literals = literal_assignments(HERE / "joint_prefix_final_data_v1.py")
        assert literals["PKEYS"] == PKEYS and literals["QKEYS"] == QKEYS
        assert tuple(valid) == literals["VALID"]
        result.update(fit_start_utc=meta["utc_started"], fit_manifest=meta_hash, store_manifest=store_hash,
                      train_count=len(train), validation_count=len(valid), train_ranks=train, validation_ranks=valid)
        closures = {name: dependencies(entry) for name, entry in ENTRYPOINTS.items()}
        files = set().union(*(set(value["local_files"]) for value in closures.values()))
        assert files == EXPECTED_LOCAL_CLOSURE, ("unexpected local dependency closure", sorted(files ^ EXPECTED_LOCAL_CLOSURE))
        hashes = {}
        for name in sorted(files):
            value = hash_file(HERE / name)
            value["precedes_fit_start"] = value["mtime_ns"] < fit_start_ns
            assert value["precedes_fit_start"], ("dependency newer than fit start", name, value)
            hashes[name] = value
        assert hashes["train_joint_prefix_final_v1.py"]["sha256"] == meta["script_sha256"]
        assert hashes["joint_prefix_final_data_v1.py"]["sha256"] == meta["loader_sha256"]
        protocol = hash_file(HERE / "JOINT_PREFIX_FINAL_FIT_PROTOCOL_V1.md")
        assert protocol["sha256"] == meta["protocol_sha256"]
        payloads = {}
        for ns, filename in (("P", "payload_compact_feed_fullpair_v1.npz"), ("Q", "payload_fullpair_final_head_v1.npz")):
            value = hash_file(HERE / filename)
            assert value["sha256"] == meta[f"original_{ns}_sha256"]
            assert value["mtime_ns"] < fit_start_ns
            payloads[ns] = dict(filename=filename, **value)
        result.update(local_dependency_graphs=closures, local_dependency_hashes=hashes,
                      local_dependencies_all_precede_fit_start=True, original_payloads=payloads,
                      protocol=protocol, parameter_accounting=parameter_audit(meta))
        record(dict(stage="dependencies_and_parameters_verified", local_files=len(files), allocated_parameters=78097,
                    structurally_inactive_parameters=2052, point_vjp_nonzero_parameters=76045))
        items = manifest["roles"]["observer_fit"]["row_manifests"]
        for rank in sorted(train + valid):
            matching = [item for item in items if item["role_rank"] == rank]
            assert len(matching) == 1, (rank, "non-unique row")
            item = matching[0]
            row_path = STORE / item["path"]
            assert row_path.resolve().is_relative_to(STORE.resolve())
            row_hash = hash_file(row_path)
            assert row_hash["sha256"] == item["sha256"], (rank, "row manifest digest mismatch")
            row = json.loads(row_path.read_text())
            assert row["role"] == "observer_fit" and row["role_rank"] == rank
            assert row["mlp_id"] == item["mlp_id"]
            declared = row["files"]["weights.npy"]
            assert declared["shape"] == [16, 1024, 1024] and declared["dtype"] == "<f4"
            weight_path = row_path.parent / "weights.npy"
            assert weight_path.resolve().is_relative_to(STORE.resolve())
            actual = hash_file(weight_path)
            assert actual["sha256"] == declared["sha256"], (rank, "WEIGHT DIGEST MISMATCH", declared["sha256"], actual["sha256"])
            assert actual["bytes"] == declared["bytes"], (rank, "weight size mismatch")
            assert actual["mtime_ns"] < fit_start_ns, (rank, "weight file newer than fit start")
            result["weight_records"].append(dict(
                rank=rank, split="train" if rank in train else "validation", mlp_id=row["mlp_id"],
                row_manifest=str(row_path), row_manifest_sha256=row_hash["sha256"],
                weight_file=str(weight_path), declared_sha256=declared["sha256"],
                declared_array_shape=declared["shape"], declared_array_dtype=declared["dtype"],
                authenticated_digest_matched=True, precedes_fit_start=True, **actual))
            if len(result["weight_records"]) % 20 == 0:
                record(dict(stage="weights_verified", verified=len(result["weight_records"]), latest_rank=rank,
                            bytes_streamed=sum(item["bytes"] for item in result["weight_records"]),
                            peak_process_rss_bytes=peak_rss()))
        assert len(result["weight_records"]) == 200
        for name, old in hashes.items():
            assert hash_file(HERE / name) == {key: value for key, value in old.items() if key != "precedes_fit_start"}, (name, "source changed during audit")
        assert hash_file(meta_path) == meta_hash
        assert hash_file(store_path) == store_hash
        assert "torch" not in sys.modules and peak_rss() < MEMORY_LIMIT
        result.update(passed=True, completed=True, weight_digest_matches=200, weight_digest_mismatches=0,
                      total_weight_bytes_streamed=sum(item["bytes"] for item in result["weight_records"]),
                      local_dependencies_unchanged_during_audit=True,
                      qualification="Additive release-hardening audit only. Current full-file weight digests match authenticated pre-fit row manifests. Current local source hashes and pre-fit mtimes are recorded, but transitive hashes were not pinned at fit launch; mtimes alone are not cryptographic proof of historical bytes. External package binaries and open-process code are outside this audit. No accuracy, metering, wall-time, submission, or pristine-holdout claim.")
    except BaseException as exc:
        result.update(exception_type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        record(dict(stage="MISMATCH_OR_AUDIT_FAILURE", exception_type=type(exc).__name__, message=str(exc),
                    verified_weights=len(result["weight_records"])))
        raise
    finally:
        result.update(utc_finished=utc(), wall_seconds=time.time() - started,
                      peak_process_rss_bytes=peak_rss(), verifier_sha256=hash_file(__file__)["sha256"],
                      verifier_python=sys.version, verifier_numpy=np.__version__)
        (OUT / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
        record({key: result[key] for key in ("passed", "completed", "wall_seconds", "peak_process_rss_bytes")})


if __name__ == "__main__":
    main()
