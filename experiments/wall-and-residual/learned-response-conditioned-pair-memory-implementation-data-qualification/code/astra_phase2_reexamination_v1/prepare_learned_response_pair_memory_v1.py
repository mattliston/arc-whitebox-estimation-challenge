"""Inert preparation of the fixed causal20/4 pair-memory dataset.

No bank rerun, fit, private data, or Monte Carlo. Reuses authenticated v2
captures. Each causal input file is sealed before the teacher is decoded.
The predictor's eight-key dictionary is physically separate from loss data.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
OUT = HERE / "learned_response_pair_memory_data_v1"
SYNTHETIC = HERE / "learned_response_pair_memory_synthetic_certificate_v1.json"
WIDE = HERE / "learned_response_pair_memory_wide_certificate_v1.json"
PINS = {
    "judge_nonlinear_pair_memory_information_v1.py": "0485c62b96a16aa712f2ca8f262af77029475a2d9c9e1efe394643f8f5fa9ac9",
    "pair_memory_action_physics_v2.py": "649d1523b2a0c764373b4a779231e74a2397368649d7e1f81c042401b31e6333",
    "LEARNED_RESPONSE_PAIR_MEMORY_PROTOCOL_DRAFT_V1.md": "1e277cd16c6596cca947172d0dcd2bd7c21c8962256596ca8ea070c65fc8754c",
    "LEARNED_RESPONSE_PAIR_MEMORY_ROOT_ACCEPTANCE_V1.md": "45c590e373fb35c6af0c9a248addf8b98d5e6830c8b6b66861fd5ced3b537e2a",
    "learned_response_pair_memory_numpy_v1.py": "5a24a77c24f08ba521fa8d0cd36a569a9eeee6bf3045cb4ee15d0117fbf4129e",
    "learned_response_pair_memory_torch_v1.py": "54e4040947cfd818246b1cc604fd79516e426771ca23b1568eb2391b394c2789",
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, obj):
    with path.open("x") as stream:
        json.dump(obj, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execute(remaining):
    require("/anaconda3/" not in sys.executable, "Use the NumPy-only repo interpreter")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch nice>=10")
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    source_hashes = {name: sha(HERE/name) for name in PINS}
    require(source_hashes == PINS, "Frozen prerequisite source mismatch")
    source_hashes[Path(__file__).name] = sha(Path(__file__))
    synthetic = json.loads(SYNTHETIC.read_text())
    require(synthetic["passed"], "Synthetic implementation certificate has not passed")
    source_hashes[SYNTHETIC.name] = sha(SYNTHETIC)
    for name, digest in synthetic["source_hashes"].items():
        require(sha(HERE/name) == digest, f"Synthetic source changed: {name}")
        source_hashes[name] = digest
    if remaining:
        wide = json.loads(WIDE.read_text())
        require(wide["passed"] and wide["rank"] == 35, "Rank35 wide certificate has not passed")
        source_hashes[WIDE.name] = sha(WIDE)
        for name, digest in wide["source_hashes"].items():
            require(sha(HERE/name) == digest, f"Wide source changed: {name}")
            source_hashes[name] = digest
        ranks = tuple(range(36, 59))
    else:
        ranks = (35,)
    for rank in ranks:
        require(not (OUT/f"row_{rank:04d}").exists(), f"Existing row{rank}; no overwrite/replay")

    import numpy as np
    from scipy.special import ndtr
    from threadpoolctl import threadpool_limits, threadpool_info
    import judge_nonlinear_pair_memory_information_v1 as old
    import pair_memory_action_physics_v2 as physics
    from learned_response_pair_memory_numpy_v1 import _psi
    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules), "Unexpected Torch")
    manifest = {}
    _, rows = old.prepare_provenance(manifest)
    OUT.mkdir(exist_ok=True)
    run_started = time.perf_counter()
    run_rows = []

    def finite(value, label):
        require(np.isfinite(value).all(), f"Nonfinite {label}")

    def save_npz(path, arrays):
        for key, value in arrays.items():
            finite(value, key)
        with path.open("xb") as stream:
            np.savez(stream, **arrays)

    with threadpool_limits(limits=2):
        for rank in ranks:
            out = OUT/f"row_{rank:04d}"
            out.mkdir(exist_ok=False)
            row_started = time.perf_counter()
            record = dict(passed=False, rank=rank, source_hashes=source_hashes,
                old_receipt_sha256=sha(rows[rank]["directory"]/"result.json"),
                original_input_manifest=manifest, teacher_decoded=False,
                started_utc=datetime.now(timezone.utc).isoformat())
            write_json(out/"started.json", record)
            try:
                row_dir = rows[rank]["directory"]
                weights = np.load(rows[rank]["weights"], mmap_mode="r", allow_pickle=False)
                with np.load(row_dir/"pre_vectors_00_14.npz", allow_pickle=False) as z:
                    pre_sigma, pre_alpha = z["sigma"], z["alpha"]
                    pre_variance, pre_alive = z["variance"], z["alive"]
                require(np.array_equal(pre_sigma, np.sqrt(np.maximum(pre_variance, 1e-10))),
                        "Pre-unit ownership changed")
                V = np.zeros((13,1024,1024), np.float32)
                bs = np.empty_like(V)
                ds = np.empty((13,1024), np.float32)
                gs = np.empty_like(ds)
                ts = np.empty_like(ds)
                previous_sigma = None
                layers = []
                for layer in range(13):
                    with np.load(row_dir/f"post_{layer:02d}.npz", allow_pickle=False) as z:
                        variance, physical, alive = z["variance"], z["physical_K21"], z["alive"]
                    s = np.sqrt(np.maximum(variance, 1e-10))
                    D = physical/(s[:,None]**2*s[None,:])
                    diagonal = np.diag(D).copy()
                    maxd = float(np.max(np.abs(diagonal)))
                    require(maxd <= math_sqrt_f32_max(np), "Skew pair product would overflowFP32")
                    source = _psi(physics.memory_source(D))
                    g = ndtr(pre_alpha[layer])
                    t = g*alive*pre_sigma[layer]/s
                    if layer:
                        V[layer] = previous_sigma[:,None]*weights[layer].astype(np.float64)/pre_sigma[layer][None,:]
                    bs[layer], ds[layer], gs[layer], ts[layer] = source, diagonal, g, t
                    with np.errstate(over="ignore", invalid="ignore"):
                        finite(ds[layer,:,None]*ds[layer,None,:], "actualFP32 skew pair products")
                    layers.append(dict(layer=layer, variance_min=float(variance.min()),
                        negative_variance_count=int((variance<0).sum()),
                        variance_unit_floor_count=int((variance<1e-10).sum()),
                        active_count=int(alive.sum()), maximum_absolute_standardized_skew=maxd))
                    previous_sigma = s
                query_V64 = previous_sigma[:,None]*weights[13].astype(np.float64)/pre_sigma[13][None,:]
                K64 = physics.star_action(D, query_V64)
                mask = pre_alive[13,:,None] & pre_alive[13,None,:]
                np.fill_diagonal(mask, False)
                inputs = dict(V=V, b=bs, d=ds, g=gs, post_scale=ts,
                    query_V=query_V64.astype(np.float32), K=K64.astype(np.float32), mask=mask)
                save_npz(out/"inputs.npz", inputs)
                inputs_sha = sha(out/"inputs.npz")
                write_json(out/"causal_seal.json", dict(rank=rank,
                    inputs_sha256=inputs_sha, teacher_decoded=False,
                    sealed_utc=datetime.now(timezone.utc).isoformat(),
                    predictor_keys=sorted(inputs), source_hashes=source_hashes))
                # All predictor arrays now sealed. Remaining fields are LOSS ONLY.
                with np.load(row_dir/"pre_13.npz", allow_pickle=False) as z:
                    covariance, sigma = z["pre_C"], z["aux_sigma"]
                    alpha, pdf, gate = z["aux_alpha"], z["aux_pdf"], z["aux_gate"]
                    require(np.array_equal(sigma, pre_sigma[13]) and
                            np.array_equal(np.diag(covariance), pre_variance[13]) and
                            np.array_equal(alpha, pre_alpha[13]), "Loss boundary ownership changed")
                    record["teacher_decoded"] = True
                    teacher = z["pre_K21_loss_only"]
                require(np.all(np.diag(covariance)>0), "Invalid original variance; no repair")
                residual = mask*(teacher/(sigma[:,None]**2*sigma[None,:])-K64)
                cov_target = physics.full_cubic_covariance_response(residual,covariance,sigma,alpha,pdf,gate)
                next_w = weights[14].astype(np.float64)
                next_pdf = np.exp(-pre_alpha[14]**2/2)/np.sqrt(2*np.pi)
                next_target = physics.next_gaussian_mean_response(cov_target,next_w,pre_sigma[14],next_pdf)
                rho = covariance/np.outer(sigma,sigma)
                np.fill_diagonal(rho,0.)
                a1,a2,a3,a4 = gate,pdf,-alpha*pdf,(alpha*alpha-1)*pdf
                forward = np.outer(a2,a1)+rho*np.outer(a3,a2)+.5*rho*rho*np.outer(a4,a3)
                reverse = np.outer(a1,a2)+rho*np.outer(a2,a3)+.5*rho*rho*np.outer(a3,a4)
                ratio = sigma*sigma/np.diag(covariance)
                cf = .5*np.outer(sigma,sigma)*forward*ratio[:,None]
                cr = .5*np.outer(sigma,sigma)*reverse*ratio[None,:]
                np.fill_diagonal(cf,0.)
                np.fill_diagonal(cr,0.)
                check = cf*residual+cr*residual.T
                relative = float(np.linalg.norm(check-cov_target)/max(np.linalg.norm(cov_target),1e-300))
                require(relative <= 2e-11, "Loss operator disagrees with certified directed v2")
                losses = dict(covariance_forward=cf.astype(np.float32),
                    covariance_reverse=cr.astype(np.float32), next_weight=next_w.astype(np.float32),
                    next_response=(next_pdf/(2*pre_sigma[14])).astype(np.float32),
                    covariance_target=cov_target.astype(np.float32), next_mean_target=next_target.astype(np.float32))
                save_npz(out/"loss.npz", losses)
                old.check_process()
                require(sha(out/"inputs.npz") == inputs_sha, "Causal input changed after teacher decode")
                record.update(passed=True, layers=layers, predictor_keys=sorted(inputs),
                    loss_only_keys=sorted(losses), directed_operator_relative_error=relative,
                    zero_covariance_energy=float(np.mean(cov_target*cov_target)),
                    zero_next_mean_energy=float(np.mean(next_target*next_target)),
                    backend=threadpool_info(), bank_replayed=False, mc_or_private_truth_accessed=False)
            except BaseException as exc:
                record["passed"] = False
                record.update(exception_type=type(exc).__name__, exception=str(exc))
            record.update(wall_s=time.perf_counter()-row_started,
                peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                artifacts={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(out.glob("*.npz"))},
                finished_utc=datetime.now(timezone.utc).isoformat())
            record["passed"] = bool(record["passed"] and record["peak_rss_bytes"] <= 6*(1<<30))
            write_json(out/"result.json", record)
            print(json.dumps({k:record[k] for k in ("passed","rank","wall_s","peak_rss_bytes")}),flush=True)
            run_rows.append(dict(rank=rank, passed=record["passed"], result_sha256=sha(out/"result.json")))
            require(record["passed"], f"Preparation failed at{rank}; preserve partial files")
    old.recheck_manifest(manifest)
    require({name:sha(HERE/name) for name in source_hashes} == source_hashes,
            "A source or numerical prerequisite changed")
    write_json(OUT/("remaining_receipt.json" if remaining else "rank35_receipt.json"),
        dict(passed=True, rows=run_rows, source_hashes=source_hashes,
             wall_s=time.perf_counter()-run_started, hashes_unchanged=True))


def math_sqrt_f32_max(np):
    return float(np.sqrt(float(np.finfo(np.float32).max)))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--remaining", action="store_true")
    args=parser.parse_args()
    if args.execute:
        execute(args.remaining)
    else:
        print("INERT: authenticated existingdata; rank35first, remaining23onlyafterwidequalifier.")
