"""Fixed target-free rejection-only screen. Inert without --run.

No training, scoring, candidate execution, full-estimator admission or retries.
All eight groups complete before the sole prospective statistical decision.
"""

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = HERE / "paired_mehler_probe_rejection_v1"
N, WIDTH, DEPTH, GROUPS = 32, 1024, 16, 4
RANKS = (35, 100)
RSS_LIMIT = 2 * 1024**3
WALL_LIMIT = 180.0
C_FLOOR = 32 * WIDTH * (2 * WIDTH - 1)
T = 1e-9 * 2**41 / C_FLOOR
SOURCES = {
    "PAIRED_MEHLER_PROBE_REJECTION_PROTOCOL_V1.md": "2c557fbd3607193b8636dd052de056f4c9395aa6151a96c76068fd8148167375",
    "paired_mehler_probe_core_v1.py": "507293c3efab1c299d6c43628709ef9aad51ddfbae64c101e9efcbe77a186b49",
    "qualify_paired_mehler_probe_v1.py": "d2040f4a786f0d49310511ce3480a0cc11a71ef7e84fd8b77f48efeea7b57bb8",
    "paired_mehler_probe_tiny_v1.json": "5e69f79244f5afb8ea9abba357b365ad189737c02d5c1afbd370382f39f88c97",
}
INPUTS = {
    35: {
        "mu.npy": "c805845cea2a181411234fdb87485b209042c93ec13940f7f0c8a8ca03b67aa0",
        "sigma.npy": "0b3a141b31a7f3a1e66171986b60d6a789ad865c01ab63d9821505fb8f851ca9",
        "weights.npy": "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2",
    },
    100: {
        "mu.npy": "1925384d0d6687fe6500ad736ae8c17ee7b4dd9e34ae07fb21d52a18e7de97c9",
        "sigma.npy": "a93398e87aaa620ec8371942436e95fe15147ece43bc902141a839e5a6cebaa6",
        "weights.npy": "c4923171c8ddef28396aabd70335dad29141361efc13865ccb660a4a57421ee5",
    },
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def binding(path):
    path = Path(path).resolve()
    before = path.stat()
    value = digest(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"File changed while hashing: {path}")
    return dict(path=str(path), sha256=value, bytes=after.st_size,
                mtime_ns=after.st_mtime_ns)


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def main():
    if sys.argv[1:] != ["--run"]:
        print("Inert. Fixed input-only screen requires --run; no other options.")
        return
    require(not OUT.exists(), "Refusing existing diagnostic receipt directory")
    OUT.mkdir()
    began = time.monotonic()
    bindings, results = {}, []

    def guard(layer=None):
        require(time.monotonic() - began <= WALL_LIMIT, "Diagnostic wall guard")
        require(rss_bytes() <= RSS_LIMIT, "Diagnostic RSS guard")

    try:
        require(Path(sys.prefix).resolve() == Path("/Users/matt/anaconda3"),
                "Only the reviewed Anaconda FP64 diagnostic environment")
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "nice >=10 required")
        for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            require(os.environ.get(name) == "1", f"Explicit one-thread environment: {name}")
        require(os.environ.get("PYTHONDONTWRITEBYTECODE") == "1", "No bytecode writes")
        for name, expected in SOURCES.items():
            bindings[name] = binding(HERE / name)
            require(bindings[name]["sha256"] == expected, f"Source pin mismatch: {name}")
        bindings[Path(__file__).name] = binding(__file__)
        tiny = read_json(HERE / "paired_mehler_probe_tiny_v1.json")
        require(tiny["passed"] is True, "Tiny qualification must precede wide inputs")
        require(tiny["core_sha256"] == SOURCES["paired_mehler_probe_core_v1.py"],
                "Tiny qualified a different core")
        bindings["store_manifest"] = binding(STORE / "manifest.json")
        entries = read_json(STORE / "manifest.json")["roles"]["observer_fit"]["row_manifests"]
        rows = []
        for rank in RANKS:
            entry = entries[rank] if isinstance(entries, list) else entries[str(rank)]
            manifest = (STORE / entry["path"]).resolve()
            require(manifest == STORE / f"rows/observer_fit/{rank:04d}/row_manifest.json",
                    "Canonical fixture location differs")
            bindings[f"canonical_manifest_{rank}"] = binding(manifest)
            canonical = manifest.parent / "weights.npy"
            bindings[f"canonical_weights_{rank}"] = binding(canonical)
            require(bindings[f"canonical_weights_{rank}"]["sha256"] == INPUTS[rank]["weights.npy"],
                    "Canonical development weights differ")
            reference = HERE / f"rows_current_transport_shadow_v2/row_{rank:04d}"
            for name, expected in INPUTS[rank].items():
                key = f"reference_{rank}_{name}"
                bindings[key] = binding(reference / name)
                require(bindings[key]["sha256"] == expected, f"Reference pin mismatch: {key}")
            bindings[f"reference_metadata_{rank}"] = binding(reference / "result.json")
            metadata = read_json(reference / "result.json")
            require(metadata["target_access"] is False and metadata["rank"] == rank,
                    "Reference target-free provenance differs")
            row = read_json(manifest)
            rows.append(dict(rank=rank, mlp_id=row["mlp_id"], mlp_seed=row["mlp_seed"],
                             canonical_weights=str(canonical), reference=str(reference)))
            guard()
        started = dict(schema="paired_mehler_probe_rejection_v1", utc=utc(),
                       sources_and_inputs=bindings, rows=rows, pid=os.getpid(),
                       executable=sys.executable, nice=os.getpriority(os.PRIO_PROCESS, 0),
                       seeds={str(rank): [937109 + rank * 101 + group for group in range(GROUPS)]
                              for rank in RANKS},
                       n=N, depth=DEPTH, width=WIDTH, groups_per_net=GROUPS,
                       probes_per_replica=1, independent_probe_replicas=2,
                       precision="FP64 interpretation of fixed FP32 weights and references",
                       reference_ownership="archived compact-fullpair-P, not current native head parity",
                       truth_accessed=False, training=False, candidate=False,
                       c_floor=C_FLOOR, T=T, rejection_threshold=20*T,
                       wall_limit_seconds=WALL_LIMIT, peak_rss_limit_bytes=RSS_LIMIT,
                       decision="one aggregate after all eight groups; low Y inconclusive")
        write_json(OUT / "started.json", started)
        import numpy as np
        import scipy
        import paired_mehler_probe_core_v1 as core

        for row in rows:
            rank = row["rank"]
            reference = Path(row["reference"])
            weights = np.load(row["canonical_weights"], allow_pickle=False).astype(np.float64)
            mu = np.load(reference / "mu.npy", allow_pickle=False).astype(np.float64)
            sigma = np.load(reference / "sigma.npy", allow_pickle=False).astype(np.float64)
            require(weights.shape == (DEPTH, WIDTH, WIDTH), "Wrong weight shape")
            require(mu.shape == sigma.shape == (DEPTH, WIDTH), "Wrong reference shape")
            require(np.isfinite(weights).all() and np.isfinite(mu).all()
                    and np.isfinite(sigma).all() and np.all(sigma > 0), "Invalid input values")
            require(float(np.max(np.abs(mu/sigma))) <= 64, "Reference alpha outside tested limits")
            guard()
            for group in range(GROUPS):
                seed = 937109 + rank * 101 + group
                rng = np.random.default_rng(seed)
                x, z = rng.normal(size=(2, N, WIDTH))
                t = rng.uniform(0.0, 1.0, size=N)
                probes = rng.normal(size=(2, N, WIDTH))
                require(np.isfinite(x).all() and np.isfinite(z).all()
                        and np.isfinite(probes).all() and np.isfinite(t).all()
                        and np.all((t >= 0) & (t <= 1)), "Invalid random input")
                got = core.evaluate(weights, mu, sigma, x, z, t, probes,
                                    detailed=True, guard=guard)
                require(len(got["histories"]) == DEPTH, "Incomplete layer history")
                for history in got["histories"]:
                    require(all(np.isfinite(value).all() for value in history.values()),
                            "Nonfinite actual, tangent, source or recurrence history")
                got["histories"].clear()
                guard()
                require(np.isfinite(got["mean_difference"]).all()
                        and np.isfinite(got["statistic"]) and got["statistic"] >= 0,
                        "Invalid final difference or squared-norm statistic")
                results.append(dict(rank=rank, group=group, seed=seed, Y=got["statistic"],
                                    max_abs_standardized_deviation=got["max_abs_standardized_deviation"]))
                print(f"Completed fixed group {len(results)}/8; no interim statistical decision", flush=True)
            del weights, mu, sigma
        require(len(results) == 8, "Incomplete fixed screen")
        for name, item in bindings.items():
            require(binding(item["path"]) == item, f"Source/input changed: {name}")
        guard()
        mean_y = float(sum(item["Y"] for item in results)/len(results))
        rejected = mean_y >= 20*T
        result = dict(schema="paired_mehler_probe_rejection_v1", utc=utc(),
                      started_sha256=digest(OUT / "started.json"), groups=results,
                      mean_Y=mean_y, c_floor=C_FLOOR, T=T, rejection_threshold=20*T,
                      mean_Y_over_T=mean_y/T,
                      ideal_law_markov_tail_bound=min(1., T/mean_y) if mean_y > 0 else 1.,
                      decision="REJECT_FIXED_DEV2_NECESSARY_CONDITION" if rejected else "INCONCLUSIVE",
                      board_or_panel_score=False, raw_error_measured=False,
                      no_full_candidate_cost_measurement=True, truth_accessed=False,
                      source_and_input_bindings_unchanged=True,
                      diagnostic_literal_gemm_arithmetic_flops=8*139*N*WIDTH*(2*WIDTH-1)*2,
                      cost_scope="FP64 GEMM subtotal only; all other diagnostic work excluded",
                      test_scope="fixed FP64 reference/control/probe/ordinary-dense circuit; dev2 average only",
                      numerical_scope="finite tiny grid and algebra checks; not a uniform error bound or FP32 qualification",
                      diagnostic_history_checks="finite actual, tangent and recurrence arrays at every layer; copies included in measured resource use",
                      probability_scope="ideal Gaussian/uniform law; floating-point PRNG/evaluation approximation",
                      numpy_version=np.__version__, scipy_version=scipy.__version__,
                      wall_seconds=time.monotonic()-began, peak_rss_bytes=rss_bytes())
        write_json(OUT / "result.json", result)
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    except BaseException:
        failure = dict(utc=utc(), error=traceback.format_exc(), groups_completed=results,
                       bindings=bindings, wall_seconds=time.monotonic()-began,
                       peak_rss_bytes=rss_bytes(), decision="STOP; no qualifying statistic")
        write_json(OUT / "failure.json", failure)
        raise


if __name__ == "__main__":
    main()
