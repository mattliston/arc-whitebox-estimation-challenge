"""Fixed, small host-input ownership diagnostic; no estimator or truth access.

Compare the native harness's memmap weight views with Whest's fnp.array
weight contract. Exercise only the weight-preparation/mixed-array pattern,
not a full card or a score. Three fixed repetitions per representation.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import weakref

HERE = Path(__file__).resolve().parent
OUT = HERE / "input_ownership_probe_v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run():
    assert not OUT.exists()
    assert os.getpriority(os.PRIO_PROCESS, 0) >= 10
    assert "data_gate/.venv" in sys.prefix and "anaconda3" not in sys.prefix
    OUT.mkdir()
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from whestbench.domain import MLP
    from threadpoolctl import threadpool_limits
    started = json.loads((HERE / "native_validation_bank_v1/started.json").read_text())
    path = next(Path(k) for k in started["source_bindings"]
                if k.endswith("/weights.npy") and "/observer_fit/0037/" in k)
    assert digest(path) == started["source_bindings"][str(path)]
    result = dict(passed=False, source_sha256=digest(__file__),
                  weights_sha256=digest(path), truth_accessed=False,
                  estimator_executed=False, rows=[])
    try:
        with threadpool_limits(limits=1):
            for representation in ("memmap", "fnp_array"):
                roots = []
                for repeat in range(3):
                    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
                    roots.append(weakref.ref(mapped))
                    weights = list(mapped) if representation == "memmap" else [
                        fnp.array(w, dtype=fnp.float32) for w in mapped]
                    layer_refs = [weakref.ref(w) for w in weights]
                    mlp = MLP(width=1024, depth=16, weights=weights)
                    with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=30, quiet=True) as ctx:
                        src = fnp.ones((1024, 8), dtype=fnp.float32)
                        for w in mlp.weights:
                            a = w.astype(fnp.float32)
                            a2 = a * a
                            a3 = a2 * a
                            value = a3.T @ src
                            assert tuple(value.shape) == (1024, 8)
                        bill = int(ctx.flops_used)
                    del mapped, weights, mlp, w, a, a2, a3, src, value, ctx
                    collected = gc.collect()
                    row = dict(representation=representation, repeat=repeat,
                               flops=bill, alive_mapping_roots=sum(r() is not None for r in roots),
                               alive_layer_refs=sum(r() is not None for r in layer_refs),
                               collected=collected,
                               peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    result["rows"].append(row)
                    print(json.dumps(row), flush=True)
                    assert row["peak_rss_bytes"] < 2_000_000_000
        result["passed"] = True
    finally:
        with (OUT / "result.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("60s guard")))
    signal.alarm(60)
    run()
