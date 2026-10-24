#!/usr/bin/env python3
"""Deployed (host+J) output for a fresh instance. Usage: <virgin_seed>"""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_attribution import load_host

def main():
    seed = int(sys.argv[1])
    import flopscope as flops
    from threadpoolctl import threadpool_limits
    from whestbench import sample_mlp
    from whestbench.seeds import derive_estimator_seed
    mlp = sample_mlp(1024, 16, seed=seed)
    mlp.seed = int(derive_estimator_seed(seed))
    mod = load_host(f"fdep_{seed}")
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=600.0, quiet=True):
            dep = np.asarray(mod.Estimator().predict(mlp, 2**43), dtype=np.float64)
    np.save(HERE / f"fresh_{seed}_deployed.npy", dep)
    print(json.dumps({"seed": seed, "ok": bool(np.isfinite(dep).all())}))

if __name__ == "__main__":
    main()
