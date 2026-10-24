#!/usr/bin/env python3
"""TAG p2-d11-census (dtype-attributed): per-call-site billed-flop census of the
workspace v4 card (sha 2fd2d40d) on stored net rank 35 at board conditions
(BudgetContext flop_budget=2^41, predict(mlp, 2^41), threadpool limit 2).

Observation-only monkeypatch of BudgetContext._append_op_record /
_replace_op_record: for every OpRecord we capture the innermost estimator.py
frame (file:line + function) and the live layer_index from the _predict frame,
then aggregate billed flops by (line, op) and by (layer, function).  Billing
itself is untouched; gate = flops_used must equal the sealed literal
2132366833051 and the output sha must equal the sealed rank-35 hash.

Run:  (data_gate venv python) census_runner.py
Knobs: none.  Writes CENSUS_RAW.json next to this file.
"""

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
CARD_DIR = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
CARD_SOURCE = CARD_DIR / "estimator.py"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3"
         / "sealed_target_blind_feature_store_v3")
EXPECTED_FLOPS_FIRST_CALL = 2132366833051
EXPECTED_OUTPUT_SHA = (
    "9f434f8f7f8eef8452560a13909026eaa397b4e85be9e92d7816e63cc9620047")
PHASE2_BUDGET = 2 ** 41
RANK = 35

import flopscope as flops
import flopscope._budget as fbudget
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed


def array_sha256(value):
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


CARD_PATH_STR = str(CARD_SOURCE)
# index-aligned with the budget's op_log: one site tuple per OpRecord
SITES = []


def _capture_site():
    """Innermost card frame (line, func) + layer phase from the live stack."""
    frame = sys._getframe(2)
    line = 0
    func = "?"
    phase = "outside_card"
    seen_card = False
    while frame is not None:
        code = frame.f_code
        if code.co_filename == CARD_PATH_STR:
            if not seen_card:
                line = frame.f_lineno
                func = code.co_name
                seen_card = True
            name = code.co_name
            if name == "_initialize_h1":
                phase = "h1_init"
            elif name in ("transport_correction", "materialize_parameters",
                          "_silu"):
                phase = "transport_correction"
            elif name == "_predict":
                if phase == "outside_card":
                    idx = frame.f_locals.get("layer_index")
                    phase = ("predict_pre_loop" if idx is None
                             else f"layer_{idx:02d}")
                break
            elif name == "_predict_corrected" or name == "predict":
                if phase == "outside_card":
                    phase = "predict_glue"
                break
        frame = frame.f_back
    return line, func, phase


_orig_append = fbudget.BudgetContext._append_op_record
_orig_replace = fbudget.BudgetContext._replace_op_record


def _patched_append(self, record):
    index = _orig_append(self, record)
    site = _capture_site()
    while len(SITES) <= index:
        SITES.append(None)
    SITES[index] = site
    return index


def _patched_replace(self, index, record):
    _orig_replace(self, index, record)
    # cost is re-read from op_log at aggregation time; site stays as captured


fbudget.BudgetContext._append_op_record = _patched_append
fbudget.BudgetContext._replace_op_record = _patched_replace


def load_card():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("billing_audit_card", CARD_SOURCE)
    mod = ilu.module_from_spec(spec)
    sys.modules["billing_audit_card"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    source_sha = hashlib.sha256(CARD_SOURCE.read_bytes()).hexdigest()
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][RANK]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    W = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
    mlp = MLP(width=1024, depth=16,
              weights=[W[layer] for layer in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=f"billing-audit-rank-{RANK}")
    mlp.validate()
    mod = load_card()
    estimator = mod.Estimator()
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        context = flops.BudgetContext(
            flop_budget=PHASE2_BUDGET,
            wall_time_limit_s=600.0,  # local instrumentation slack; the
            quiet=True)               # board wall gate is not under audit
        with context:
            value = estimator.predict(mlp, PHASE2_BUDGET)
    wall = time.perf_counter() - started
    output = np.ascontiguousarray(np.asarray(value), dtype=np.float32)
    out_sha = array_sha256(output)
    flops_used = int(context.flops_used)
    log = context.op_log
    gates = {
        "flops_exact_sealed": flops_used == EXPECTED_FLOPS_FIRST_CALL,
        "output_exact_sealed": out_sha == EXPECTED_OUTPUT_SHA,
        "sites_cover_log": len(SITES) == len(log) and None not in SITES,
        "source_sha_2fd2d40d": source_sha.startswith("2fd2d40d"),
    }
    print("flops_used", flops_used, "wall_s", round(wall, 2))
    print("gates", json.dumps(gates))

    by_line = {}
    by_phase = {}
    examples = {}
    for record, site in zip(log, SITES):
        line, func, phase = site
        key = f"{line}|{record.op_name}"
        agg = by_line.setdefault(key, {
            "line": line, "func": func, "op": record.op_name,
            "calls": 0, "flops": 0})
        agg["calls"] += 1
        agg["flops"] += int(record.flop_cost)
        pkey = f"{phase}|{func}"
        pag = by_phase.setdefault(pkey, {
            "phase": phase, "func": func, "calls": 0, "flops": 0})
        pag["calls"] += 1
        pag["flops"] += int(record.flop_cost)
        if key not in examples:
            examples[key] = {
                "subscripts": record.subscripts,
                "shapes": repr(record.shapes),
                "dtype": record.resolved_dtype,
            }
    for key, agg in by_line.items():
        agg.update(examples.get(key, {}))
    by_dtype = {}
    for record, site in zip(log, SITES):
        dt = str(record.resolved_dtype)
        d = by_dtype.setdefault(dt, {"calls": 0, "flops": 0})
        d["calls"] += 1
        d["flops"] += int(record.flop_cost)
    # per-layer covariance recursion (the pre_covariance double matmul)
    phase_totals = {}
    for pag in by_phase.values():
        pt = phase_totals.setdefault(pag["phase"], {"calls": 0, "flops": 0})
        pt["calls"] += pag["calls"]
        pt["flops"] += pag["flops"]
    result = {
        "by_dtype": by_dtype,
        "schema": "p2-d11-census-v1",
        "card_source_sha256": source_sha,
        "rank": RANK,
        "flop_budget": PHASE2_BUDGET,
        "flops_used": flops_used,
        "counted_factor_this_card": flops_used / PHASE2_BUDGET,
        "wall_s": wall,
        "gates": gates,
        "op_records": len(log),
        "by_line": sorted(by_line.values(), key=lambda a: -a["flops"]),
        "by_phase_func": sorted(by_phase.values(), key=lambda a: -a["flops"]),
        "phase_totals": phase_totals,
    }
    (HERE / "CENSUS_RAW.json").write_text(
        json.dumps(result, indent=1, default=str) + "\n", encoding="utf-8")
    print("wrote", HERE / "CENSUS_RAW.json")
    top = result["by_line"][:12]
    for agg in top:
        print(f"line {agg['line']:5d} {agg['func']:24s} {agg['op']:12s} "
              f"calls={agg['calls']:5d} flops={agg['flops']:15d}")


if __name__ == "__main__":
    main()
