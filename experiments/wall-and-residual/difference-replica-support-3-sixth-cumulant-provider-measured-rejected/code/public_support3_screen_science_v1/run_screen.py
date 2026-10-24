#!/usr/bin/env python3
"""TAG support3-screen-science-v1: Gate-1 science-only execution of the frozen
315-cell split action-energy screen (15 layers x 3 registered roles x 7
sectors). Law bytes: SOURCE_LAW_ARCHIVE_V2.py (sha 049c71d6...) with exactly
the documented guard raise removed and flopscope.numpy -> numpy. Roles and
seeds from the consumed ROLE_SCOPE (sha 0e36053a...). Target-blind: weights
only, no truth, no estimator, no release, no authority claimed or consumed.
Gate per cell: S>0 and N/S<=1/9. ANY failure closes the spelling (no retry).
Custody: the per-role path-stream sha256 must equal the registered value."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import source_law_science_v1 as law

ROLE_SCOPE = json.loads(
    (HERE / "../public_q24_ng_k6_difference_replica_source_gate_role_scope_v1/ROLE_SCOPE.json"
     ).read_text())

report = {"schema": "support3-screen-science-v1",
          "gate": "S>0 and N/S<=1/9 in all 105 cells per role (315 total)",
          "roles": {}, "all_pass": True}
for role in ROLE_SCOPE["roles"]:
    rank = role["role_rank"]
    weights = np.load(role["weight_path"], mmap_mode="r", allow_pickle=False)
    assert list(weights.shape) == role["weight_shape"], weights.shape
    w_by_layer = [np.ascontiguousarray(weights[l], dtype=np.float32)
                  for l in range(16)]
    signal, noise, stream_sha = law._source_screen_literal(
        w_by_layer, int(role["mlp_seed"]))
    S = np.asarray(signal, dtype=np.float64)
    N = np.asarray(noise, dtype=np.float64)
    custody_ok = (stream_sha == role["path_stream_sha256"])
    cells = []
    fails = []
    for L in range(15):
        for s_i, sector in enumerate(law.SECTOR_ORDER):
            s_v, n_v = float(S[L, s_i]), float(N[L, s_i])
            ok = (s_v > 0.0) and (n_v >= 0.0) and (n_v <= s_v / 9.0)
            if not ok:
                fails.append({"layer": L, "sector": "".join(map(str, sector)),
                              "S": s_v, "N": n_v,
                              "N_over_S": (n_v / s_v if s_v > 0 else None)})
            cells.append(ok)
    n_pass = int(sum(cells))
    report["roles"][str(rank)] = {
        "mlp_seed": role["mlp_seed"],
        "custody_stream_sha_matches_registered": custody_ok,
        "cells_pass": n_pass, "cells_total": 105,
        "worst_N_over_S": float(np.max(N / np.where(S > 0, S, np.inf))),
        "min_S": float(np.min(S)),
        "failures": fails[:20],
        "S_by_layer_sector": [[float(x) for x in row] for row in S],
        "N_by_layer_sector": [[float(x) for x in row] for row in N],
    }
    if not custody_ok or n_pass != 105:
        report["all_pass"] = False
    print(json.dumps({"role": rank, "custody": custody_ok,
                      "pass": f"{n_pass}/105",
                      "worst_N_over_S": report["roles"][str(rank)]["worst_N_over_S"],
                      "min_S": report["roles"][str(rank)]["min_S"]}), flush=True)

(HERE / "SCREEN_RESULT.json").write_text(
    json.dumps(report, indent=1) + "\n")
print(json.dumps({"ALL_PASS": report["all_pass"]}))
