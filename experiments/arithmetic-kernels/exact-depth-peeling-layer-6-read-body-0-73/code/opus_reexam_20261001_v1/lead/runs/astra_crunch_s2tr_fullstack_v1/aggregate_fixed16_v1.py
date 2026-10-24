"""Combine the frozen fixed16 cohort without selecting on outcomes."""
import json
import math
import statistics
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parent
SHA = "48c126c61f8d4f571768529d9d6610d8e732de24107fce75d9422e3d417a9700"
root_ranks = [16, 35, 150, 200, 250, 259, 277, 287]
boundary_ranks = [300, 310, 312, 316, 320, 328, 333]
rows = []

def native(rank, directory):
    result = json.loads((directory / f"res_{rank}_v1.json").read_text())
    guard = json.loads((directory / f"guard_{rank}_v1.json").read_text())
    assert guard["completed"] and guard["exit_code"] == 0 and guard["whole_group_retired"]
    assert result["source_sha"] == SHA
    control = json.loads((R / f"opus_kinkoracle_v1/res_ctl_{rank}.json").read_text())
    return dict(rank=rank, U=result["U"], control_U=control["U"],
                cold_flops=result["flops"], control_cold_flops=control["flops"],
                harness="scientific_native", raw_pct=100 * (result["U"] / control["U"] - 1))

rows.extend(native(rank, D) for rank in root_ranks)
rows.append(native(293, R / "astra_crunch_compute_v1/peel_fullstack8_v1"))
batch = json.loads((R / "astra_crunch_compute_v1/peel_boundary_v1/SCIENCE7_PARTIAL_v1.json").read_text())
assert batch["source_sha256"] == SHA
by_rank = {row["rank"]: row for row in batch["rows"]}
assert set(by_rank) == set(boundary_ranks), "Fixed cohort is not complete"
for rank in boundary_ranks:
    row = by_rank[rank]
    assert row["guard_pass"] and row["exit_code"] == row["grade_exit_code"] == 0
    rows.append(dict(rank=rank, U=row["U"], control_U=row["control_U"],
                     cold_flops=row["flops"], control_cold_flops=row["control_cold_flops"],
                     harness="production_boundary", raw_pct=row["raw_pct"]))

raw = [row["raw_pct"] for row in rows]
result = dict(source_sha=SHA, n=len(rows), rows=rows,
              mean_raw_pct=statistics.mean(raw),
              se_raw_pct=statistics.stdev(raw) / math.sqrt(len(raw)),
              wins=sum(value < 0 for value in raw),
              pooled_raw_pct=100 * (sum(row["U"] for row in rows) / sum(row["control_U"] for row in rows) - 1),
              control_bill_weighted_raw_pct=100 * (
                  sum(row["U"] * row["control_cold_flops"] for row in rows) /
                  sum(row["control_U"] * row["control_cold_flops"] for row in rows) - 1),
              note="Science only; final-source warm bill, setup, residual and prediction parity require separate release evidence. Native300 guard failure is preserved; its result is from the actual two-process boundary with standard limits.")
out = D / "FIXED16_SCIENCE_v1.json"
assert not out.exists()
out.write_text(json.dumps(result, indent=2))
print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
