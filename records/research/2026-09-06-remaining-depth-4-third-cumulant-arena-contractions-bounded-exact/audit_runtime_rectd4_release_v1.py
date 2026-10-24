"""Bind the total-wall repair to native, real-client, panel and package gates."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

here = Path(__file__).resolve().parent
name = "candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py"
candidate = here / name
sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
assert sha == "3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757"
archive = here / "fullpair_runtime_rectd4_bulkcopies_v1_whest016.tar.gz"
out = here / "verified_local_fullpair_runtime_rectd4_bulkcopies_v1.json"
assert not out.exists()
native = json.loads((here / "fullpair_runtime_rectd4_bulkcopies_native_v1.json").read_text())
assert native["source_sha256"] == sha and native["pass_gate"] and native["repeat_byte_identical"]
assert [r["rank"] for r in native["trials"]] == [35, 100, 35]
tag = "fullpair_runtime_rectd4_bulkcopies_openblas_series_v1"
series = json.loads((here / f"{tag}.json").read_text())
resources = json.loads((here / f"{tag}_resources.json").read_text())
parent = json.loads((here / "fullpair_runtime_d5_t8192_bulkcopies_openblas_series_v1.json").read_text())
assert series["source_sha256"] == sha and series["status"] == "completed"
assert series["repeated_output_byte_identical"]
assert [r["rank"] for r in series["records"]] == [35, 100, 500, 3016500, 35]
assert resources["peak_combined_rss_bytes"] < 8*1024**3
ratios = []
for new, old in zip(series["records"], parent["records"], strict=True):
    assert new["rank"] == old["rank"]
    wall = new["timing"]["wall_time_s"]
    ratio = wall / old["timing"]["wall_time_s"]
    assert wall < 31 and ratio < .81
    assert new["timing"]["residual_wall_time_s"] < .13
    assert new["flops"] < 1.09*old["flops"]
    ratios.append(ratio)
assert statistics.mean(ratios) <= .80
panel_dir = here.parent / "public_g22_operator_features_v1"
rows = [json.loads(line) for line in (panel_dir / "panel_astra_fullpair_runtime_rectd4_bulkcopies_v1.jsonl").read_text().splitlines() if line.strip()]
old_rows = {r["rank"]:r for line in (panel_dir / "panel_astra_fullpair_runtime_d5_t8192_bulkcopies_v1.jsonl").read_text().splitlines() if line.strip() for r in [json.loads(line)]}
assert len(rows) == 14 and {r["rank"] for r in rows} == {35,100,150,200,250,300,400,500,*range(3016500,3016506)}
assert all(r["candidate"] == name and r["finite"] for r in rows)
assert all(math.isfinite(r[k]) for r in rows for k in ("final_mse_nc", "util", "adjusted"))
assert all(abs(r["final_mse_nc"] / old_rows[r["rank"]]["final_mse_nc"] - 1) < .003 for r in rows)
groups = {}
for label, group in (("public8", [r for r in rows if r["rank"] < 3016500]),
                     ("virgin6", [r for r in rows if r["rank"] >= 3016500]), ("combined14", rows)):
    groups[label] = dict(n=len(group), **{k:statistics.mean(r[k] for r in group) for k in ("final_mse_nc", "util", "adjusted")})
    assert groups[label]["adjusted"] < 5.6e-9
pack = json.loads((here / "package_runtime_rectd4_v1.json").read_text())
assert pack["passed"] and pack["source_sha256"] == sha
archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
assert archive_sha == pack["archive_sha256"]
with tarfile.open(archive, "r:gz") as tar:
    assert sorted(tar.getnames()) == ["estimator.py", "manifest.json"]
    assert hashlib.sha256(tar.extractfile("estimator.py").read()).hexdigest() == sha
    manifest = json.loads(tar.extractfile("manifest.json").read())
    assert manifest["files"] == [{"name":"estimator.py", "sha256":sha}]
record = dict(candidate=name, source_sha256=sha, archive=str(archive), archive_sha256=archive_sha,
    panel_groups=groups, mean_matched_wall_ratio=statistics.mean(ratios), per_call_wall_ratios=ratios,
    remote_series=series["records"], remote_peak_combined_rss_bytes=resources["peak_combined_rss_bytes"],
    local_release_pass=True, official_passing_grade=False, goal_achieved=False,
    prior_submission=329884, prior_failure="99/100 complete, jerry-kerr TIME_EXHAUSTED; unusable aggregate.",
    authorization="Matt pre-authorized bank repair submissions; no discussion-forum posts.",
    qualification="MacOS one-thread OpenBLAS/NumPy-free client is not the official Linux worker. Wall repair trades ~7.6% extra metered compute for ~22% faster local wall; no below-3e-9 claim.")
out.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({k:v for k,v in record.items() if k != "remote_series"}, indent=2), flush=True)
