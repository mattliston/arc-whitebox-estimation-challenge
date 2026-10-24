"""Bind exact runtime-repaired source to all required local release checks."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import tarfile

HERE=Path(__file__).resolve().parent
out=HERE/"verified_local_fullpair_runtime_d5_t8192_v1.json"
if out.exists():
    raise FileExistsError(out)
candidate=HERE/"candidate_fullpair_runtime_d5_t8192_v1.py"
archive=HERE/"fullpair_runtime_d5_t8192_v1_whest016.tar.gz"
sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
native=json.loads((HERE/"fullpair_runtime_d5_t8192_native_v1.json").read_text())
assert native["pass_gate"] and native["repeat_byte_identical"]
assert native["source_sha256"]==sha
assert [r["rank"] for r in native["trials"]]==[35,100,35]
assert all(r["peak_rss_bytes"]<8*1024**3 and r["output_max_abs"]<1e-5 for r in native["trials"])
assert all(r.get("delta_max_abs",0)<1e-6 and r.get("final_delta_max_abs",0)<1e-6 for r in native["trials"])
remotes=[]
for rank,tag in ((35,"fullpair_runtime_d5_t8192_remote_t1_v1"),
                 (100,"fullpair_runtime_d5_t8192_remote_r100_t1_v1"),
                 (3016500,"fullpair_runtime_d5_t8192_remote_virgin_t1_v1")):
    r=json.loads((HERE/f"{tag}.json").read_text())
    gate=json.loads((HERE/f"{tag}_runtime_gate.json").read_text())
    assert r["source_sha256"]==gate["source_sha256"]==sha
    assert gate["threads"]==1 and gate["under_8gib"]
    assert gate["runtime_margin_pass"] and r["timing"]["wall_time_s"]<60
    assert gate["residual_pass"] and r["timing"]["residual_wall_time_s"]<.4
    assert r["output_shape"]==[16,1024]
    remotes.append(dict(rank=rank,tag=tag,wall_s=r["timing"]["wall_time_s"],
                        residual_s=r["timing"]["residual_wall_time_s"],
                        peak_rss_bytes=gate["max_child_rss_bytes"],util=r["util"]))
with tarfile.open(archive,"r:gz") as tar:
    assert sorted(tar.getnames())==["estimator.py","manifest.json"]
    assert hashlib.sha256(tar.extractfile("estimator.py").read()).hexdigest()==sha
    manifest=json.loads(tar.extractfile("manifest.json").read())
    assert manifest["files"]==[{"name":"estimator.py","sha256":sha}]
panel=HERE.parent/"public_g22_operator_features_v1/panel_astra_fullpair_runtime_d5_t8192_v1.jsonl"
rows=[json.loads(line) for line in panel.read_text().splitlines() if line.strip()]
assert len(rows)==14 and {r["rank"] for r in rows}=={35,100,150,200,250,300,400,500,*range(3016500,3016506)}
assert all(r["candidate"]==candidate.name and r["finite"] for r in rows)
assert all(math.isfinite(r[key]) for r in rows for key in ("final_mse_nc","util","adjusted"))
groups={}
for name,rs in (("public8",[r for r in rows if r["rank"]<3016500]),
                ("virgin6",[r for r in rows if r["rank"]>=3016500]),("combined14",rows)):
    groups[name]=dict(n=len(rs),**{key:statistics.mean(r[key] for r in rs) for key in ("final_mse_nc","util","adjusted")})
receipt=dict(candidate=str(candidate),source_sha256=sha,archive=str(archive),
             archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
             panel=str(panel),panel_groups=groups,remote_one_thread_trials=remotes,
             repeated_native_parity_and_memory_pass=True,local_release_pass=True,
             prior_failure_submission=329843,prior_failure="4 x TIME_EXHAUSTED, no completed MLP; no score",
             upload_authorization="Matt explicitly authorized submission once the bank was fixed, then continuing score research.",
             official_submission=False,goal_achieved=False,
             qualification="60-second one-thread local margin is a stronger test, not an official-worker guarantee; an official passing grade is still required.")
out.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2),flush=True)
