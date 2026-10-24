"""Bind the best local panel, resource receipts, and archive to one source.

This verifies local readiness only. It neither uploads an archive nor
asserts an official score or that the approximately 4e-9 goal is achieved.
"""
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

HERE=Path(__file__).resolve().parent
out=HERE/"verified_local_fullpair_scalar_final_v1.json"
if out.exists():
    raise FileExistsError(out)
candidate=HERE/"candidate_fullpair_scalar_final_batched_d6_v1.py"
archive=HERE/"fullpair_scalar_final_batched_d6_v1_whest016.tar.gz"
sha=hashlib.sha256(candidate.read_bytes()).hexdigest()
native=json.loads((HERE/"fullpair_scalar_final_native_gate_v1.json").read_text())
remote=json.loads((HERE/"fullpair_scalar_final_remote_v1.json").read_text())
resource=json.loads((HERE/"fullpair_scalar_final_remote_v1_resources.json").read_text())
assert native["pass_gate"] and native["repeat_byte_identical"]
assert native["source_sha256"]==remote["source_sha256"]==sha
assert remote["timing"]["wall_time_s"]<120
assert remote["timing"]["residual_wall_time_s"]<.4
assert resource["under_8gib"] and resource["max_child_rss_bytes"]<8*1024**3
assert all(r["peak_rss_bytes"]<8*1024**3 for r in native["trials"])
with tarfile.open(archive,"r:gz") as tar:
    names=sorted(tar.getnames())
    assert names==["estimator.py","manifest.json"],names
    member=tar.extractfile("estimator.py").read()
    assert hashlib.sha256(member).hexdigest()==sha
panel_path=HERE.parent/"public_g22_operator_features_v1/panel_astra_fullpair_scalar_final_batched_d6_v1.jsonl"
rows=[json.loads(line) for line in panel_path.read_text().splitlines() if line.strip()]
expected={35,100,150,200,250,300,400,500,*range(3016500,3016506)}
assert len(rows)==14 and {r["rank"] for r in rows}==expected
assert all(r["finite"] and r["candidate"]==candidate.name for r in rows)
groups={}
for name,rs in (("public8",[r for r in rows if r["rank"]<3016500]),("virgin6",[r for r in rows if r["rank"]>=3016500]),("combined14",rows)):
    groups[name]={"n":len(rs),**{key:statistics.mean(r[key] for r in rs) for key in ("final_mse_nc","util","adjusted")}}
receipt=dict(candidate=str(candidate),source_sha256=sha,archive=str(archive),
             archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
             archive_members=names,panel=str(panel_path),panel_groups=groups,
             local_parity_and_resources_pass=True,official_submission=False,
             official_best_submission=329790,official_best_adjusted=1.436993412541317e-8,
             upload_authorization="Still pending explicit user approval after prior review rejection.",
             goal_achieved=False)
out.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt,indent=2),flush=True)
