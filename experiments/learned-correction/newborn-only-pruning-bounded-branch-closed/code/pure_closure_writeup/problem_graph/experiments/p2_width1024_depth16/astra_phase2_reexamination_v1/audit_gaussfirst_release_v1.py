"""Bind the Gaussian-first candidate to its own completed local gates."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / "audit_fullpair_scalar_release_v1.py"
source = path.read_text()
edits = {
    "verified_local_fullpair_scalar_final_v1.json": "verified_local_fullpair_gaussfirst_v1.json",
    "candidate_fullpair_scalar_final_batched_d6_v1.py": "candidate_fullpair_scalar_final_gaussfirst_v1.py",
    "fullpair_scalar_final_batched_d6_v1_whest016.tar.gz": "fullpair_scalar_final_gaussfirst_v1_whest016.tar.gz",
    "fullpair_scalar_final_native_gate_v1.json": "fullpair_scalar_final_gaussfirst_native_v1.json",
    "fullpair_scalar_final_remote_v1.json": "fullpair_scalar_final_gaussfirst_remote_v1.json",
    "fullpair_scalar_final_remote_v1_resources.json": "fullpair_scalar_final_gaussfirst_remote_v1_resources.json",
    "panel_astra_fullpair_scalar_final_batched_d6_v1.jsonl": "panel_astra_fullpair_scalar_final_gaussfirst_v1.jsonl",
}
for old, new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
exec(compile(source, str(path), "exec"), dict(__file__=str(path), __name__="__main__"))
