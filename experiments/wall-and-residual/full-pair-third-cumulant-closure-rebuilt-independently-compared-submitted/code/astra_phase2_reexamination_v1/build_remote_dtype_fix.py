"""Normalize dtype metadata across native and RemoteArray backends."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
source=HERE/"candidate_radial_gnn_splitread_v1.py"
target=HERE/"candidate_radial_gnn_remote_dtype_v2.py"
if target.exists():raise FileExistsError(target)
text=source.read_text()
old='a.dtype!=fnp.float32 or b.dtype!=fnp.float32'
new='str(a.dtype)!="float32" or str(b.dtype)!="float32"'
assert text.count(old)==1
target.write_text(text.replace(old,new))
print(target)
