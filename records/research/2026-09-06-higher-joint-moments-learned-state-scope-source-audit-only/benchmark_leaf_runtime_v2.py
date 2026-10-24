"""Use the production padded arena capacities for rectangular benchmarks.

The v1 small certificate passed, but its exact-aspect arena allocation was
not one of the production lifetime-certified capacity shapes. It stopped
before any large benchmark. Production uses fixed capacity views then slices
the active dimensions; this wrapper now tests that actual arrangement.
"""
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE/"benchmark_leaf_runtime_v1.py"
source=path.read_text()
edits={
    "leaf_runtime_benchmark_v1.json":"leaf_runtime_benchmark_v2.json",
    "            for kind,depth,name in":
        "            capacity=(m,k,n) if max(m,k,n)<=128 else ((1024,1024,4096) if n>1024 else (1024,4096,1024))\n            d6_reference=None\n            for kind,depth,name in",
    "streamed_layout(m,k,n)":"streamed_layout(*capacity)",
    "layout(m,k,n,depth)":"layout(*capacity,depth)",
    "                records.append(row)":
        "                if kind=='d6':\n                    d6_reference=np.asarray(result).copy()\n                if kind=='streamed6':\n                    row['versus_d6_max_abs']=float(np.max(np.abs(np.asarray(result)-d6_reference)))\n                    row['versus_d6_byte_identical']=bool(np.array_equal(np.asarray(result),d6_reference))\n                records.append(row)",
}
for old,new in edits.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(path),"exec"),dict(__file__=str(path),__name__="__main__"))
