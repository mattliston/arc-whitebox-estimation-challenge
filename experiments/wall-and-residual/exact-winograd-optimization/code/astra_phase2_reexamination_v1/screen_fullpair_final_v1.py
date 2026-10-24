"""Matched frozen-prefix comparison for the trace-aware scalar final read."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE/"screen_compact_final_v1.py"
source=path.read_text()
edits={
    "from compact_final_reference_v1 import predict":
        "from compact_fullpair_final_reference_v1 import predict",
    "payload_binary_compact_g01_v1.npz":"payload_compact_feed_fullpair_v1.npz",
    "payload_compact_final_head_v1.npz":"payload_fullpair_final_head_v1.npz",
    "rows_compact_final_head_v1":"rows_fullpair_final_head_v1",
}
for old,new in edits.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
exec(compile(source,str(path),"exec"),dict(__file__=str(path),__name__="__main__"))
