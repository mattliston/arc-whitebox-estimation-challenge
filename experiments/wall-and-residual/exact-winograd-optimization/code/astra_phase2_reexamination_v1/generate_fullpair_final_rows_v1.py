"""Freeze the improved source/pair-map prefix for a matched scalar final head."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE/"generate_compact_final_rows_v1.py"
source=path.read_text()
edits={
    "from radial_binary_compact import predict":"from radial_compact_feed_fullpair_v1 import predict",
    "rows_compact_final_head_v1":"rows_fullpair_final_head_v1",
    "payload_binary_compact_g01_v1.npz":"payload_compact_feed_fullpair_v1.npz",
}
for old,new in edits.items():
    assert old in source,old
    source=source.replace(old,new)
exec(compile(source,str(path),"exec"),dict(__file__=str(path),__name__="__main__"))
