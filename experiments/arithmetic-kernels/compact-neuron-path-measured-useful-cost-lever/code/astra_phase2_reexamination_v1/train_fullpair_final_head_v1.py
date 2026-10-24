"""Same fixed final-only fit, on the newly frozen source/pair-map prefix."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
path=HERE/"train_compact_final_head_v1.py"
source=path.read_text()
edits={
    "compact_final_head_training_v1.json":"fullpair_final_head_training_v1.json",
    "payload_compact_final_head_v1.npz":"payload_fullpair_final_head_v1.npz",
    "payload_binary_compact_g01_v1.npz":"payload_compact_feed_fullpair_v1.npz",
    "rows_compact_final_head_v1":"rows_fullpair_final_head_v1",
    'torch.tensor([gain*(P["head"][15]@P["enc4_b"]+P["head_b"][15])], device=dev)':
        'torch.tensor([gain*(P["head"][15]@P["enc4_b"]+P["head_b"][15])], dtype=torch.float32, device=dev)',
}
for old,new in edits.items():
    assert old in source,old
    source=source.replace(old,new)
exec(compile(source,str(path),"exec"),dict(__file__=str(path),__name__="__main__"))
