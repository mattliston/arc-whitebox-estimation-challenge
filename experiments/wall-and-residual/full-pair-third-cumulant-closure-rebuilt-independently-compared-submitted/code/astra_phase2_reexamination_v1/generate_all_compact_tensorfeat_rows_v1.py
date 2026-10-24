"""Two CPU workers only; deliberately does NOT launch an MPS trainer."""
from pathlib import Path

HERE=Path(__file__).resolve().parent
runner=HERE/"chain_radial_training.py"
source=runner.read_text().split("row_seconds = time.time()-started")[0]
edits={
    "rows_radial_covmetric_v1":"rows_compact_tensorfeat_v1",
    "ASTRA_RADIALCOV_V1_C64_E40":"ASTRA_COMPACT_TENSORFEAT_V1_C64_E40",
    "chain_radial_training_v1.json":"rows_compact_tensorfeat_v1_receipt.json",
    'str(HERE/"generate_radial_rows.py")':'str(HERE/"generate_compact_tensorfeat_rows_v1.py")',
    "ThreadPoolExecutor(max_workers=3)":"ThreadPoolExecutor(max_workers=2)",
    'frozen_names = ("k3_radial_numpy.py","radial_features.py","generate_radial_rows.py")':
        'frozen_names = ("k3_radial_numpy.py","radial_cubic.py","radial_binary_compact.py","compact_tensor_directions_v1.py","radial_compact_tensorfeat_v1.py","radial_features.py","generate_radial_rows.py","generate_compact_tensorfeat_rows_v1.py")',
}
for old,new in edits.items():
    assert old in source,old
    source=source.replace(old,new)
source+='''
import numpy as np
profile=[sum(r["one_step_mse"][l] for r in rows)/len(rows) for l in range(16)]
assert 1e-10<profile[-1]<2*profile[10],("unhealthy honest rows",profile)
for rank in range(35,235):
    fresh=ROWS/f"row_{rank:04d}"
    old=HERE/"rows_binary_compact_g01_v1"/f"row_{rank:04d}"
    assert np.array_equal(np.load(fresh/"host.npy"),np.load(old/"host.npy")),rank
receipt.write_text(json.dumps(dict(rows=200,workers=2,wall_s=time.time()-started,source_hashes=hashes,host_all_byte_identical=True,profile=profile),indent=2)+"\\n")
print("TENSORFEAT_ROWS_COMPLETE; MPS training NOT launched",flush=True)
'''
exec(compile(source,str(runner),"exec"),dict(__file__=str(runner),__name__="__main__"))
