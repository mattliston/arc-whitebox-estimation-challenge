"""One matched current-card graph round, preserving the frozen parent."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
runner = HERE / "chain_radial_training.py"
source = runner.read_text()
edits = {
    "rows_radial_covmetric_v1": "rows_current_graph_v1",
    "ASTRA_RADIALCOV_V1_C64_E40": "ASTRA_CURRENT_GRAPH_V1_C64_E40",
    "chain_radial_training_v1.json": "chain_current_graph_training_v1.json",
    "payload_radialcov_v1.npz": "payload_current_graph_v1.npz",
    'str(HERE/"generate_radial_rows.py")': 'str(HERE/"generate_current_graph_rows_v1.py")',
    'frozen_names = ("k3_radial_numpy.py","radial_features.py","generate_radial_rows.py")':
        'frozen_names = ("k3_radial_numpy.py","radial_binary_compact.py",'
        '"radial_compact_radialfeed_v1.py","radial_compact_feed_fullpair_v1.py",'
        '"radial_mehler.py","radial_features.py","radial_graph_features_v1.py",'
        '"generate_radial_rows.py","generate_current_graph_rows_v1.py")',
    "row_seconds = time.time()-started":
        'assert all(r["parent_host_sigma_features_byte_exact"] for r in rows)\n'
        'profile=[sum(r["one_step_mse"][l] for r in rows)/len(rows) for l in range(16)]\n'
        'assert 1e-10<profile[-1]<2*profile[10],("unhealthy honest rows",profile)\n'
        'print("GRAPH_200_PARENT_BYTE_PARITY_PASS",profile,flush=True)\n'
        'row_seconds = time.time()-started',
}
for old, new in edits.items():
    assert old in source, old
    source = source.replace(old, new)
exec(compile(source, str(runner), "exec"), dict(__file__=str(runner), __name__="__main__"))
