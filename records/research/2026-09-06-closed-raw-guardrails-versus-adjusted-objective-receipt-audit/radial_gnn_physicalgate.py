"""Matched inference: use the physical activation probability in messages."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "radial_gnn_numpy.py").read_text()
old = '    x = np.clip((x.astype(np.float32)-P["centers"][l])/P["scales"][l],-8,8)\n    g = x[:,5:6]'
new = '    g = x[:,5:6].astype(np.float32)\n    x = np.clip((x.astype(np.float32)-P["centers"][l])/P["scales"][l],-8,8)'
assert source.count(old) == 1
exec(compile(source.replace(old, new), str(HERE / "radial_gnn_numpy.py"), "exec"))
