"""Execute the pinned original census prefix through setup; no prediction loop."""
from pathlib import Path
import ast
import hashlib

source = Path(__file__).resolve().parent.parent/'census_boundary_setup_snapshot_v1.py'
raw = source.read_bytes()
assert hashlib.sha256(raw).hexdigest() == 'ee9218fac30b82b4796bc3bb7ecc0c8421b732f4af9f9b9023034f6ca7777281'
prefix, suffix = raw.decode().split('KERNEL = ', 1)
assert len(prefix.splitlines()) == 77 and suffix.startswith('("_packed_product_cached"')
tree = ast.parse(prefix)
assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('predict', 'load') for n in ast.walk(tree))
exec(compile(prefix, str(source), 'exec'), {'__name__': '__main__', '__file__': str(source)})
