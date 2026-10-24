#!/usr/bin/env python3
"""TAG fable-engineer (E-53 v3): --ipt v3 post-stage = ipt_fragment_v3.py (v2 with fewer requests: chunk views by one unstack per level,
binding temporaries held, per-call quadrants by unstack, caps 160 per side). Apply LAST. Refuses to overwrite.
usage: build_ipt_v3.py <in.py> <out.py>"""
import sys, ast
from pathlib import Path
HERE = Path(__file__).resolve().parent
src_path, out_path = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
if out_path.exists():
    sys.exit(f"refusing to overwrite {out_path}")
src = src_path.read_text()
tree = ast.parse(src)
defs = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
assigns = {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
need_defs = {'_packed_product_cached', '_cp_d6_shared', '_packed_read5', '_cache_put', '_bb_product', '_w5_port_product_body'}
need_names = {'_BH_HOLD', '_CAP_BY_ID', '_SCRATCH'}
missing = sorted((need_defs - defs) | (need_names - assigns - defs))
if missing:
    sys.exit(f"input lacks {missing}")
if 'BEGIN FABLE-ENGINEER IN-PLACE TREE WINOGRAD KERNEL' in src:
    sys.exit("input already has --ipt")
out = src.rstrip("\n") + "\n" + (HERE / "ipt_fragment_v3.py").read_text()
ast.parse(out)
out_path.write_text(out)
print(f"wrote {out_path} ({len(out.splitlines())} lines)")
