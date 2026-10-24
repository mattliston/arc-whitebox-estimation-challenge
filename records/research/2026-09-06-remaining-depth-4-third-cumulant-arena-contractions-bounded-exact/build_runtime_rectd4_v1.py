"""Bank wall-time repair: D4 rectangles, retain bank D3/D4/D5 square schedule.

Same moment algorithm, factors, masks and learned payload as submission329884.
The separate D4 arena is lifetime-audited by the established layout builder.
Public FlopScope ops only; billed compute can rise and is measured normally.
"""
import ast
import hashlib
import json
from pathlib import Path
from packed_arena_generic import layout

here = Path(__file__).resolve().parent
parent = here / "candidate_fullpair_runtime_d5_t8192_bulkcopies_v1.py"
target = here / "candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py"
receipt = here / "build_runtime_rectd4_v1.json"
assert not target.exists() and not receipt.exists()
source = parent.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == "2a666b6fc74406d1d002472e9a7c08004b1b6f4ee71236ec2a287df1865fcbd6"
lines = source.splitlines(True)
funcs = {n.name: "".join(lines[n.lineno-1:n.end_lineno])
         for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
mm = funcs["_mm"]
old = "_packed_product_5(aa, bb, out, _SCRATCH[1] if n > 1024 else _SCRATCH[2])"
assert mm.count(old) == 1
mm = mm.replace(old, "_packed_product_4(aa, bb, out, _SCRATCH[1] if n > 1024 else _SCRATCH[2])")
push_size, push, _ = layout(1024, 1024, 8192, 4)
read_size, read, _ = layout(1024, 8192, 1024, 4)
square_size, _, _ = layout(1024, 1024, 1024, 5)
size = max(push_size, read_size, square_size)
source += "\n\n# Versioned total-wall repair: D4 rectangles, unchanged square schedules.\n"
for name, value in dict(_PACKED_ELEMENTS=size, _PACKED_PUSH=push, _PACKED_READ=read).items():
    source += name + " = " + repr(value) + "\n"
source += "\n" + mm
ast.parse(source)
target.write_text(source)
record = dict(candidate=target.name, parent=parent.name,
              sha256=hashlib.sha256(source.encode()).hexdigest(),
              arena_bytes=size*4, rectangle_depth=4, maximum_square_depth=5,
              semantics="Unchanged estimator equations and payload; different floating-point matrix association only.")
receipt.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
