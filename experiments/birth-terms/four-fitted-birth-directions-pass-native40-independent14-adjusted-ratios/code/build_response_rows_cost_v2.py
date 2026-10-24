"""Versioned FlopScope scalar immutability fix; no numerical change."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "candidate_response_rows_cost_v1.py").read_text()
out = HERE / "candidate_response_rows_cost_v2.py"
assert not out.exists()
old = "    contraction -= fnp.sum(diagonal * a * beta)"
assert source.count(old) == 1
source = source.replace(old, "    contraction = contraction - fnp.sum(diagonal * a * beta)")
ast.parse(source)
out.write_text(source)
print(out, flush=True)
