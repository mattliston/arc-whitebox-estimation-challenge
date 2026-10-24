"""Two unlearned cost controls; neither is a submission candidate."""
import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent_path = HERE / "candidate_fullpair_runtime_d5_t8192_v1.py"
source = parent_path.read_text()
lines = source.splitlines(keepends=True)
tree = ast.parse(source)
functions = {node.name: "".join(lines[node.lineno-1:node.end_lineno])
             for node in tree.body if isinstance(node, ast.FunctionDef)}
predict = functions["_predict_binary"]
edits = {
    "        live = alive[:, None] * alive[None, :]":
        "        live = alive[:, None] * alive[None, :]\n"
        "        response_gate=flops.stats.norm.cdf(mu/fnp.sqrt(fnp.maximum(vv,1e-10)))\n"
        "        response_score=fnp.where(alive,fnp.minimum(response_gate,1-response_gate),-1.)\n"
        "        response_indices,response_mask=select(response_score,gate_floor)\n"
        "        omitted=alive & ~response_mask",
    "            sub = pp[live_indices]\n            fnp.put(cp_pre, live_indices[:, None] * n + live_indices[None, :], plane(sub, lam))":
        "            left=pp[response_indices]; right=pp[live_indices]\n"
        "            response_plane=_mm(left*left,(right*lam).T).astype(fnp.float64)\n"
        "            fnp.put(cp_pre,response_indices[:,None]*n+live_indices[None,:],response_plane)",
    "        host = m.copy()":
        "        host = m.copy()\n"
        "        if len(lam):\n"
        "            source_sum=omitted_radial_source(pp,lam,diagonal,omitted,alive,aux,host)\n"
        "            denominator=fnp.trace(C)**2+2*fnp.sum(C*C) if metric=='covariance' else n*(n+2)\n"
        "            radial_new=radial_new+source_sum/denominator",
}
for old, new in edits.items():
    assert predict.count(old) == 1, old
    predict = predict.replace(old, new)
helper_source = (HERE / "radial_response_rows_v1.py").read_text()
helper_lines = helper_source.splitlines(keepends=True)
helper_node = next(node for node in ast.parse(helper_source).body
                   if isinstance(node,ast.FunctionDef) and node.name=="omitted_radial_source")
helper = "".join(helper_lines[helper_node.lineno-1:helper_node.end_lineno]).replace("np.", "fnp.")
control = """

# Diagnostic only: retain payload bytes but apply no learned corrections.
class Estimator(BaseEstimator):
    def predict(self,mlp,budget):
        assert int(mlp.width)==1024 and int(mlp.depth)==16 and int(budget)>=2**41
        return _predict_binary(mlp)
"""
outputs = {"candidate_response_rows_parent_cost_v1.py": source + control,
           "candidate_response_rows_cost_v1.py": source + "\n\n" + helper + "\n\n" + predict + control}
assert all(not (HERE/name).exists() for name in outputs)
for name, text in outputs.items():
    ast.parse(text)
    (HERE/name).write_text(text)
    print(name, flush=True)
