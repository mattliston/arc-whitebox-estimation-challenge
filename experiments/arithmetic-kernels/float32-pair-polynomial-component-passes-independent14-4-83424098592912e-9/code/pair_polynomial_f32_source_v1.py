"""Extract a literal parent pair map and one frozen precision partition."""

import ast
from hashlib import sha256
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE / "candidate_response_rows_current_runtime_v1.py"
PARENT_SHA = "5a07832fee84b94c3a98589dc3ef19d718c1b062f5c4d3e04452a56ab9f11172"


def sources():
    raw = PARENT.read_bytes()
    if sha256(raw).hexdigest() != PARENT_SHA:
        raise RuntimeError("Parent hash differs")
    text = raw.decode()
    tree = ast.parse(text)
    names = ("gaussian_raw", "derivatives", "_nonlinear_pair_batch")
    found = {name: [node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == name]
             for name in names}
    if any(len(nodes) != 1 for nodes in found.values()):
        raise RuntimeError("Ambiguous literal source extraction")
    code = {name: ast.get_source_segment(text, found[name][0]) for name in names}
    pair = code["_nonlinear_pair_batch"]
    changes = (
        ("def _nonlinear_pair_batch(", "def _nonlinear_pair_batch_fp32poly("),
        ("ans=fnp.zeros((3,n,n),dtype=fnp.float64)",
         "ans=fnp.zeros((3,n,n),dtype=fnp.float32)"),
        ("rr=fnp.ones_like(C)", "\n    ".join([
            "F=F.astype(fnp.float32)", "G=G.astype(fnp.float32)",
            "rho=rho.astype(fnp.float32)", "T21=T21.astype(fnp.float32)",
            "T22=T22.astype(fnp.float32)", "T31=T31.astype(fnp.float32)",
            "t3=t3.astype(fnp.float32)", "t4=t4.astype(fnp.float32)",
            "rr=fnp.ones_like(rho)"])),
        ("cov=ans[0]*fnp.outer(s,s)",
         "assert str(ans.dtype)==str(rr.dtype)==str(ec.dtype)=='float32'\n"
         "    ans=ans.astype(fnp.float64)\n"
         "    cov=ans[0]*fnp.outer(s,s)"),
    )
    for old, new in changes:
        if pair.count(old) != 1:
            raise RuntimeError(f"Precision splice is not unique: {old}")
        pair = pair.replace(old, new)
    ast.parse(pair)
    return code, pair


def functions(flops, fnp):
    code, pair = sources()
    namespace = dict(flops=flops, fnp=fnp)
    for name in ("gaussian_raw", "derivatives", "_nonlinear_pair_batch"):
        exec(compile(code[name], f"<literal-parent:{name}>", "exec"), namespace)
    exec(compile(pair, "<fp32-polynomial-partition>", "exec"), namespace)
    return namespace["_nonlinear_pair_batch"], namespace["_nonlinear_pair_batch_fp32poly"]
