"""Build current-card NT join arms; retain compact JSK and Qcore reuse.

The original NT test predates compact JSK. Extra landmark columns use a thin
plain product; the existing rank-384 block retains its current kernel.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

K = Path(__file__).resolve().parent
source = (K / "candidate_rel_v4a_sci_v1.py").read_text()
nodes = [n for n in ast.parse(source).body
         if isinstance(n, ast.FunctionDef) and n.name == "_astra_bcc_push_impl"]
node = nodes[-1]
function = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"


def replace(old, new):
    global function
    assert function.count(old) == 1, old
    function = function.replace(old, new)


replace("        Omega = w32[previous, :rank]",
        "        Omega = w32[previous, :rank]\n"
        "        _nt_p = int(_NT_P.get(layer, 0)) if _JOIN_NYSTROM else 0")
replace("            _jn_rows = fnp.sort(fnp.argsort(-(_jn_d.astype(fnp.float64)), kind='stable')[:rank])",
        "            _jn_rows = (fnp.argsort(-_jn_d.astype(fnp.float64), kind='stable')[:rank + _nt_p] if _nt_p else\n"
        "                        fnp.sort(fnp.argsort(-(_jn_d.astype(fnp.float64)), kind='stable')[:rank]))")
replace("            Y = _shared_old384_mm(J1, weight[_jn_cols1][:, None] * J1[_jn_rows, :].T)",
        "            _nt_rhs = weight[_jn_cols1][:, None] * J1[_jn_rows, :].T\n"
        "            Y = _nt_split_product(J1, _nt_rhs, rank) if _nt_p else _shared_old384_mm(J1, _nt_rhs)\n"
        "            del _nt_rhs")
replace("                Y = Y + _shared_old384_mm(_jn_Qcore, (Qp[_jn_rows, :]).T); del _jn_Qcore",
        "                _nt_rhs = Qp[_jn_rows, :].T\n"
        "                Y = Y + (_nt_split_product(_jn_Qcore, _nt_rhs, rank) if _nt_p else _shared_old384_mm(_jn_Qcore, _nt_rhs))\n"
        "                del _nt_rhs, _jn_Qcore")
replace("        if _need_core:   # opus-engineer a08:",
        "        if _nt_p:\n"
        "            Qn, core = _nt_truncate_current(rank, Qn, unused_R, _jn_rows, _need_core)\n"
        "        elif _need_core:   # opus-engineer a08:")

helper = '''
_NT_P = {}
_NT_Q = 128
_NT_SPLIT = True


def _nt_split_product(a, b, rank):
    if not _NT_SPLIT:
        return _shared_old384_mm(a, b)
    head = _shared_old384_mm(a, b[:, :rank])
    tail = fnp.matmul(a, b[:, rank:])
    return fnp.concatenate((head, tail), axis=1)


def _nt_truncate_current(rank, Q, R, rows, need_core):
    k = int(Q.shape[1])
    C = fnp.linalg.solve(Q[rows, :].astype(fnp.float32), R.astype(fnp.float32).T).astype(fnp.float32)
    C = (C + C.T) * 0.5
    h = rank - int(_NT_Q)
    assert 0 < h < rank and k > rank
    CT = C[:, h:].copy()
    CTT, CHT = CT[h:, :].copy(), CT[:h, :].copy()
    M = fnp.matmul(CTT, CTT) + 2.0 * fnp.matmul(CHT.T, CHT)
    M = (M + M.T) * 0.5
    ev, V = fnp.linalg.eigh(M)
    X = V[:, (k - h) - _NT_Q:][:, ::-1].copy()
    Qn = fnp.concatenate((Q[:, :h], fnp.matmul(Q[:, h:], X)), axis=1)
    core = None
    if need_core:
        CX = fnp.matmul(CT, X)
        HX = CX[:h, :].copy()
        core = fnp.concatenate((
            fnp.concatenate((C[:h, :h], HX), axis=1),
            fnp.concatenate((HX.T, fnp.matmul(X.T, CX[h:, :])), axis=1),
        ), axis=0).astype(fnp.float32)
        core = (core + core.T) * 0.5
    return Qn.astype(fnp.float32), core

'''
stamp = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()
common = f"\n\n# [astra-lead] {stamp} NT current compact JSK, split extra columns\n" + helper + function
variants = {
    "ntc32_all": ({7: 32, 10: 32, 13: 32}, 128),
    "ntc64_all": ({7: 64, 10: 64, 13: 64}, 192),
    "ntc32_l7": ({7: 32}, 128),
}
for tag, (extra, q) in variants.items():
    tail = common + f"\n_NT_P = {extra!r}\n_NT_Q = {q}\n"
    compile(tail, tag, "exec")
    path = K / f"tail_{tag}.txt"
    with path.open("x") as f:
        f.write(tail)
    subprocess.run(["bash", str(K / "make_stack_v1.sh"), tag, str(path)], check=True)
settings = [{"_NT_P": {}, "_NT_Q": 128},
            {"_NT_P": {7: 32, 10: 32, 13: 32}, "_NT_Q": 128},
            {"_NT_P": {7: 64, 10: 64, 13: 64}, "_NT_Q": 192}]
(K / "ntc_screen_settings_v1.json").write_text(json.dumps(settings, indent=2) + "\n")
print(json.dumps(dict(utc=stamp, base_sha256=hashlib.sha256(source.encode()).hexdigest(),
                     variants=list(variants), live_function_line=node.lineno)))
