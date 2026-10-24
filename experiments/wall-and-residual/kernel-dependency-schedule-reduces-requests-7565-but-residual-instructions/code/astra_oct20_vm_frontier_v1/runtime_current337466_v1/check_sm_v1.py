"""Tiny official-API identity/bill checks; no estimator import or prediction."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS=HERE.parent.parent
D=RUNS/"astra_oct20_bill_v1"
K=RUNS/"fable_oct19_kc_v1"
assert not any(name.startswith("flopscope") for name in sys.modules)
import flopscope as fs
from flopscope import numpy as fnp

assert fs.__version__.startswith("0.12.3")
fs.configure(symmetry_warnings=False)
source = (K / "candidate_astra_o21_edge8_local90_per_layer960_v1_sci_v1.py").read_text()
tree = ast.parse(source)
env = dict(fnp=fnp, _rc_math=math, _BH_HOLD=[], _L3_STATS=dict(quads=0),
           _ASTRA_QL_ON=False)


def install(name, minimum=0, maximum=1000000):
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name == name and minimum <= n.lineno <= maximum]
    assert len(nodes) == 1, (name, [n.lineno for n in nodes])
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<extracted-current>", "exec"), env)


for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "_ASTRA_SM40_RECORDS":
                env[target.id] = ast.literal_eval(node.value)
env["_SMX_RECORDS"] = dict(env["_ASTRA_SM40_RECORDS"])
for node in tree.body:
    if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name)
            and t.value.id == "_SMX_RECORDS" for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<literal-record>", "exec"), env)
install("_Sm40Prepared")
install("_astra_sm40_linear_grouped", 20000, 20100)
env["_LINEAR_PARENT_GROUPED"] = lambda *a, **k: (_ for _ in ()).throw(AssertionError("unexpected fallback"))
env["_TPS_PARENT_GROUPED"] = env["_astra_sm40_linear_grouped"]
env["_TPS_META"] = {}; env["_TPS_COEFFICIENTS"] = {}
install("_astra_sm40_linear_grouped", 20930, 21000)
install("_sm40_program_key", 21420, 21450)
sm_parent = env["_astra_sm40_linear_grouped"]
exec((D / "tail_sm_direct_v1.py").read_text(), env)
sm_new = env["_astra_sm40_linear_grouped"]
rng = np.random.default_rng(20261020)
sm_rows = []
with fs.BudgetContext(flop_budget=2**39, quiet=True) as ctx:
    for record in env["_SMX_RECORDS"].values():
        for kind in ("L", "R"):
            plan = record["prepared"][kind]
            inputs_count = max(plan[0].values()) + 1
            for shape in ((2, 3), (3, 2), (4, 5)):
                for strided in (False, True):
                    inputs = tuple(fnp.asarray(rng.normal(size=shape).astype("float32"))
                                   for _ in range(inputs_count))
                    total = sum(len(nodes) for _, nodes in plan[1])
                    storage = fnp.empty((total, *shape), dtype=fnp.float32)
                    cuts = np.cumsum([len(nodes) for _, nodes in plan[1]])[:-1].tolist()
                    prepared = env["_Sm40Prepared"](
                        (tuple(fnp.split(storage, cuts, axis=0)),
                         tuple(fnp.unstack(storage, axis=0))))
                    count = len(plan[2])
                    if strided:
                        owner = fnp.empty((count, shape[0] + 2, shape[1] + 3), dtype=fnp.float32)
                        out = owner[:, :shape[0], :shape[1]]
                    else:
                        out = fnp.empty((count, *shape), dtype=fnp.float32)
                    stats = dict(temp_peak=0, temp_live=0)
                    before = ctx.flops_used
                    old = np.asarray(sm_parent(fnp, plan, inputs, shape, stats,
                                               out=out, prepared=prepared)).copy()
                    old_flops = ctx.flops_used - before
                    before = ctx.flops_used
                    new = np.asarray(sm_new(fnp, plan, inputs, shape, stats,
                                           out=out, prepared=prepared)).copy()
                    new_flops = ctx.flops_used - before
                    assert np.array_equal(old, new), (record["name"], kind, shape, strided)
                    assert old_flops - new_flops == count * math.prod(shape)
                    # A repeat uses cached argument routing; off uses the live
                    # parent rather than accidentally leaving a direct route.
                    assert np.array_equal(new, np.asarray(
                        sm_new(fnp, plan, inputs, shape, stats, out=out, prepared=prepared)))
                    env["_O20_SMDIRECT_ON"] = False
                    assert np.array_equal(old, np.asarray(
                        sm_new(fnp, plan, inputs, shape, stats, out=out, prepared=prepared)))
                    env["_O20_SMDIRECT_ON"] = True
                    key = env["_sm40_program_key"](plan, inputs, out, prepared)
                    sm_rows.append(dict(record=record["name"], kind=kind, shape=shape,
                                        strided_out=strided, exact=True,
                                        old_flops=old_flops,
                                        new_flops=new_flops,
                                        old_calls=len(plan[1]) + 1,
                                        new_calls=len(prepared.programs[key][2])))

report=dict(utc=subprocess.check_output(["date","-u","+%Y-%m-%dT%H:%M:%SZ"],text=True).strip(),
 source_sha256=hashlib.sha256(source.encode()).hexdigest(),official_version=fs.__version__,
 successful=True,tests=sm_rows,sm_stats=env["_O20_SMDIRECT_STATS"],
 exact_output_all_contiguous_strided_and_cold_warm=True,
 actual_terminal_parent_wrapper_included=True,
 limitations=["Tiny extracted official checks only; no full model/native residual claim.","Grouped old mode costs exactly one output stack and has fewer arithmetic calls."])
with (HERE/'SM_CHECK_v1.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps(dict(successful=True,cases=len(sm_rows),source_sha256=report['source_sha256'])))
