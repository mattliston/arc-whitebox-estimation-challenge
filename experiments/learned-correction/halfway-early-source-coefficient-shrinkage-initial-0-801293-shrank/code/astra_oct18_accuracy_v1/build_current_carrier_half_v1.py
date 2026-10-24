"""Root-chosen t=.5 toward formal early-source coefficients; no roster changes."""
import ast
import copy
import hashlib
import json
from pathlib import Path

D = Path(__file__).resolve().parent
build = json.loads((D / "BUILD_GROW_CARRIER_CALIBRATION_v1.json").read_text())
P = Path(build["parent"])
src = P.read_text()
assert hashlib.sha256(src.encode()).hexdigest() == build["parent_sha256"]
tree = ast.parse(src)
ns = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        names = [x.id for x in node.targets if isinstance(x, ast.Name)]
        if any(n.startswith("_NP_") or n == "_S19_LAM" for n in names):
            try:
                value = ast.literal_eval(node.value)
            except (TypeError, ValueError):
                continue
            for name in names:
                ns[name] = value
for name in ("_np_ages", "_np_key", "_np_tables"):
    node = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name][0]
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<parent-pure-table>", "exec"), ns)
base = ns["_np_tables"](1024)
theta = {k: 1 + .5 * (v - 1) for k, v in build["baseline_theta"].items()}
ns.update(theta)
other = ns["_np_tables"](1024)
for key in base:
    if key not in ("coefH", "key"):
        assert base[key] == other[key], key
patched = copy.deepcopy(base)
patched["coefH"][1:7] = other["coefH"][1:7]
ns.update(build["baseline_theta"])
assert base["key"] == ns["_np_key"](1024)
fragment = (
    "\n# ASTRA fixed current-roster carrier t=.5; source law only at births1-6.\n"
    f"# theta={theta!r}; complete coupled coefH regenerated offline.\n"
    f"_NP_TABLES['coefH'][1:7] = {patched['coefH'][1:7]!r}\n"
)
nodes = ast.parse(fragment).body
assert len(nodes) == 1 and ast.unparse(nodes[0].targets[0]) == "_NP_TABLES['coefH'][1:7]"
assert patched["coefH"][:1] == base["coefH"][:1] and patched["coefH"][7:] == base["coefH"][7:]
output = src + fragment
dest = D / "candidate_current_carrier_half_HELD_v1.py"
compile(output, str(dest), "exec")
dest.open("x").write(output)
f = D / "fragment_current_carrier_half_v1.py.txt"
f.open("x").write(fragment)
anchor = 1.4515052706908603e-9
conditional = -.4268755426504807 * 2
receipt = {
    "status": "BUILT_HELD_UNTESTED_TRANSFER_AND_STEP",
    "parent": str(P), "parent_sha256": build["parent_sha256"],
    "candidate": str(dest), "sha256": hashlib.sha256(output.encode()).hexdigest(),
    "fragment": str(f), "t": .5, "theta": theta,
    "scope": "Only one coefficient literal assignment for births1-6; all current v61j rosters and runtime source retained.",
    "shared_audit": "Same complete polynomial-table generator and same index/shapes as t=.25; all non-coefH tables and later rows exact. Globals/key unchanged, so _np_const binds the patched matrix as previously certified.",
    "source_only_no_roster_hooks": "_EBF_" not in fragment and "def " not in fragment,
    "direct_extra_products": 0, "direct_extra_requests": 0,
    "conditional_linear_extrapolation_adjusted_pct": conditional,
    "conditional_linear_extrapolation_score": anchor * (1 + conditional / 100),
    "caveat": "Transfer from grown to current rosters and doubling the step are untested. This is a hypothesis, not a forecast or banked gain.",
    "root_plan": "Test only fixed t=.5 first on35/150/200 after scheduled metadata/broad2; extend7 only if useful. t=.25 remains a held fallback, not a scheduled grid.",
}
(D / "BUILD_CURRENT_CARRIER_HALF_v1.json").open("x").write(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
