"""Build a six-operation edge nonlinearity approximation; no estimator run."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

D = Path(__file__).resolve().parent
K = D.parent / "fable_oct19_kc_v1"
science = (K / "candidate_rel_v4a_sci_v1.py").read_text()
assert hashlib.sha256(science.encode()).hexdigest() == "e2f0c4cc4552c4770b383fa96ab2dc30b3a54dc76a27f2bcf614c386119975f9"
node = [n for n in ast.parse(science).body
        if isinstance(n, ast.FunctionDef) and n.name == "_ecut_raw"][0]
body = "\n".join(science.splitlines()[node.lineno - 1:node.end_lineno]) + "\n"
first = body.index("    if _ECUT_STANDIN:\n")
last = body.index("    block = ", first)
removed = body[first:last]
assert "fnp.minimum(w, 81.0" in removed and "fnp.tanh(h" in removed
replacement = """    # Six metered elementwise operations, replacing the ten-op stand-in.
    w = fnp.empty((ni, nj, int(h.shape[2])), dtype=fnp.float32)
    hold.append(fnp.multiply(h, h, out=w))
    hold.append(fnp.add(w, _O20_ECUT_RADICAL_B, out=w))
    hold.append(fnp.sqrt(w, out=w))
    hold.append(fnp.add(w, _O20_ECUT_RADICAL_A, out=w))
    hold.append(fnp.divide(h, w, out=h))
    hold.append(fnp.clip(h, -1.0, 1.0, out=h))
"""
body = body[:first] + replacement + body[last:]
guard = """    if not _O20_ECUT_RADICAL_ON or not _ECUT_STANDIN:
        return _O20_ECUT_RADICAL_PARENT(left, right, t, k, pf, gf, hold)
"""
needle = "    _e2_H = _E2Hold();"
assert body.count(needle) == 1
body = body.replace(needle, guard + needle, 1)
stamp = subprocess.check_output(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], text=True).strip()
tail = f"""# [astra-bill] {stamp}: six-op clipped radical edge approximation.
# Constants fitted only to the original scalar function on a fixed grid.
# This changes numeric outputs; no accuracy gain or identity is claimed.
assert _ECUT_C1 == 0.165 and _ECUT_C2 == 0.0098
_O20_ECUT_RADICAL_ON = True
_O20_ECUT_RADICAL_A = -0.2684473340048872
_O20_ECUT_RADICAL_B = 1.5215732663431163
_O20_ECUT_RADICAL_PARENT = _ecut_raw

""" + body
tail_path = D / "tail_ecut_radical_v1.py"
with tail_path.open("x") as handle:
    handle.write(tail)
outputs = []
for source_name, target, sha in (
        ("candidate_rel_v4a_sci_v1.py", "candidate_ecut_radical_sci_v1.py",
         "e2f0c4cc4552c4770b383fa96ab2dc30b3a54dc76a27f2bcf614c386119975f9"),
        ("candidate_rel_v4a.py", "candidate_ecut_radical.py",
         "b41627a6cc4d3c67280ab85fd7400ff07a0a167621326fcdb07690abe84ccac7")):
    source = (K / source_name).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == sha
    built = source + "\n" + tail
    compile(built, target, "exec")
    path = D / target
    with path.open("x") as handle:
        handle.write(built)
    outputs.append(dict(path=str(path), sha256=hashlib.sha256(built.encode()).hexdigest()))
(D / "ecut_radical_settings_v1.json").write_text(
    json.dumps([dict(_O20_ECUT_RADICAL_ON=False),
                dict(_O20_ECUT_RADICAL_ON=True)], indent=2) + "\n")
receipt = dict(
    utc=stamp, reference="science32_rel_v4a_m15", outputs=outputs,
    tail=dict(path=str(tail_path), sha256=hashlib.sha256(tail.encode()).hexdigest()),
    scalar_function="clip(h / (a + sqrt(b + h*h)), -1, 1)",
    constants=dict(a=-0.2684473340048872, b=1.5215732663431163),
    fit_data="Fixed scalar x grid [0,16] step .001, original scalar stand-in; no network or truth data",
    scalar_grid_max_abs_error=0.009119341223626676,
    scalar_grid_rms_error_0_4=0.005633087919102513,
    limitation="Approximate numeric candidate only. Tiny official billing and actual fresh MSE have not run at build time.",
)
(D / "ECUT_RADICAL_BUILD_v1.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt, indent=2))
