#!/usr/bin/env python3
"""Inert, reversible source builder. No estimator or numerical imports.

Only --build writes build_v1/{estimator.py,zero_estimator.py,build.json,
completion.json}. The zero source uses a literal disabled branch. Building
sources neither admits a numerical run nor supplies score evidence.
"""
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import traceback

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PARENT = NEW / "birth_direction_fit_v1/build_terminal_v1/estimator.py"
PARENT_SHA = "d50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451"
PROTOCOL = HERE / "PROTOCOL.md"
PROTOCOL_SHA = "f4bf2907fa1973b6076d6f485cd2436d6b5bf13cfe26077653e52af3c3e59dad"
DESIGN = NEW / "strategic_reset_20260909_v1/terminal_fourth_tree_route.md"
DESIGN_SHA = "078cdf2b62b71acf249a1fd1ea3ccd3bfd7f8a745fe59540bd9b579cc967be00"
OUT = HERE / "build_v1"

# This exact bundle is also exercised by check_tiny.py, without importing the
# parent. All source arithmetic runs inside the original participant context.
HELPERS = '''def _terminal_tree_source(Cpre, Cpost, aux, Wnext):
    # Complete THREE-EDGE Gaussian fourth sector, not the full cumulant.
    # Full unmasked legs; symmetrization defines only this auxiliary source.
    C = (Cpre + Cpre.T) * 0.5
    Wnext = fnp.asarray(Wnext, dtype=fnp.float64)
    g = fnp.asarray(aux['gate'], dtype=fnp.float64)
    pdf = fnp.asarray(aux['pdf'], dtype=fnp.float64)
    sigma = fnp.asarray(aux['sigma'], dtype=fnp.float64)
    alpha = fnp.asarray(aux['alpha'], dtype=fnp.float64)
    pre_v = fnp.asarray(aux['pre_v'], dtype=fnp.float64)
    n = len(g)
    assert C.shape == Cpost.shape == Wnext.shape == (n, n)
    assert all(str(x.dtype) == 'float64' for x in (C, Cpost, Wnext, g, pdf, sigma, alpha, pre_v))
    assert all(x.shape == (n,) for x in (pdf, sigma, alpha, pre_v))
    assert all(fnp.all(fnp.isfinite(x)) for x in (C, Cpost, Wnext, g, pdf, sigma, alpha, pre_v))
    assert fnp.all(pre_v >= 1e-10) and fnp.all(sigma > 0)
    d = pdf / sigma
    e = -alpha * pdf / pre_v
    B = C @ (g[:, None] * Wnext)
    T = (d[:, None] * Wnext) * B
    Y = C @ T
    rays = 12 * fnp.sum(T * Y, axis=0) + 4 * fnp.sum((e[:, None] * Wnext) * (B * B * B), axis=0)
    del B, T, Y, Wnext
    v = fnp.diag(C)
    h = d * g
    hv = h * v
    C2 = C * C
    double_trace = 4 * (fnp.sum(((d * d + e * g) * v) * (C2 @ (g * g)))
                        + fnp.sum(hv * (C @ hv))
                        + fnp.sum(h * ((C2 * C) @ h)))
    # Match the actual reset denominator BEFORE the P hook and alive mask.
    denominator = fnp.trace(Cpost) ** 2 + 2 * fnp.sum(Cpost * Cpost)
    assert fnp.isfinite(denominator) and denominator > 0
    rho_source = 3 * double_trace / denominator
    assert str(rays.dtype) == 'float64'
    assert fnp.all(fnp.isfinite(rays)) and fnp.isfinite(rho_source)
    return rays, rho_source


def _terminal_tree_mean_delta(rays, rho_source, vv, aux):
    # vv is the actual layer15 raw variance, not a recomputed source metric.
    a = aux['alpha']
    pdf = aux['pdf']
    v = aux['pre_v']
    s = aux['sigma']
    assert all(str(x.dtype) == 'float64' for x in (rays, vv, a, pdf, v, s))
    assert all(x.shape == rays.shape for x in (vv, a, pdf, v, s))
    assert all(fnp.all(fnp.isfinite(x)) for x in (rays, vv, a, pdf, v, s))
    assert fnp.isfinite(rho_source) and fnp.all(v >= 1e-10) and fnp.all(s > 0)
    delta_k4 = rays - rho_source * vv * vv
    f = (a * a - 1) * pdf / (v * s)
    delta = delta_k4 * f / 24
    assert fnp.all(fnp.isfinite(delta))
    return delta


'''

SOURCE_INSERT = """        if _TERMINAL_TREE_ENABLED and l == 14:
            assert depth == 16 and metric == 'covariance'
            _tree_rays, _tree_rho = _terminal_tree_source(Cpre, C, aux, W[15])
"""
FINAL_INSERT = """        if _TERMINAL_TREE_ENABLED and l == 15:
            m = m + _terminal_tree_mean_delta(_tree_rays, _tree_rho, vv, aux)
"""
SOURCE_ANCHOR = "        host = m.copy()\n"
FINAL_ANCHOR = "        means.append(m.copy())\n"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")


def authenticate(source_sha256, protocol_sha256):
    require(protocol_sha256 == PROTOCOL_SHA, "protocol digest is not the fixed protocol")
    pins = {str(Path(__file__).resolve()): source_sha256,
            str(PARENT): PARENT_SHA, str(PROTOCOL): PROTOCOL_SHA,
            str(DESIGN): DESIGN_SHA}
    for path, digest in pins.items():
        require(isinstance(digest, str) and len(digest) == 64 and sha(path) == digest,
                "source binding mismatch: " + path)
    return pins


def function_span(source):
    nodes = [x for x in ast.parse(source).body
             if isinstance(x, ast.FunctionDef) and x.name == "_predict_binary"]
    require(len(nodes) >= 2, "expected shadowed parent definitions")
    node = nodes[-1]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1]))
    end = sum(map(len, lines[:node.end_lineno]))
    return start, end, source[start:end]


def prelude(enabled):
    require(type(enabled) is bool, "enable must be a literal bool")
    return "_TERMINAL_TREE_ENABLED = " + repr(enabled) + "\n\n\n" + HELPERS


def restore(candidate, enabled):
    start, end, body = function_span(candidate)
    require(body.count(SOURCE_INSERT) == body.count(FINAL_INSERT) == 1,
            "candidate must have exactly two active-function inserts")
    body = body.replace(SOURCE_INSERT, "", 1).replace(FINAL_INSERT, "", 1)
    restored = candidate[:start] + body + candidate[end:]
    bundle = prelude(enabled)
    require(restored.count(bundle) == 1, "helper/enable bundle identity")
    return restored.replace(bundle, "", 1)


def candidate_source(parent, enabled):
    require(text_sha(parent) == PARENT_SHA, "parent source bytes")
    require("_terminal_tree_" not in parent and "_TERMINAL_TREE_ENABLED" not in parent,
            "parent unexpectedly owns terminal tree names")
    start, end, body = function_span(parent)
    require(body.count(SOURCE_ANCHOR) == body.count(FINAL_ANCHOR) == 1,
            "active recurrence insertion anchors")
    body = body.replace(SOURCE_ANCHOR, SOURCE_ANCHOR + SOURCE_INSERT, 1)
    body = body.replace(FINAL_ANCHOR, FINAL_INSERT + FINAL_ANCHOR, 1)
    candidate = parent[:start] + prelude(enabled) + body + parent[end:]
    ast.parse(candidate)
    require(restore(candidate, enabled) == parent, "exact source inverse failed")
    return candidate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--source-sha256")
    parser.add_argument("--protocol-sha256")
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return
    pins = authenticate(args.source_sha256, args.protocol_sha256)
    parent = PARENT.read_bytes().decode("utf-8")
    sources = {"enabled": candidate_source(parent, True),
               "zero": candidate_source(parent, False)}
    require(sources["enabled"].replace("_TERMINAL_TREE_ENABLED = True\n",
                                     "_TERMINAL_TREE_ENABLED = False\n", 1) == sources["zero"],
            "zero differs by more than the enable literal")
    OUT.mkdir(exist_ok=False)
    result = {"schema": "terminal_full_fourth_tree_build_v1", "passed": False,
              "completed": False, "source_bindings": pins, "artifacts": {},
              "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "parent_sha256": PARENT_SHA, "helper_sha256": text_sha(HELPERS),
              "natural_coefficient": 1, "source_layer": 14, "readout_layer": 15,
              "full_unmasked_legs": True, "trees": 16, "paths": 12, "stars": 4,
              "three_edge_sector_not_full_cumulant": True,
              "truth_accessed": False, "numerical_imports": False,
              "numerical_run_or_score_admitted": False}
    try:
        result["candidates"] = {}
        for role, filename in (("enabled", "estimator.py"), ("zero", "zero_estimator.py")):
            path = OUT / filename
            with path.open("xb") as f:
                f.write(sources[role].encode("utf-8"))
            digest = sha(path)
            require(digest == text_sha(sources[role]), "written candidate bytes")
            require(restore(path.read_bytes().decode("utf-8"), role == "enabled") == parent,
                    "written candidate inverse")
            result["artifacts"][str(path)] = digest
            result["candidates"][role] = {"path": str(path), "sha256": digest,
                                          "enabled": role == "enabled"}
        for path, digest in pins.items():
            require(sha(path) == digest, "source changed during build: " + path)
        result.update(passed=True, completed=True, hashes_unchanged=True,
                      sources_unchanged=True, exact_source_inverse=True,
                      only_enable_literal_differs=True)
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    write(OUT / "build.json", result)
    write(OUT / "completion.json", {"passed": result["passed"],
          "completed": result["completed"], "result_sha256": sha(OUT / "build.json")})
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
