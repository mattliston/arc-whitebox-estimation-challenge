"""Inert, read-only adjudication of one frozen synthetic certificate.

No NumPy, estimator, GNN, feature-module import, finite difference, or rerun.
Only authenticated source text and JSON scalar receipts are read.  The
original result remains FAILED and is never rewritten.  The additive verdict
corrects a zero-curvature classification omission, not a measured derivative:

  x[i,3]  = mu[i]/sigma[i]
  x[i,26] = sum_j rho[i,j]*mu[j]/sigma[j]/n.

With the certificate's fixed C/sigma/rho, both second derivatives with
respect to mu[i] vanish exactly, including below the fixed sigma floor.
They are affine, not constant, so their FIRST-derivative gates remain intact.
The frozen feature source returns exact zero second jets for both columns.

Apply the already-declared 1e-8 dimensionless zero-curvature safeguard to
ALL eight second-derivative records for these columns, not only failures.
Every original scalar comparison and every other gate is checked unchanged.
This qualifies only the recorded synthetic continuous-extension test; it
does not qualify current-network values, served FP32 derivatives, a sampler,
an accuracy/score improvement, or any later experimental stage.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback


HERE = Path(__file__).resolve().parent
NEW = HERE.parent
PINS = {
    "tiny_result.json": "f218295f1013a6d95b09db7292a77454ad354f8aff2fc57150de5fe9e9845851",
    "tiny_started.json": "2e8f38d876f43eca8b874af841ce35ba10488361415fb35a202acb6fa7006033",
    "certify_tiny.py": "bd8594f6a6138f75996f9ac764e062c684a30a68a18de5ec91f6d224a31e4ab7",
    "PROTOCOL.md": "4a3c272b23bcafb4c36167e272449d18b95f4a452da48b49072440b862984534",
    "feature_second_jet_reference.py": "dd4484fc3baaefb783ff54497b43ff801214e0583cd16d28a9d9b4c212cb218f",
    "gnn_second_jet_reference.py": "2ce74c478040d456078091751520d3a845b2fc8ca47acd6269d8fd8d255d8b21",
    "../unused_materializations_v1/estimator.py": "409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f",
}
SOURCE_NAMES = set(PINS) - {"tiny_result.json", "tiny_started.json"}
FIXED_FEATURES = {0, 1, 2, 4, 10, 12, 13, 14, 15, 16, 17, 18, 19, 22, 23, 24, 25}
AFFINE_FEATURES = {3, 26}
ZERO_SECOND_ABSOLUTE = 1e-8
VERSIONS = dict(numpy="2.4.6", flopscope="0.12.0", whestbench="0.16.0", threadpoolctl="3.6.0")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authenticated_sources():
    bindings = {}
    for name, expected in PINS.items():
        path = (HERE / name).resolve()
        actual = digest(path)
        require(actual == expected, "Hash mismatch: " + name)
        bindings[name] = dict(path=str(path), sha256=actual)
    return bindings


def load_json(name):
    def reject_constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads((HERE / name).read_text(), parse_constant=reject_constant)


def prove_affine_source():
    """Structural checks only; never compile or evaluate the feature source."""
    tree = ast.parse((HERE / "feature_second_jet_reference.py").read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}

    def assignment(function, name):
        found = [node.value for node in functions[function].body
                 if isinstance(node, ast.Assign) and len(node.targets) == 1
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]
        require(len(found) == 1, "Unexpected assignment ownership: " + name)
        return found[0]

    def same(node, expression):
        expected = ast.parse(expression, mode="eval").body
        require(ast.dump(node) == ast.dump(expected), "Affine proof source mismatch: " + expression)

    same(assignment("raw_jets", "v"), "np.diag(pre_cov).copy()")
    same(assignment("raw_jets", "sv"), "np.maximum(v, 1e-10)")
    same(assignment("raw_jets", "sigma"), "np.sqrt(sv)")
    same(assignment("raw_jets", "shift"),
         "Jet(np.asarray(mu, np.float64), np.ones_like(mu), np.zeros_like(mu))")
    same(assignment("raw_jets", "alpha"), "shift / sigma")
    same(assignment("features", "n"), "len(mu)")
    same(assignment("features", "sigma"), "aux['sigma']")
    same(assignment("features", "rho"), "pre_cov / np.outer(sigma, sigma)")
    same(assignment("features", "gate_message"),
         "Jet(rho @ aux['alpha'].value / n, np.diag(rho) / (n*sigma), np.zeros(n))")
    values = assignment("features", "vals")
    require(isinstance(values, ast.Tuple) and len(values.elts) == 30, "Feature count changed")
    same(values.elts[3], "aux['alpha']")
    same(values.elts[26], "gate_message")
    protocol = (HERE / "PROTOCOL.md").read_text()
    require("absolute safeguards1e-10/1e-8 for genuinely zero" in protocol,
            "Original zero-derivative safeguard not found")


def finite_nonnegative(value, name):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0, name)
    return value


def comparison(record, tolerance, absolute):
    """Recheck the ORIGINAL reported scalar guard, not its numerical inputs."""
    required = {"error_norm", "reference_norm", "relative_error", "allowed_absolute", "passed"}
    require(set(record) in (required, required | {"feature"}), "Unexpected comparison schema")
    error = finite_nonnegative(record["error_norm"], "Invalid error norm")
    reference = finite_nonnegative(record["reference_norm"], "Invalid reference norm")
    relative = finite_nonnegative(record["relative_error"], "Invalid relative error")
    require(record["allowed_absolute"] == absolute, "Original absolute allowance mismatch")
    require(relative == error / max(reference, 1e-300), "Relative error arithmetic mismatch")
    passed = error <= absolute + tolerance * reference
    require(type(record["passed"]) is bool and record["passed"] is passed, "Original guard mismatch")
    return passed


def zero_counts(values, label):
    require(isinstance(values, dict) and bool(values), label + " missing")
    require(all(type(value) is int and value == 0 for value in values.values()), label + " nonzero")


def adjudicate():
    bindings = authenticated_sources()
    prove_affine_source()
    result, started = load_json("tiny_result.json"), load_json("tiny_started.json")
    require(result["started"] == started, "Started receipt mismatch")
    expected_sources = {bindings[name]["path"]: PINS[name] for name in SOURCE_NAMES}
    require(started["source_sha256"] == expected_sources, "Sealed source map mismatch")
    require(started["versions"] == VERSIONS, "Version receipt mismatch")
    environment = (NEW.parent / "data_gate/.venv").resolve()
    require(started["prefix"] == str(environment), "OFFPY prefix mismatch")
    require(started["executable"] == str(environment / "bin/python"), "OFFPY executable mismatch")
    require(started["nice"] >= 10 and started["watchdog_s"] == 60, "Launch limits mismatch")
    require(started["peak_rss_limit_bytes"] == 1_000_000_000, "RSS limit mismatch")
    require(started["blas_env"] == {key: "1" for key in
            ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}, "BLAS controls mismatch")
    for key in ("targets_opened", "mlp_or_payload_arrays_opened", "score_evidence"):
        require(started[key] is False, "Unexpected scope: " + key)
    require("error" not in result, "Certificate exception present")
    require(result["passed"] is False, "Original failure must remain preserved")
    require(result["resources_passed"] is True, "Resources failed")
    wall = finite_nonnegative(result["wall_s"], "Invalid wall time")
    rss = finite_nonnegative(result["peak_rss_bytes"], "Invalid RSS")
    require(0 < wall < 60 and 0 < rss < 1_000_000_000, "Resource limits exceeded")
    require(datetime.fromisoformat(result["completed_utc"]) >= datetime.fromisoformat(started["utc"]),
            "Completion precedes start")
    require(result["clip_convention_passed"] is True and result["gnn_clip_convention_passed"] is True,
            "Clip convention failed")
    require(result["fixed_messages_byte_exact"] == [True] * 6, "Fixed messages not byte-exact")
    require(isinstance(result["threadpools"], list), "Missing threadpool record")
    require(all(pool["num_threads"] == 1 for pool in result["threadpools"]
                if pool.get("user_api") == "blas"), "Observed BLAS pool mismatch")
    require([record["case"] for record in result["records"]] == ["regular", "below_sigma_floor"],
            "Fixture cases changed")

    adjudications, failures = [], []
    for record in result["records"]:
        case = record["case"]
        require(record["passed"] is False, "Unexpected original case verdict")
        zero_counts(record["boundaries"], "Feature boundaries")
        diagnostics = record["diagnostics"]
        require(diagnostics["all_finite"] is True and diagnostics["zero_clip_endpoint_derivatives"] is True,
                "Jet diagnostics failed")
        require(diagnostics["nonpositive_head_scales"] == 0, "Nonpositive scales")
        for key in ("clip_endpoint_counts", "nonfinite", "message_preparation_nonfinite", "head_preparation_nonfinite"):
            zero_counts(diagnostics[key], key)
        require(set(record["primal"]) == {"features", "covariance", "mean", "gnn_delta", "gnn_state"},
                "Primal comparisons missing")
        require(all(comparison(item, 2e-12, 0.) for item in record["primal"].values()), "Primal failure")
        require([item["epsilon_factor"] for item in record["differences"]] == [.001, .0005],
                "Finite-difference epsilons changed")
        for difference in record["differences"]:
            require(difference["units"] == [0, 5, 10, 15], "Finite-difference units changed")
            epsilon = difference["epsilon_factor"]
            for order, tolerance, safeguard in ((1, 2e-6, 1e-10), (2, 5e-4, 1e-8)):
                require(comparison(difference["gnn_first" if order == 1 else "gnn_second"], tolerance, 0.),
                        "GNN derivative failure")
                group = difference["feature_first" if order == 1 else "feature_second"]
                require([item["feature"] for item in group["per_feature"]] == list(range(30)),
                        "Per-feature comparisons missing or reordered")
                original_passes = []
                for item in group["per_feature"]:
                    feature = item["feature"]
                    allowance = safeguard if feature in FIXED_FEATURES else 0.
                    original_pass = comparison(item, tolerance, allowance)
                    original_passes.append(original_pass)
                    if not original_pass:
                        failures.append((case, epsilon, order, feature))
                    if order == 2 and feature in AFFINE_FEATURES:
                        require(item["error_norm"] == item["reference_norm"], "Expected exact-zero analytic jet")
                        require(item["error_norm"] <= ZERO_SECOND_ABSOLUTE, "Affine zero-curvature safeguard failed")
                        adjudications.append(dict(case=case, epsilon_factor=epsilon, feature=feature,
                            original_passed=original_pass, error_norm=item["error_norm"],
                            absolute_safeguard=ZERO_SECOND_ABSOLUTE, adjudicated_passed=True))
                    else:
                        require(original_pass, "Failure outside the two affine second derivatives")
                require(group["passed"] is all(original_passes), "Feature summary verdict mismatch")

    expected_failures = [("regular", epsilon, 2, feature) for epsilon in (.001, .0005) for feature in (3, 26)]
    expected_failures += [("below_sigma_floor", epsilon, 2, 3) for epsilon in (.001, .0005)]
    require(failures == expected_failures and len(adjudications) == 8, "Failure classification differs from frozen receipt")
    require(authenticated_sources() == bindings, "Pinned inputs changed during adjudication")
    return dict(schema="current_response_affine_zero_curvature_adjudication_v1",
        utc=datetime.now(timezone.utc).isoformat(), adjudicated_passed=True,
        original_certificate_passed=False, original_result_unchanged=True,
        numerical_rerun=False, file_writes=False, arrays_or_targets_decoded=False,
        source_bindings=bindings, adjudicator_sha256=digest(Path(__file__)),
        corrected_classification="Affine columns 3 and 26 have exactly zero second derivative, not zero first derivative.",
        rule="Apply the predeclared 1e-8 dimensionless zero-curvature bound uniformly to all eight affine second-jet records.",
        adjudications=adjudications, original_failed_comparisons=len(failures),
        maximum_adjudicated_error_norm=max(item["error_norm"] for item in adjudications),
        other_original_gates_verified=True, resources=dict(wall_s=wall, peak_rss_bytes=rss),
        blas_attestation=result["blas_attestation"],
        qualification="Additive synthetic Stage0 guard-classification adjudication only. Original literal result remains failed. No current-network value, served-FP32 derivative, accuracy/score claim, or later-stage admission.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adjudicate", action="store_true")
    args = parser.parse_args()
    if not args.adjudicate:
        parser.error("Explicit --adjudicate required; inert by default")
    try:
        verdict = adjudicate()
    except Exception:
        print(json.dumps(dict(adjudicated_passed=False, numerical_rerun=False,
                              error=traceback.format_exc()), indent=2, allow_nan=False))
        return 1
    print(json.dumps(verdict, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
