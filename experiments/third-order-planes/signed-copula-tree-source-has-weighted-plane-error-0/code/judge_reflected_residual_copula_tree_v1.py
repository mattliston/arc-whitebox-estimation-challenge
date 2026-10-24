"""One inert-by-default rank35 post12 copula-tree qualification.

The fixed protocol owns the numerical spelling and loss. No fits, repairs,
alternate orders, or rollout are implemented. Existing pre13 K21 is decoded
only after the causal predictor artifact and its seal have been written.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
CAPTURE = HERE / "rows_reflected_residual_copula_tree_v1/row_0035"
OLD = HERE / "rows_two_memory_teacher_action_v2/row_0035"
OUT = HERE / "reflected_residual_copula_tree_information_v1"
PROTOCOL = "REFLECTED_RESIDUAL_COPULA_TREE_PROTOCOL_V1.md"
EXPECTED = {
    PROTOCOL: "a0c4fa413bcd607a7df2f748c86c359489b88c84672ce97314d90dec448c0c50",
    "certify_reflected_relu_marginal_v1.py": "f77b5ac71b08a172ec5cf481d4a0a8ea9387f9039cacf276aa14886e9acfa4a1",
    "reflected_relu_marginal_certificate_v1.json": "827eb01d67affe38627e5d2e816ee87216a927cac388609f10a4a46575d15fcf",
    "certify_reflected_copula_tree_action_v1.py": "42b1cb038d10bdab94aeb15bfeadbcae5cf26ab9d1e92dd6700d5157f7b8874b",
    "reflected_copula_tree_action_certificate_v1.json": "083d38baf0fceca2f32680c45f6ec9fb2a218d150a41c13bdb51b6964e84b537",
    "two_memory_capture_replay_v2_qualification.json": "af8c2b0cb66b6730af9d93b5aa0e56efcc97ba3e825093c4926f4bbd3805f328",
    "capture_reflected_residual_copula_trace_v1.py": "701f01d66689f795cec8d4d8aa34907531f6a31a4825d4c5783abc16a05e349a",
    "REFLECTED_RESIDUAL_COPULA_TRACE_CAPTURE_OWNERSHIP_V1.md": "149bbfaace9bb920a2cadf490a2a8479d736729714dbba7e05726488902eff12",
    "REFLECTED_RESIDUAL_COPULA_ROOT_ACCEPTANCE_V1.md": "a0c25bea1700f484fcab872f2c79c4f4ec2d9cbcdea9bc8d9960d4d2b365e8f2",
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def execute():
    require("/anaconda3/" not in sys.executable, "NumPy-only repo interpreter required")
    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules),
            "Fresh NumPy-only process required")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Launch at nice >=10")
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    require(not OUT.exists(), "Refusing to overwrite an existing experiment")
    paths = {name: HERE / name for name in EXPECTED}
    paths.update({
        "judge_source": Path(__file__).resolve(),
        "capture_receipt": CAPTURE / "result.json",
        "causal_inputs": CAPTURE / "causal_post12_pre13.npz",
        "old_receipt": OLD / "result.json",
        "old_post12": OLD / "post_12.npz",
        "sealed_pre13_loss": OLD / "pre_13.npz",
        "weights": HERE / "rows_current_transport_shadow_v2/row_0035/weights.npy",
    })
    capture = json.loads(paths["capture_receipt"].read_text())
    require(capture["passed"] and capture["rank"] == 35,
            "The prescribed capture has not qualified")
    require(capture["peak_rss_bytes"] <= 6 * (1 << 30), "Capture exceeded 6GiB")
    require(sha(paths["causal_inputs"]) ==
            capture["artifacts"]["causal_post12_pre13.npz"]["sha256"],
            "Causal artifact does not match the qualified capture")
    ownership = capture["trace_ownership"]
    require(ownership["current_action_loss_in_causal_artifact"] is False and
            ownership["inverse_evaluated"] is False and ownership["predictor_evaluated"] is False,
            "Capture changed causal ownership")
    require(sha(paths["old_receipt"]) == ownership["old_v2_receipt_sha256"],
            "Original receipt no longer matches qualified recapture")
    for name, digest in ownership["checked_files"].items():
        p = HERE / name
        require(sha(p) == digest, f"Authenticated original artifact changed: {name}")
        paths["authenticated_original/" + name] = p
    # Every source named by the capture is transitively bound before any inverse.
    for name, digest in capture["source_hashes"].items():
        p = HERE / name
        require(sha(p) == digest, f"Capture source changed: {name}")
        paths["capture_source/" + name] = p
    original = json.loads(paths["old_receipt"].read_text())
    require(original["passed"] and original["rank"] == 35 and
            original["target_access"] is False and original["peak_rss_bytes"] <= 6 * (1 << 30),
            "Original v2 capture did not qualify")
    for name in ("post_12.npz", "pre_13.npz"):
        require(sha(OLD / name) == original["artifacts"][name]["sha256"],
                f"Original v2 artifact changed: {name}")
    require(sha(paths["weights"]) == original["input_hashes"]["prefix/weights.npy"],
            "Different bank weights")
    hashes = {k: sha(v) for k, v in paths.items()}
    for name, expected in EXPECTED.items():
        require(hashes[name] == expected, f"Frozen prerequisite changed: {name}")
    OUT.mkdir(exist_ok=False)
    started = time.perf_counter()
    result = dict(passed=False, rank=35, post_layer=12, next_layer=13,
                  scope="One fixed numerical/domain and next-plane teacher-fidelity test",
                  raw_score_claim=False, target_access=False,
                  numerical_qualified=False, stage="started", hashes=hashes,
                  paths={k: str(v) for k, v in paths.items()})
    write_json(OUT / "started.json", result)
    try:
        import numpy as np
        import scipy
        from scipy.special import ndtr
        from numpy.polynomial.legendre import leggauss
        from threadpoolctl import threadpool_info, threadpool_limits
        from certify_reflected_relu_marginal_v1 import moments, invert, skew

        def finite(a):
            return bool(np.isfinite(a).all())

        def norm(a):
            return float(np.linalg.norm(a))

        def save_npz(name, **arrays):
            require(all(finite(v) for v in arrays.values()), f"Nonfinite artifact {name}")
            with (OUT / name).open("xb") as stream:
                np.savez(stream, **arrays)

        def resource_gate():
            require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 2 * (1 << 30),
                    "Judge exceeded 2GiB RSS")

        with threadpool_limits(limits=2):
            with np.load(paths["causal_inputs"], allow_pickle=False) as z:
                fields = {
                    "post_C", "post_B", "post_b", "post_m", "post_variance", "post_K21",
                    "post_alive", "post_radial", "pre13_Bpre", "pre13_bpre", "pre13_alive",
                    "pre13_alpha", "pre13_sigma",
                }
                require(set(z.files) == fields, "Unexpected causal contract / potential loss field")
                state = {k: z[k] for k in fields}
            require(all(finite(v) for v in state.values()), "Nonfinite causal state")
            C, B, b, S = (state[k].astype(np.float64) for k in
                          ("post_C", "post_B", "post_b", "post_K21"))
            alive = state["post_alive"]
            require(alive.shape == (1024,) and alive.dtype == np.bool_, "Bad actual live mask")
            require(C.shape == B.shape == S.shape == (1024, 1024) and b.shape == (1024,),
                    "Not the fixed Phase2 state")
            variance = np.diag(C).copy()
            require(np.array_equal(variance, state["post_variance"]), "Post variance ownership mismatch")
            expected_B = C.copy()
            np.fill_diagonal(expected_B, variance * alive)
            require(np.array_equal(B, expected_B), "Trace metric ownership mismatch")
            require(np.count_nonzero(b[~alive]) == 0, "Cold trace vector changed")
            indices = np.flatnonzero(alive)
            require(len(indices) > 1 and np.all(variance[indices] > 0), "Invalid live variance")
            live_C = C[np.ix_(indices, indices)]
            aux_C = (live_C + live_C.T) * 0.5
            result["stage"] = "auxiliary_covariance"
            result["state"] = dict(live_count=len(indices), cold_count=int((~alive).sum()),
                live_variance_min=float(variance[indices].min()),
                cold_negative_variance_count=int((variance[~alive] < 0).sum()),
                covariance_asymmetry_relative=norm(live_C-live_C.T)/max(norm(live_C), 1e-300),
                auxiliary_change_relative=norm(aux_C-live_C)/max(norm(live_C), 1e-300))
            aux_chol = np.linalg.cholesky(aux_C)
            result["state"]["auxiliary_cholesky_min_diagonal"] = float(np.diag(aux_chol).min())
            del aux_chol, expected_B
            trace = np.diag(B)[:, None] * b[None, :] + 2 * B * b[:, None]
            E21 = S - trace
            sigma = np.sqrt(variance[indices])
            gamma = np.diag(E21)[indices] / sigma**3
            require(finite(gamma), "Nonfinite residual marginal skew")
            gaussian = gamma == 0
            eta = np.where(gamma < 0, -1., 1.)
            alpha = np.zeros(len(indices))
            tau, a, q = sigma.copy(), sigma.copy(), np.zeros(len(indices))
            relu_m = np.zeros(len(indices))
            result["stage"] = "marginal_inverse"
            endpoint_low, endpoint_high = skew(12.), skew(-8.)
            bad = (~gaussian) & ((np.abs(gamma) < endpoint_low) | (np.abs(gamma) > endpoint_high))
            result["marginal"] = dict(gamma_min=float(gamma.min()), gamma_max=float(gamma.max()),
                reflected_count=int((gamma < 0).sum()), gaussian_count=int(gaussian.sum()),
                bracket_failure_count=int(bad.sum()), skew_endpoint_range=[endpoint_low, endpoint_high])
            require(not bad.any(), "Residual skew outside the fixed [-8,12] marginal bracket")
            skew_error, variance_error = 0., 0.
            for i in np.flatnonzero(~gaussian):
                alpha[i] = invert(abs(float(gamma[i])))
                mi, vi, ki = moments(float(alpha[i]))
                tau[i] = sigma[i] / math.sqrt(vi)
                a[i] = tau[i] * float(ndtr(alpha[i]))
                q[i] = eta[i] * tau[i] * math.exp(-0.5*alpha[i]**2) / math.sqrt(2*math.pi)
                relu_m[i] = mi
                skew_error = max(skew_error, abs(ki/vi**1.5-abs(gamma[i]))/abs(gamma[i]))
                variance_error = max(variance_error, abs(tau[i]**2*vi-sigma[i]**2)/sigma[i]**2)
            result["marginal"].update(alpha_min=float(alpha.min()), alpha_max=float(alpha.max()),
                max_relative_skew_error=float(skew_error), max_relative_variance_error=float(variance_error))
            require(skew_error <= 1e-8 and variance_error <= 1e-10 and finite(tau+a+q),
                    "Marginal inversion numerical gate failed")
            save_npz("marginal_parameters.npz", live_indices=indices, gamma=gamma, gaussian=gaussian,
                     alpha=alpha, eta=eta, tau=tau, sigma=sigma, a=a, q=q)

            # Hermite coefficients are normalized by exact physical sigma and sqrt(k!).
            h = np.zeros((len(indices), 13))
            h[:, 1] = a/sigma
            he_prev, he_current = np.zeros(len(indices)), np.ones(len(indices))
            density = np.exp(-0.5*alpha*alpha)/math.sqrt(2*math.pi)
            for k in range(2, 13):
                if k > 2:
                    he_prev, he_current = he_current, -alpha*he_current-(k-3)*he_prev
                h[:, k] = tau/sigma * eta**(k+1) * density * he_current / math.sqrt(math.factorial(k))
            h[gaussian, 2:] = 0
            require(finite(h), "Nonfinite normalized Hermite responses")
            ii, jj = np.triu_indices(len(indices), 1)
            target = aux_C[ii, jj] / (sigma[ii]*sigma[jj])
            coefficients = (h[ii, 1:] * h[jj, 1:]).T.copy()

            def polynomial(r):
                value = coefficients[-1].copy()
                for k in range(10, -1, -1):
                    value = coefficients[k] + r*value
                return r*value

            result["stage"] = "pair_inverse"
            low_value = polynomial(np.full(len(ii), -1.))
            high_value = polynomial(np.ones(len(ii)))
            polynomial_bad = (low_value > target) | (high_value < target)
            result["pair"] = dict(pair_count=len(ii),
                polynomial_bracket_failure_count=int(polynomial_bad.sum()),
                hermite12_variance_fraction_min=float(np.sum(h*h, axis=1).min()),
                hermite12_variance_fraction_max=float(np.sum(h*h, axis=1).max()))
            require(not polynomial_bad.any() and finite(low_value+high_value),
                    "Degree12 polynomial endpoint bracket failed (not an exact-copula impossibility)")
            linear = gaussian[ii] | gaussian[jj]
            lo, hi = np.full(len(ii), -1.), np.ones(len(ii))
            for _ in range(24):
                mid = (lo+hi)*0.5
                below = polynomial(mid) < target
                lo = np.where(below, mid, lo)
                hi = np.where(below, hi, mid)
            r = (lo+hi)*0.5
            r[linear] = target[linear] / coefficients[0, linear]
            poly_error = polynomial(r)-target
            result["pair"]["max_standardized_polynomial_residual"] = float(np.max(np.abs(poly_error)))
            result["pair"]["latent_abs_offdiagonal_max"] = float(np.max(np.abs(r)))
            require(finite(r) and np.all(np.abs(r) < 1) and np.max(np.abs(poly_error)) <= 1e-6,
                    "Pair inverse range/residual failed")
            del coefficients, low_value, high_value, lo, hi, mid, poly_error
            resource_gate()

            # Independent Price integral and exact 1-D endpoints; no root retries.
            def price_covariance(order):
                nodes, weights = leggauss(order)
                nodes, weights = (nodes+1)*0.5, weights*0.5
                answer = np.empty(len(ii))
                for begin in range(0, len(ii), 16384):
                    end = min(begin+16384, len(ii))
                    i, j = ii[begin:end], jj[begin:end]
                    ri = eta[i]*eta[j]*r[begin:end]
                    ai, aj = alpha[i], alpha[j]
                    integral = np.zeros(end-begin)
                    for u, weight in zip(nodes, weights):
                        t = ri*u
                        denominator = 1-t*t
                        integral += weight*(1-u)*np.exp(-(ai*ai-2*t*ai*aj+aj*aj)/(2*denominator))/np.sqrt(denominator)
                    ordinary = ri*ndtr(ai)*ndtr(aj)+ri*ri*integral/(2*math.pi)
                    answer[begin:end] = eta[i]*eta[j]*(tau[i]/sigma[i])*(tau[j]/sigma[j])*ordinary
                answer[linear] = r[linear]*h[ii[linear], 1]*h[jj[linear], 1]
                return answer

            ordinary_alpha_i, ordinary_alpha_j = alpha[ii], alpha[jj]
            amin = np.minimum(ordinary_alpha_i, ordinary_alpha_j)
            amax = np.maximum(ordinary_alpha_i, ordinary_alpha_j)
            mean_product = relu_m[ii]*relu_m[jj]
            endpoint_plus = (1+ordinary_alpha_i*ordinary_alpha_j)*ndtr(amin) + amax*np.exp(-0.5*amin*amin)/math.sqrt(2*math.pi)-mean_product
            probability = ndtr(ordinary_alpha_j)-ndtr(-ordinary_alpha_i)
            product_minus = ((ordinary_alpha_i*ordinary_alpha_j-1)*probability +
                ordinary_alpha_j*density[ii]+ordinary_alpha_i*density[jj])
            product_minus = np.where(ordinary_alpha_i+ordinary_alpha_j > 0, product_minus, 0.)
            endpoint_minus = product_minus-mean_product
            sign = eta[ii]*eta[jj]
            scale = (tau[ii]/sigma[ii])*(tau[jj]/sigma[jj])
            exact_lo = sign*scale*np.where(sign > 0, endpoint_minus, endpoint_plus)
            exact_hi = sign*scale*np.where(sign > 0, endpoint_plus, endpoint_minus)
            exact_lo[linear] = -h[ii[linear], 1]*h[jj[linear], 1]
            exact_hi[linear] = -exact_lo[linear]
            exact_bad = (target < exact_lo-1e-6) | (target > exact_hi+1e-6)
            result["pair"]["exact_endpoint_bracket_failure_count"] = int(exact_bad.sum())
            require(not exact_bad.any() and finite(exact_lo+exact_hi), "Exact pair endpoint feasibility failed")
            result["stage"] = "independent_pair_quadrature"
            c32, c64 = price_covariance(32), price_covariance(64)
            quad_error = c64-c32
            model_error = c64-target
            result["pair"].update(
                max_standardized_quadrature_discrepancy=float(np.max(np.abs(quad_error))),
                max_standardized_exact_covariance_residual=float(np.max(np.abs(model_error))),
                standardized_covariance_residual_relative_frobenius=norm(model_error)/max(norm(target), 1e-300),
                physical_covariance_residual_relative_frobenius=norm(model_error*sigma[ii]*sigma[jj])/max(norm(aux_C[ii,jj]),1e-300),
                quadrature_failure_count=int((np.abs(quad_error)>1e-8).sum()),
                covariance_residual_failure_count=int((np.abs(model_error)>1e-6).sum()))
            require(finite(c32+c64) and np.max(np.abs(quad_error)) <= 1e-8 and
                    np.max(np.abs(model_error)) <= 1e-6, "Independent exact-covariance qualification failed")
            R = np.eye(len(indices))
            R[ii, jj] = r
            R[jj, ii] = r
            require(np.array_equal(R, R.T) and np.array_equal(np.diag(R), np.ones(len(indices))),
                    "Latent matrix assembly changed diagonal/symmetry")
            result["stage"] = "latent_positive_definiteness"
            chol = np.linalg.cholesky(R)
            result["pair"]["latent_cholesky_min_diagonal"] = float(np.diag(chol).min())
            result["numerical_qualified"] = True
            del chol, c32, c64, quad_error, model_error

            # Exact FP64 one-action diagnostic, not a deployed FP32 cost claim.
            U = np.zeros((1024, 1024))
            U[np.ix_(indices, indices)] = a[:, None]*R
            full_q, full_a = np.zeros(1024), np.zeros(1024)
            full_q[indices], full_a[indices] = q, a
            tree21 = (U*U)*full_q[None, :] + 2*(full_q*full_a)[:, None]*U.T
            A21 = E21-tree21
            encoded = 3*A21.T-2*np.diag(np.diag(A21))
            restored = tree21+(encoded.T+2*np.diag(np.diag(encoded)))/3+trace
            restoration = restored-S
            result["stage"] = "tree_and_star_action"
            result["restoration"] = dict(relative_frobenius=norm(restoration)/max(norm(S), 1e-300),
                max_entry_normalized=float(np.max(np.abs(restoration)))/max(float(np.max(np.abs(S))),1e-300))
            require(result["restoration"]["relative_frobenius"] <= 1e-10, "Current repeated-plane restoration failed")
            weights = np.load(paths["weights"], mmap_mode="r", allow_pickle=False)
            require(weights.shape == (16,1024,1024) and weights.dtype == np.float32, "Wrong Phase2 weights")
            w = weights[13].astype(np.float64)
            rows = np.arange(0, 1024, 32)
            X, WT = w.T@U, w.T
            tree_next = (X[rows]*X[rows])@(WT*full_q).T + 2*(X[rows]*WT[rows])@(X*full_q).T
            star_E = A21-(2/3)*np.diag(np.diag(A21))
            star_Q = star_E@w
            star_next = (w[:,rows]*w[:,rows]).T@star_Q + 2*(w[:,rows]*star_Q[:,rows]).T@w
            nontrace = tree_next+star_next
            diagonal = nontrace[np.arange(len(rows)), rows].copy()
            pre_alive = state["pre13_alive"]
            require(pre_alive.shape == (1024,) and pre_alive.dtype == np.bool_, "Bad pre13 support")
            nontrace *= pre_alive[rows, None]*pre_alive[None, :]
            nontrace[np.arange(len(rows)), rows] = diagonal
            Bpre, bpre = state["pre13_Bpre"], state["pre13_bpre"]
            next_trace = np.diag(Bpre)[rows,None]*bpre[None,:]+2*Bpre[rows]*bpre[rows,None]
            prediction = nontrace+next_trace
            pre_alpha, pre_sigma = state["pre13_alpha"], state["pre13_sigma"]
            require(np.all(pre_sigma > 0), "Invalid pre13 sigma")
            consumer_weight = 0.5*np.exp(-0.5*pre_alpha[rows]**2)[:,None]/(math.sqrt(2*math.pi)*pre_sigma[rows,None])*ndtr(pre_alpha)[None,:]
            save_npz("sealed_predictor.npz", query_rows=rows, predicted_K21=prediction,
                     consumer_weight=consumer_weight, U=U, q=full_q, residual_star_plane=A21,
                     latent_R=R, live_indices=indices)
            predictor_hash = sha(OUT/"sealed_predictor.npz")
            write_json(OUT/"predictor_seal.json", dict(predictor_sha256=predictor_hash,
                sealed_utc=datetime.now(timezone.utc).isoformat(), target_access=False,
                hashes=hashes, numerical_qualified=True, result_before_teacher=result))
            resource_gate()
            result["predictor_sha256"] = predictor_hash
            result["stage"] = "sole_teacher_fidelity"
            result["target_access"] = True
            with np.load(paths["sealed_pre13_loss"], allow_pickle=False) as z:
                teacher = z["pre_K21_loss_only"][rows].astype(np.float64)
            delta = prediction-teacher
            denominator = norm(consumer_weight*teacher)
            require(math.isfinite(denominator) and denominator > 0, "Invalid teacher denominator")
            fidelity = norm(consumer_weight*delta)/denominator
            result["fidelity"] = dict(weighted_relative=fidelity,
                unweighted_relative=norm(delta)/max(norm(teacher),1e-300),
                diagonal_relative=norm(delta[np.arange(len(rows)),rows])/max(norm(teacher[np.arange(len(rows)),rows]),1e-300),
                weighted_teacher_norm=denominator, teacher_norm=norm(teacher), prediction_norm=norm(prediction),
                fixed_gate=0.03, passed=bool(math.isfinite(fidelity) and fidelity <= 0.03))
            save_npz("teacher_fidelity_loss_only.npz", query_rows=rows, teacher_K21_loss_only=teacher,
                     error_loss_only=delta)
            result["passed"] = result["fidelity"]["passed"]
            result["stage"] = "complete"
            result["backend"] = threadpool_info()
            result["numpy_version"], result["scipy_version"] = np.__version__, scipy.__version__
            resource_gate()
    except BaseException as exc:
        result["passed"] = False
        result["exception_type"] = type(exc).__name__
        result["exception"] = str(exc)
    finally:
        result["wall_s"] = time.perf_counter()-started
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        result["peak_rss_bytes"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        result["final_hashes"] = {k: sha(v) for k, v in paths.items()}
        result["hashes_unchanged"] = result["final_hashes"] == hashes
        result["passed"] = bool(result["passed"] and result["hashes_unchanged"] and
                                result["peak_rss_bytes"] <= 2*(1 << 30))
        result["artifacts"] = {p.name: sha(p) for p in sorted(OUT.glob("*.npz"))}
        write_json(OUT/"result.json", result)
        print(json.dumps({k:v for k,v in result.items() if k not in
                          ("paths","hashes","final_hashes","backend")}, allow_nan=False), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print("INERT: one fixed rank35 inverse/fidelity check; --execute required.")
