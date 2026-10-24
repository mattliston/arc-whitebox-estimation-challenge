"""Inert, unexecuted prototype for one fixed Phase-2 positive GMM4 smoke.

No execution is authorized by writing this file. Root source review and
separate resource authorization precede --execute. Default/--help import
no numerical library, open no array, and do no model work.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time

HERE = Path(__file__).resolve().parent
DESIGN = "POSITIVE_GMM4_SPLIT_MERGE_DESIGN_V1.md"
OUT = HERE / "positive_gmm4_split_merge_smoke_v1"
CERT_SOURCE = "certify_positive_gmm_merge_algebra_v1.py"
CERT_RECEIPT = "positive_gmm_merge_algebra_certificate_v1.json"
CERT_SOURCE_SHA = "7eae99bcc973d78b0def3ccd8bb72ea75e897c8ea5215355489ed1efe9050735"
CERT_RECEIPT_SHA = "673e1c45ffb1debfa7642852f9f039fa41cec093f7056ff9da81006af728dc0b"
QUERY_SOURCE = "certify_gmm_gaussian_query_merge_v1.py"
QUERY_RECEIPT = "gmm_gaussian_query_merge_certificate_v1.json"
QUERY_SOURCE_SHA = "fad6272813d8c921ca6d122ea14bcdcf124c53baeb3a233d2027867ec33514d4"
QUERY_RECEIPT_SHA = "d9068cbf1305e356e557ee89b160330e3be60a015d6e80de53fe00021a0c3e82"
RANKS = (35, 100)
ORDER = 12
POWER_STEPS = 4
RAW_GATE = 6e-8
CONTROL_RATIO_GATE = .25
AUXILIARY_ENVELOPE = 35_000_000_000
RSS_CAP = 2 * 1024**3
MODEL_WALL_CAP = 120.


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def marginal(mu, variance, stats):
    require(np.isfinite(mu).all() and np.isfinite(variance).all(), "Nonfinite marginal input")
    require(float(variance.min()) >= 0, "Negative input variance: no jitter/floor repair")
    sigma = np.sqrt(variance)
    active = sigma > 0
    safe = np.where(active, sigma, 1.)
    alpha = np.where(active, mu/safe, 0.)
    magnitude = np.abs(alpha)
    pdf = np.exp(-.5*magnitude*magnitude)/math.sqrt(2*math.pi)
    tail_probability = ndtr(-magnitude)
    gate = np.where(alpha >= 0, 1-tail_probability, tail_probability)
    excess = pdf-magnitude*tail_probability
    mean = np.where(alpha >= 0, mu+sigma*excess, sigma*excess)
    mean = np.where(active, mean, np.maximum(mu, 0.))
    tolerance = 64*np.finfo(np.float64).eps*np.maximum(sigma, 1e-300)
    require(np.all(mean >= -tolerance), "Gaussian mean cancellation exceeded fixed roundoff tolerance")
    stats["mean_roundoff_lift_max"] = max(stats["mean_roundoff_lift_max"],
                                           float(np.max(np.maximum(-mean, 0))))
    mean = np.maximum(mean, 0.)
    # For alpha>=0, avoid subtracting two alpha^2-sized second moments.
    positive_variance = 1+(magnitude*magnitude-1)*tail_probability-magnitude*pdf-excess*excess
    negative_variance = (magnitude*magnitude+1)*tail_probability-magnitude*pdf-excess*excess
    post_variance = np.where(active, variance*np.where(alpha >= 0, positive_variance, negative_variance), 0.)
    vtolerance = 64*np.finfo(np.float64).eps*np.maximum(variance, 1e-300)
    require(np.all(post_variance >= -vtolerance),
            "Gaussian variance cancellation exceeded fixed roundoff tolerance")
    stats["variance_roundoff_lift_max"] = max(stats["variance_roundoff_lift_max"],
                                               float(np.max(np.maximum(-post_variance, 0))))
    return mean, np.maximum(post_variance, 0.), sigma, alpha, pdf, gate


def gaussian_map(mu, covariance, stats):
    """Exact Gaussian scalar moments; fixed positive Hermite12 covariance."""
    stats["gaussian_pair_maps"] += 1
    v = np.diag(covariance).copy()
    mean, post_v, sigma, alpha, pdf, gate = marginal(mu, v, stats)
    active = sigma > 0
    require(np.count_nonzero(covariance[~active]) == 0,
            "Zero-variance row has nonzero covariance")
    safe = np.where(active, sigma, 1.)
    rho = covariance/(safe[:, None]*safe[None, :])
    np.fill_diagonal(rho, active.astype(np.float64))
    require(np.isfinite(rho).all() and float(np.max(np.abs(rho))) <= 1+1e-6,
            "Rounded covariance failed fixed correlation-domain check; no clipping")
    # c_r = sigma*A_r/sqrt(r!), A_1=Phi and A_r=(-1)^(r-2)He_(r-2)*phi.
    coefficients = [sigma*gate]
    h0, h1 = np.ones_like(alpha), alpha.copy()
    for r in range(2, ORDER+1):
        if r == 2:
            h = h0
        elif r == 3:
            h = h1
        else:
            h = alpha*h1-(r-3)*h0
            h0, h1 = h1, h
        coefficients.append(sigma*((-1)**(r-2))*h*pdf/math.sqrt(math.factorial(r)))
    cov = np.zeros_like(covariance)
    rho_power = rho.copy()
    diagonal_partial = np.zeros_like(mu)
    for r, c in enumerate(coefficients, 1):
        cov += (c[:, None]*c[None, :])*rho_power
        diagonal_partial += c*c
        if r != ORDER:
            rho_power *= rho
    tail = post_v-diagonal_partial
    tolerance = 64*np.finfo(np.float64).eps*np.maximum(v, 1e-300)
    require(np.all(tail >= -tolerance), "Hermite diagonal tail is negative beyond roundoff")
    stats["tail_roundoff_lift_max"] = max(stats["tail_roundoff_lift_max"],
                                          float(np.max(np.maximum(-tail, 0))))
    # Explicit O(n) roundoff-limited nonnegative tail, not an eigenvalue repair.
    diagonal = np.diag_indices_from(cov)
    cov[diagonal] += np.maximum(tail, 0.)
    cov = .5*(cov+cov.T)
    require(np.isfinite(cov).all(), "Nonfinite Gaussian covariance")
    return mean, cov


def factor_state(component, stats):
    """A retained L32, not the rounded input matrix, owns next covariance."""
    covariance = component["C"]
    v = np.diag(covariance).copy()
    require(float(v.min()) >= 0 and np.isfinite(covariance).all(), "Invalid retained covariance")
    active = v > 0
    require(np.count_nonzero(covariance[~active]) == 0,
            "Nonzero covariance attached to deterministic coordinate")
    factor = np.zeros_like(covariance, dtype=np.float32)
    if np.any(active):
        indices = np.flatnonzero(active)
        sigma = np.sqrt(v[indices])
        correlation = covariance[np.ix_(indices, indices)]/(sigma[:, None]*sigma[None, :])
        correlation = .5*(correlation+correlation.T)
        # Unit diagonal defines the already-known variance, not added jitter.
        np.fill_diagonal(correlation, 1.)
        stats["cholesky_dimensions"].append(int(len(indices)))
        lower = np.linalg.cholesky(correlation)  # exactly one attempt, float64
        physical_lower = sigma[:, None]*lower
        lower32 = physical_lower.astype(np.float32)
        difference = lower32.astype(np.float64)-physical_lower
        norm = float(np.linalg.norm(physical_lower))
        relative = float(np.linalg.norm(difference))/max(norm, 1e-300)
        require(relative <= 1e-6 and np.isfinite(lower32).all(), "Factor rounding qualification failed")
        stats["factor_rounding_relative_max"] = max(stats["factor_rounding_relative_max"], relative)
        factor[np.ix_(indices, indices)] = lower32
        diag_factor = np.sum(lower32.astype(np.float64)*lower32.astype(np.float64), axis=1)
        stats["factor_zeroed_variance_coordinates"] += int(np.count_nonzero((v[indices] > 0) & (diag_factor == 0)))
    return dict(id=component["id"], p=component["p"], m=component["m"], L=factor)


def split_direction(component, stats):
    C, mu = component["C"], component["m"]
    q = np.full(len(mu), 1/math.sqrt(len(mu)), dtype=np.float64)
    for _ in range(POWER_STEPS):
        y = C@q
        stats["direction_matvecs_float64"] += 1
        norm = float(np.linalg.norm(y))
        if norm == 0:
            return None  # fixed zero-seed-action convention; no other direction
        require(math.isfinite(norm), "Nonfinite covariance power action")
        q = y/norm
    cq = C@q
    stats["direction_matvecs_float64"] += 1
    rayleigh = float(q@cq)
    require(rayleigh >= 0, "Negative split Rayleigh quotient: no direction repair")
    if rayleigh == 0:
        return None
    a = cq/math.sqrt(2*rayleigh)
    variance = np.diag(C)
    require(float(variance.min()) >= 0, "Negative variance before split selection")
    sigma = np.sqrt(variance)
    safe = np.where(sigma > 0, sigma, 1.)
    alpha = mu/safe
    curvature = np.where(sigma > 0,
                        np.exp(-.5*alpha*alpha)/(math.sqrt(2*math.pi)*safe), 0.)
    score = float(component["p"]*np.sum(curvature*a*a))
    require(math.isfinite(score) and score >= 0, "Invalid split score")
    return score, a


def merge_query_risk(a, b, stats):
    stats["merge_pair_comparisons"] += 1
    total = a["p"]+b["p"]
    alpha, beta = a["p"]/total, b["p"]/total
    d = a["m"]-b["m"]
    B = a["C"]-b["C"]+((beta-alpha)/3*d)[:, None]*d[None, :]
    bd = B@d
    coefficient = total*alpha*beta
    norm_squared = coefficient**2*(3*float(d@d)*float(np.sum(B*B))+6*float(bd@bd))
    trace_vector = coefficient*(d*float(np.trace(B))+2*bd)
    trace_squared = float(trace_vector@trace_vector)
    risk = 6*norm_squared+9*trace_squared
    require(math.isfinite(risk) and risk >= 0, "Invalid isotropic scalar-third merge risk")
    return risk, norm_squared, trace_squared


def merge_pair(a, b, new_id):
    total = a["p"]+b["p"]
    alpha, beta = a["p"]/total, b["p"]/total
    d = a["m"]-b["m"]
    mean = alpha*a["m"]+beta*b["m"]
    covariance = alpha*a["C"]+beta*b["C"]+(alpha*beta*d)[:, None]*d[None, :]
    return dict(id=new_id, p=total, m=mean, C=.5*(covariance+covariance.T))


def predict(W, mixture):
    require(W.shape == (16, 1024, 1024) and W.dtype == np.float32 and np.isfinite(W).all(),
            "Only fixed width1024/depth16 Phase2 weights are accepted")
    n = W.shape[1]
    stats = dict(gemm_float32=0, gaussian_pair_maps=0, cholesky_dimensions=[],
                 direction_matvecs_float64=0, merge_pair_comparisons=0,
                 mean_roundoff_lift_max=0., variance_roundoff_lift_max=0.,
                 tail_roundoff_lift_max=0., factor_rounding_relative_max=0.,
                 factor_zeroed_variance_coordinates=0, layers=[])
    states = [dict(id=0, p=1., m=np.zeros(n), L=None)]
    next_id = 1
    predictions = []
    for layer in range(16):
        w32 = W[layer]
        w64 = w32.astype(np.float64)
        prior = []
        if layer == 15:
            final = np.zeros(n, np.float64)
            for comp in states:
                mu = w64.T@comp["m"]
                B = comp["L"].T@w32
                stats["gemm_float32"] += 1
                B64 = B.astype(np.float64)
                variance = np.sum(B64*B64, axis=0)
                mean = marginal(mu, variance, stats)[0]
                final += comp["p"]*mean
            predictions.append(final)
            stats["layers"].append(dict(layer=layer, input_components=len(states), scalar_final=True))
            break
        for comp in states:
            mu = w64.T@comp["m"]
            if layer == 0:
                require(comp["L"] is None and len(states) == 1, "Input-I special case changed")
                raw = w32.T@w32
                stats["gemm_float32"] += 1
            else:
                B = comp["L"].T@w32
                raw = B.T@B
                stats["gemm_float32"] += 2
            C = raw.astype(np.float64)
            C = .5*(C+C.T)
            require(np.isfinite(C).all() and float(np.diag(C).min()) >= 0,
                    "Invalid rounded affine covariance")
            prior.append(dict(id=comp["id"], p=comp["p"], m=mu, C=C))
        chosen, split_score = None, None
        if mixture:
            options = []
            for comp in prior:
                direction = split_direction(comp, stats)
                if direction is not None and direction[0] > 0:
                    options.append((direction[0], comp["id"], direction[1]))
            if options:
                split_score, chosen, shift = min(options, key=lambda x: (-x[0], x[1]))
                children = []
                for comp in prior:
                    if comp["id"] != chosen:
                        children.append(comp)
                        continue
                    residual = comp["C"]-shift[:, None]*shift[None, :]
                    residual = .5*(residual+residual.T)
                    for node, weight in ((-math.sqrt(3), 1/6), (0., 2/3), (math.sqrt(3), 1/6)):
                        children.append(dict(id=next_id, p=comp["p"]*weight,
                            m=comp["m"]+node*shift, C=residual))
                        next_id += 1
                prior = children
        mapped = []
        for comp in prior:
            mean, covariance = gaussian_map(comp["m"], comp["C"], stats)
            mapped.append(dict(id=comp["id"], p=comp["p"], m=mean, C=covariance))
        merges = []
        while mixture and len(mapped) > 4:
            mapped.sort(key=lambda c: c["id"])
            scores = []
            for i in range(len(mapped)):
                for j in range(i+1, len(mapped)):
                    risk, norm_squared, trace_squared = merge_query_risk(mapped[i], mapped[j], stats)
                    scores.append((risk, mapped[i]["id"], mapped[j]["id"], i, j,
                                   norm_squared, trace_squared))
            score, ida, idb, i, j, norm_squared, trace_squared = min(scores, key=lambda x: (x[0], x[1], x[2]))
            merged = merge_pair(mapped[i], mapped[j], next_id)
            next_id += 1
            mapped = [c for k, c in enumerate(mapped) if k not in (i, j)]+[merged]
            merges.append(dict(left_id=ida, right_id=idb, isotropic_scalar_third_risk=score,
                               lost_third_squared=norm_squared, lost_third_trace_squared=trace_squared))
        mapped.sort(key=lambda c: c["id"])
        mass = math.fsum(c["p"] for c in mapped)
        require(abs(mass-1.) <= 1e-12 and all(c["p"] > 0 for c in mapped), "Mixture mass invalid")
        for c in mapped:
            c["p"] /= mass  # explicit roundoff-only positive mass normalization
        predictions.append(sum((c["p"]*c["m"] for c in mapped), np.zeros(n)))
        states = [factor_state(c, stats) for c in mapped]
        stats["layers"].append(dict(layer=layer, mapped_components=len(prior),
             retained_components=len(states), split_parent_id=chosen, split_score=split_score,
             merges=merges, pre_normalization_mass=mass))
    known = 2*n**3*stats["gemm_float32"] + (2/3)*sum(d**3 for d in stats["cholesky_dimensions"])
    stats["known_gemm_cholesky_charge_upper"] = known
    stats["auxiliary_charge_provisional_reserve"] = AUXILIARY_ENVELOPE if mixture else None
    stats["mixture_total_charge_envelope"] = known+AUXILIARY_ENVELOPE if mixture else None
    stats["mixture_utilization_envelope"] = (known+AUXILIARY_ENVELOPE)/2**41 if mixture else None
    stats["bill_is_metered"] = False
    return np.stack(predictions), stats


def execute():
    require("/anaconda3/" not in sys.executable, "Use the repo NumPy interpreter, never AP/Torch")
    require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, "Use an explicitly nice10 worker")
    require(not OUT.exists(), "Preserve previous smoke artifacts; refusing overwrite")
    sources = [Path(__file__).name, DESIGN, CERT_SOURCE, CERT_RECEIPT, QUERY_SOURCE, QUERY_RECEIPT]
    source_hashes = {name: sha(HERE/name) for name in sources}
    require(source_hashes[CERT_SOURCE] == CERT_SOURCE_SHA and
            source_hashes[CERT_RECEIPT] == CERT_RECEIPT_SHA, "Frozen independent algebra certificate changed")
    require(json.loads((HERE/CERT_RECEIPT).read_text())["passed"], "Independent algebra did not qualify")
    require(source_hashes[QUERY_SOURCE] == QUERY_SOURCE_SHA and
            source_hashes[QUERY_RECEIPT] == QUERY_RECEIPT_SHA, "Frozen independent Wick certificate changed")
    require(json.loads((HERE/QUERY_RECEIPT).read_text())["passed"], "Independent Wick identity did not qualify")
    paths = {rank: HERE/"rows_current_transport_shadow_v2"/f"row_{rank:04d}"/"weights.npy" for rank in RANKS}
    weight_hashes = {str(rank): sha(path) for rank, path in paths.items()}
    global np, ndtr
    import numpy as np
    from scipy.special import ndtr
    from threadpoolctl import threadpool_limits, threadpool_info
    require(not any(k == "torch" or k.startswith("torch.") for k in sys.modules), "Unexpected Torch import")
    OUT.mkdir(exist_ok=False)
    write_json(OUT/"started.json", dict(source_hashes=source_hashes, weight_hashes=weight_hashes,
               ranks=RANKS, truth_access=False, training=False, raw_gate=RAW_GATE,
               control_ratio_gate=CONTROL_RATIO_GATE, threads=2))
    started = time.perf_counter()
    saved, runs = {}, []
    try:
        with threadpool_limits(limits=2):
            for rank in RANKS:
                W = np.load(paths[rank], allow_pickle=False)
                for mixture in (False, True):
                    begin = time.perf_counter()
                    prediction, stats = predict(W, mixture)
                    wall = time.perf_counter()-begin
                    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                    require(wall <= MODEL_WALL_CAP and rss <= RSS_CAP, "Fixed research resource cap exceeded")
                    require(np.isfinite(prediction).all() and float(prediction.min()) >= 0, "Invalid prediction")
                    if mixture:
                        require(stats["mixture_utilization_envelope"] <= .15, "Static complete-envelope gate failed")
                    name = f"rank_{rank:04d}_{'gmm4' if mixture else 'single'}.npz"
                    with (OUT/name).open("xb") as stream:
                        np.savez(stream, prediction_unclipped=prediction,
                                 prediction_served_float32=prediction.astype(np.float32))
                    saved[(rank, mixture)] = prediction.astype(np.float32).astype(np.float64)
                    record = dict(rank=rank, mixture=mixture, stats=stats, wall_s=wall,
                                  process_peak_rss_bytes=rss, output=name, output_sha256=sha(OUT/name))
                    runs.append(record)
                    print(json.dumps(dict(rank=rank, mixture=mixture, wall_s=wall, peak_rss_bytes=rss)), flush=True)
            backend = threadpool_info()
        require(source_hashes == {name: sha(HERE/name) for name in sources}, "Source changed before predictions froze")
        require(weight_hashes == {str(rank): sha(path) for rank, path in paths.items()}, "Weights changed")
        write_json(OUT/"predictions_frozen_before_truth.json", dict(source_hashes=source_hashes,
                   weight_hashes=weight_hashes, runs=runs, backend=backend, truth_access=False))
        # Both arms on BOTH fixed nets are frozen before any existing truth is opened.
        judged = []
        for rank in RANKS:
            path = HERE.parent/"public_g22_operator_features_v1"/"truthpost"/f"truthpost_rank{rank:04d}.npz"
            truth_hash = sha(path)
            with np.load(path, allow_pickle=False) as z:
                truth = z["mc_post"].astype(np.float64)
            require(truth.shape == (16, 1024) and np.isfinite(truth).all(), "Wrong fixed existing truth shape")
            mse_single = np.mean((saved[(rank, False)]-truth)**2, axis=1)
            mse_gmm = np.mean((saved[(rank, True)]-truth)**2, axis=1)
            require(float(mse_single[-1]) > 0, "Zero control error is inconclusive")
            ratio = float(mse_gmm[-1]/mse_single[-1])
            require(sha(path) == truth_hash, "Truth changed while scoring")
            judged.append(dict(rank=rank, existing_truth_path=str(path.relative_to(HERE.parent)),
                existing_truth_sha256=truth_hash, single_per_layer_mse=mse_single.tolist(),
                gmm4_per_layer_mse=mse_gmm.tolist(), final_raw_ratio_to_control=ratio,
                passed=bool(mse_gmm[-1] <= RAW_GATE and ratio <= CONTROL_RATIO_GATE)))
        require(source_hashes == {name: sha(HERE/name) for name in sources}, "Source changed during scoring")
        result = dict(passed=all(r["passed"] for r in judged), source_hashes=source_hashes,
            weight_hashes=weight_hashes, records=judged, runs=runs, new_mc=False, training=False,
            wall_s=time.perf_counter()-started,
            interpretation="Fixed two-net existing-truth raw smoke; pass earns only a separate small coadaptation review, not panel/meter/runtime/submission")
        write_json(OUT/"result.json", result)
        print(json.dumps(result, allow_nan=False), flush=True)
    except BaseException as exc:
        write_json(OUT/"failure.json", dict(passed=False, exception_type=type(exc).__name__,
                   message=str(exc), source_hashes=source_hashes, runs=runs,
                   wall_s=time.perf_counter()-started))
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--execute", action="store_true", help="Separate root approval required")
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps(dict(execute=False, ranks=RANKS, gaussian_order=ORDER,
              power_steps=POWER_STEPS, maximum_retained_components=4,
              raw_gate=RAW_GATE, control_ratio_gate=CONTROL_RATIO_GATE,
              numerical_imports=False, truth_access=False, output=str(OUT))))
        return
    execute()


if __name__ == "__main__":
    main()
