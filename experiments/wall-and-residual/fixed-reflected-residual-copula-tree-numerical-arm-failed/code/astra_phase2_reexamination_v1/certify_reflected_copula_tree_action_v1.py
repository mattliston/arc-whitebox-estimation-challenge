"""Synthetic algebra only; default inert. No network weights or truth labels.

Certifies reflected-ReLU first two Hermite responses, the three-product
tree action, and explicit trace/repeated-plane ownership. It does not
certify a copula inverse, PSD feasibility, truncation fidelity or a score.
"""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.integrate import quad


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def cdf(x):
    return 0.5 * math.erfc(-x / math.sqrt(2))


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def star_tensor(s):
    n = len(s)
    e = s - (2.0 / 3.0) * np.diag(np.diag(s))
    eye = np.eye(n)
    return (np.einsum("ij,ik->ijk", eye, e)
            + np.einsum("ik,ij->ijk", eye, e)
            + np.einsum("jk,ji->ijk", eye, e))


def star_action(s, w):
    e = s - (2.0 / 3.0) * np.diag(np.diag(s))
    q = e @ w
    return (w * w).T @ q + 2 * (w * q).T @ w


def plane(t):
    return np.einsum("iij->ij", t)


def pushed_plane(t, w):
    pushed = np.einsum("ijk,ia,jb,kc->abc", t, w, w, w, optimize=True)
    return plane(pushed)


def execute(out):
    if out.exists():
        raise FileExistsError(out)
    rng = np.random.default_rng(90622471)
    endpoints = []
    for alpha in (-3.0, -1.0, 0.0, 0.75, 3.0):
        for eta in (-1.0, 1.0):
            tau = 0.7 if eta < 0 else 1.2
            m0 = phi(alpha) + alpha * cdf(alpha)
            cut = -eta * alpha

            def response(order):
                def integrand(z):
                    f = eta * tau * (max(eta * z + alpha, 0.0) - m0)
                    h = (1.0, z, z * z - 1.0)[order]
                    return f * h * phi(z)
                left = quad(integrand, -np.inf, cut, epsabs=1e-12,
                            epsrel=1e-12, limit=100)[0]
                right = quad(integrand, cut, np.inf, epsabs=1e-12,
                             epsrel=1e-12, limit=100)[0]
                return left + right

            measured = np.array([response(d) for d in range(3)])
            expected = np.array([0.0, tau * cdf(alpha), eta * tau * phi(alpha)])
            error = float(np.max(np.abs(measured - expected)
                                 / np.maximum(1.0, np.abs(expected))))
            endpoints.append(dict(alpha=alpha, eta=eta, tau=tau,
                                  normalized_absolute_error=error))

    fixtures = []
    for fixture in range(12):
        n = 4 + fixture % 4
        m = 3 + fixture % 5
        z = rng.normal(size=(n, n))
        raw = z @ z.T + np.eye(n)
        scale = np.sqrt(np.diag(raw))
        r = raw / np.outer(scale, scale)
        a = 0.5 + np.abs(rng.normal(size=n))
        q = rng.normal(size=n)
        b = a[:, None] * r
        anchor = np.diag(q)
        tree = (np.einsum("ir,jr,kr->ijk", anchor, b, b)
                + np.einsum("ir,jr,kr->ijk", b, anchor, b)
                + np.einsum("ir,jr,kr->ijk", b, b, anchor))
        tree21 = b * b * q[None, :] + 2 * (q * a)[:, None] * b.T
        metric = raw
        trace_vector = rng.normal(size=n)
        trace = (np.einsum("ij,k->ijk", metric, trace_vector)
                 + np.einsum("ik,j->ijk", metric, trace_vector)
                 + np.einsum("jk,i->ijk", metric, trace_vector))
        trace21 = (np.diag(metric)[:, None] * trace_vector[None, :]
                   + 2 * metric * trace_vector[:, None])
        supplied21 = rng.normal(size=(n, n))
        rest = supplied21 - tree21 - trace21
        total = tree + trace + star_tensor(rest)
        w = rng.normal(size=(n, m)) / math.sqrt(n)
        x = w.T @ b
        wt = w.T
        tree_read = (x * x) @ (wt * q).T + 2 * (x * wt) @ (x * q).T
        metric_next = w.T @ metric @ w
        vector_next = w.T @ trace_vector
        trace_read = (np.diag(metric_next)[:, None] * vector_next[None, :]
                      + 2 * metric_next * vector_next[:, None])
        full_read = tree_read + trace_read + star_action(rest, w)
        errors = dict(tree_plane=relative(tree21, plane(tree)),
                      trace_plane=relative(trace21, plane(trace)),
                      restored_plane=relative(plane(total), supplied21),
                      tree_affine=relative(tree_read, pushed_plane(tree, w)),
                      trace_affine=relative(trace_read, pushed_plane(trace, w)),
                      full_affine=relative(full_read, pushed_plane(total, w)),
                      symmetry=relative(total, total.transpose(2, 1, 0)))
        fixtures.append(dict(fixture=fixture, input_width=n, output_width=m,
                             errors=errors))

    worst_endpoint = max(row["normalized_absolute_error"] for row in endpoints)
    worst_tensor = max(value for row in fixtures for value in row["errors"].values())
    result = dict(scope="synthetic reflected responses/tree action/plane ownership only",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  finished_utc=datetime.now(timezone.utc).isoformat(),
                  thresholds=dict(endpoint=1e-11, tensor=1e-12),
                  worst_endpoint_error=worst_endpoint, worst_tensor_error=worst_tensor,
                  endpoints=endpoints, fixtures=fixtures,
                  passed=bool(worst_endpoint <= 1e-11 and worst_tensor <= 1e-12))
    with out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("endpoints", "fixtures")}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).with_name("reflected_copula_tree_action_certificate_v1.json"))
    args = parser.parse_args()
    if args.execute:
        execute(args.output)
    else:
        print("INERT: --execute runs synthetic algebra only; no network or target inputs.")
