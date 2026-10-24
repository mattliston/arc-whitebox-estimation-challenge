"""Fixed target-blind correlation-neighborhood features for the current card."""
import numpy as np
from radial_features import features as parent_features, FEATURE_NAMES as PARENT_NAMES

GRAPH_NAMES = (
    "graph_top8_signed_alpha", "graph_top8_signed_gate", "graph_top8_signed_pdf",
    "graph_top8_abs_alpha", "graph_top8_abs_gate", "graph_top8_abs_pdf",
    "graph_top8_r2", "graph_max_abs_rho",
    "graph_soft8_alpha", "graph_soft8_gate", "graph_soft8_pdf",
    "graph_soft32_alpha", "graph_soft32_gate", "graph_soft32_pdf",
    "graph_post_power8_loading", "graph_post_rho_mean", "graph_post_rho_rms",
    "graph_post_centered_log_var", "graph_post_cov_alpha", "graph_post_cov_gate")
FEATURE_NAMES = PARENT_NAMES + GRAPH_NAMES


def graph_features(m, C, aux):
    n = len(m)
    s = aux["sigma"]
    rho = np.clip(aux["pre_cov"] / np.outer(s, s), -1., 1.)
    rho = rho.copy()
    np.fill_diagonal(rho, 0.)
    absrho = np.abs(rho)
    index = np.argpartition(absrho, n-8, axis=1)[:, -8:]
    selected = np.take_along_axis(rho, index, axis=1)
    xs = np.stack((aux["alpha"], aux["gate"], aux["pdf"]), axis=1)
    neighbor = xs[index]
    signed = np.sum(selected[:, :, None] * neighbor, axis=1)
    absolute = np.sum(np.abs(selected)[:, :, None] * neighbor, axis=1)
    r2 = np.sum(selected * selected, axis=1)
    largest = np.max(np.abs(selected), axis=1)
    soft = []
    for gamma in (8., 32.):
        logits = gamma * rho * rho
        logits = logits - np.max(logits, axis=1)[:, None]
        weights = np.exp(logits)
        np.fill_diagonal(weights, 0.)
        weights = weights / np.maximum(np.sum(weights, axis=1)[:, None], 1e-30)
        soft.append(weights @ xs)
    post = (C + C.T) * .5
    pv = np.maximum(np.diag(post), 1e-30)
    ps = np.sqrt(pv)
    pr = np.clip(post / np.outer(ps, ps), -1., 1.)
    np.fill_diagonal(pr, 0.)
    i = np.arange(n, dtype=float)[:, None]
    j = np.arange(8, dtype=float)[None, :]
    Q = np.linalg.qr(np.cos(np.pi * (i + .5) * j / n), mode="reduced")[0]
    for _ in range(8):
        Q = np.linalg.qr(post @ Q, mode="reduced")[0]
    loading = np.sum(Q * Q, axis=1)
    post_block = np.stack((loading, np.mean(pr, axis=1),
                           np.sqrt(np.mean(pr * pr, axis=1)),
                           np.log(pv / np.mean(pv)),
                           (post @ aux["alpha"]) / ps,
                           (post @ aux["gate"]) / ps), axis=1)
    return np.concatenate((signed, absolute, r2[:, None], largest[:, None],
                           soft[0], soft[1], post_block), axis=1).astype(np.float32)


def features(l, m, C, K21, aux, W):
    original = parent_features(l, m, C, K21, aux, W)
    return np.concatenate((original, graph_features(m, C, aux)), axis=1)
