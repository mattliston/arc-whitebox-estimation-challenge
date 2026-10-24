"""Freeze causal pre-plane inputs; verify unchanged bank prefix, no targets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import resource
import time
import numpy as np
from threadpoolctl import threadpool_limits
import compact_fullpair_final_reference_v1 as parent
from k3_radial_numpy import gaussian_raw
from radial_features import features
from radial_gnn_numpy import gnn_step

HERE = Path(__file__).resolve().parent
OUT = HERE / 'cohort_preplane_gain_inputs_mini0057_v1.npz'
assert not OUT.exists() and not OUT.with_suffix('.json').exists()
frozen = HERE / 'public_p2_third_cell_mini0057_v1.npz'
fr = json.loads(frozen.with_suffix('.json').read_text())
assert fr['passed'] and hashlib.sha256(frozen.read_bytes()).hexdigest() == fr['output_sha256']
f = dict(np.load(frozen))
W = np.load(HERE / 'public_p2_third_cell_mini0057_weights_v1.npy')
assert hashlib.sha256(W.tobytes()).hexdigest() == fr['weights_array_sha256']
P = dict(np.load(HERE / 'payload_compact_feed_fullpair_v1.npz'))
state = np.zeros((1024, 64), np.float32)
prefix, xs, saved = [], [], {}
start = time.time()


class Complete(Exception):
    pass


def preplane(layer, mu, Cpre, pp, lam, cohort_counts, live_indices, w):
    if layer != 13:
        return
    assert not saved and len(cohort_counts) == 13
    assert 2 * sum(cohort_counts) == len(lam) == pp.shape[1]
    sigma, alpha, pdf, gate, *_ = gaussian_raw(mu, np.diag(Cpre))
    cpdiag = np.sum(pp.astype(np.float64) ** 3 * lam.astype(np.float64), axis=1)
    src = state @ P['msg_w'].T + P['msg_b']
    cm = int(P['msg_w'].shape[0]) // 3
    assert cm == 16
    w = w.astype(np.float32); w2 = w * w; w3 = w2 * w
    m1 = w.T @ src[:, :cm]; m2 = w2.T @ src[:, cm:2*cm]; m3 = w3.T @ src[:, 2*cm:]
    e1 = w2.T @ (src[:, :cm] ** 2)
    e2 = (w2 * w2).T @ (src[:, cm:2*cm] ** 2)
    e3 = (w3 * w3).T @ (src[:, 2*cm:] ** 2)
    message = np.concatenate([m1, m2, m3, np.log1p(np.maximum(e1, 0)),
                              np.log1p(np.maximum(e2, 0)), np.log1p(np.maximum(e3, 0))], axis=1)
    local = np.column_stack([np.ones(len(mu)), alpha, gate, pdf, alpha * pdf,
                             (alpha * alpha - 1) * pdf, np.log(sigma), cpdiag / sigma ** 3])
    F = np.column_stack([local, message.astype(np.float64)])
    assert F.shape == (1024, 104) and np.isfinite(F).all()
    offset = 0
    for count in cohort_counts:
        assert np.array_equal(lam[offset:offset+count], -lam[offset+count:offset+2*count])
        offset += 2 * count
    saved.update(features=F, pushed_factors=pp.copy(), signed_weights=lam.copy(),
                 cohort_counts=np.array(cohort_counts, dtype=np.int64), live_indices=live_indices.copy(),
                 incoming_state=state.copy(), cp_diagonal=cpdiag, pre_mu=mu.copy(), pre_covariance=Cpre.copy())


def hook(layer, m, C, K21, aux, w):
    global state
    x = features(layer, m, C, K21, aux, w)
    delta, state = gnn_step(layer, x, aux['sigma'], w, state, P)
    prefix.append(m + delta); xs.append(x)
    if layer == 13:
        assert saved
        current = dict(Cpre=aux['pre_cov'], K21=K21, mu=aux['alpha'] * aux['sigma'], sigma=aux['sigma'],
                       alpha=aux['alpha'], pdf=aux['pdf'], gate=aux['gate'], gaussian_mean=aux['gaussian_mean'],
                       host=m, post_covariance=C, post13_head_delta=delta, state=state, x=x,
                       prefix_means=np.stack(prefix), prefix_features=np.stack(xs))
        checks = {key: bool(np.array_equal(value, f[key])) for key, value in current.items()}
        assert all(checks.values()), checks
        np.savez(OUT, **saved)
        result = dict(passed=True, network='gregory-wilson', ordinal=57, role='train', pre_layer=13,
                      feature_count=104, factor_columns=int(saved['pushed_factors'].shape[1]),
                      cohort_counts=saved['cohort_counts'].tolist(), frozen_pre13_byte_checks=checks,
                      current_plane_or_nonlinear_head_in_features=False, target_values_decoded=0,
                      frozen_parent_inputs_sha256=fr['output_sha256'], weights_array_sha256=fr['weights_array_sha256'],
                      instrument_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                      output_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),
                      wall_s=time.time()-start, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                      utc_finished=datetime.now(timezone.utc).isoformat(),
                      qualification='Target-free unchanged-parent replay; features captured before current CP plane and nonlinear/head calls. Incoming recurrent messages omit unavailable normalized-current-g multiplier. No target source file decoded, coefficient fit or state modification.')
        OUT.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2), flush=True)
        raise Complete
    return delta


source = parent.source
edits = {
    '    lam = np.zeros(0, np.float32)': '    lam = np.zeros(0, np.float32)\n    cohort_counts = []',
    '        cp_pre = np.zeros((n, n))':
        '        preplane(l,mu,Cpre,pp,lam,cohort_counts,live_indices,W[l])\n        cp_pre = np.zeros((n, n))',
    '        lam = np.concatenate([lam, a[selected].astype(np.float32), -a[selected].astype(np.float32)])':
        '        lam = np.concatenate([lam, a[selected].astype(np.float32), -a[selected].astype(np.float32)])\n        cohort_counts.append(len(selected))',
}
for old, new in edits.items():
    assert source.count(old) == 1, old
    source = source.replace(old, new)
env = dict(parent.env, preplane=preplane)
exec(source, env)
with threadpool_limits(limits=3):
    try:
        env['predict'](W, hook=hook, gate_floor=.01, cov32=True, metric='covariance')
    except Complete:
        pass
assert OUT.exists()
