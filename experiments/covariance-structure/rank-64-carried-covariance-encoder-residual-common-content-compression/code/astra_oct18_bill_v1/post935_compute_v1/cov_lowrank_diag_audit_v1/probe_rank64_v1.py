"""Fixed saved-data discriminator only; no SDK, targets, eigensolver or model."""
import hashlib, json, os, pathlib, resource, subprocess, time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    assert os.environ.get(name) == '1', name
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE.parents[2]
ACC = RUNS / 'astra_oct18_accuracy_v1'
P2 = RUNS.parents[2]
CAP = ACC / 'PRICE_CAPTURE_RANK16_v1.npz'
WEIGHTS = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/0016/weights.npy'
EXPECTED_CAP = '563744cd09dcaeb68243c986eaf8f4b05082c0b914c5acb50791d6e78806fbea'
EXPECTED_W = 'b1ef275271cc3d03f6f91b3692ce68738363bf74d5b20ba55f6a8e19eb100a48'

def free_pct():
    text = subprocess.check_output(['/usr/bin/memory_pressure'], text=True)
    return int(text.split('System-wide memory free percentage: ')[1].split('%')[0])

def pchol64(C):
    # First-index tie break; fixed rank, no adaptive choice of rank/algorithm.
    A = (C + C.T) * .5
    n = len(A); Z = np.zeros((n, 64)); d = np.diag(A).copy(); pivots = []
    tolerance = 100*np.finfo(np.float64).eps*max(1., float(np.max(d)))
    for j in range(64):
        p = int(np.argmax(d)); pivot = float(d[p])
        if pivot <= tolerance:
            return Z[:, :j], pivots, {'terminated': True, 'pivot': pivot, 'tolerance': tolerance}
        v = A[:, p] - Z[:, :j] @ Z[p, :j]
        Z[:, j] = v / np.sqrt(pivot)
        d -= Z[:, j]**2
        pivots.append(p)
    return Z, pivots, {'terminated': False, 'min_residual_diagonal': float(d.min())}

def norm(x): return float(np.linalg.norm(x))
def ratio(a, b): return float(a/max(b, np.finfo(np.float64).tiny))
def metric(C, W):
    Z, pivots, status = pchol64(C)
    D = np.diag(C) - np.sum(Z*Z, axis=1)  # Preserve all residual D, including any negatives.
    C_hat = Z @ Z.T
    np.fill_diagonal(C_hat, np.diag(C))
    E = C_hat-C
    off = C.copy(); np.fill_diagonal(off, 0.)
    full = W.T @ (C @ W)
    U = W.T @ Z
    low_push = U @ U.T
    weighted_gram = W.T @ (D[:, None]*W)
    approximation = low_push + weighted_gram
    pushed_E = approximation-full
    direct_E = W.T @ (E @ W)
    v, dv = np.diag(full), np.diag(pushed_E)
    return {
        'shape_C': list(C.shape), 'shape_W': list(W.shape), 'rank': Z.shape[1],
        'pivots': pivots, 'factor_status': status,
        'input_asymmetry_relative_fro': ratio(norm(C-C.T), norm(C)),
        'offdiag_relative_fro_error': ratio(norm(E), norm(off)),
        'covariance_relative_fro_error': ratio(norm(E), norm(C)),
        'exact_diagonal_max_error': float(np.max(np.abs(np.diag(E)))),
        'weighted_gram_explicit': True, 'D_min': float(D.min()), 'D_max': float(D.max()),
        'D_trace_fraction': ratio(float(D.sum()), float(np.trace(C))),
        'push_relative_fro_error': ratio(norm(pushed_E), norm(full)),
        'push_diag_relative_rms_error': ratio(norm(dv), norm(v)),
        'push_diag_relative_sd_error': ratio(float(np.std(dv)), float(np.std(v))),
        'push_diag_error_sd_over_mean_variance': ratio(float(np.std(dv)), float(np.mean(v))),
        'push_diag_relative_bias': ratio(float(np.mean(dv)), float(np.mean(v))),
        'push_identity_relative_error': ratio(norm(direct_E-pushed_E), norm(full)),
        'weighted_gram_relative_fro': ratio(norm(weighted_gram), norm(full)),
    }

def main():
    started = time.monotonic(); free_before = free_pct(); assert free_before >= 50
    assert hashlib.sha256(CAP.read_bytes()).hexdigest() == EXPECTED_CAP
    assert WEIGHTS.exists(), 'Matching W absent: STOP, do not capture.'
    W_all = np.load(WEIGHTS, allow_pickle=False)
    assert hashlib.sha256(W_all.tobytes()).hexdigest() == EXPECTED_W
    assert W_all.shape == (16,1024,1024)
    saved = np.load(CAP, allow_pickle=False)
    def get(l, key):
        for k in (f'L{l}_{key}', f'L{l}/{key}', f'{l}_{key}'):
            if k in saved: return saved[k]
        raise KeyError((l, key, saved.files))
    rows=[]
    for l in (13,14):
        host = get(l,'host').astype(np.float64)
        C_archive = get(l,'cov_after_edge').astype(np.float64)
        prediction = saved['predictions'][l].astype(np.float64)
        dm = prediction-host
        diag0 = np.diag(C_archive).copy()
        diag = ((diag0-host*dm)-host*dm)-dm*dm
        rc = -.08-.5438712708150119
        diag -= rc*(diag0-diag)
        m32, d32 = host.astype(np.float32), dm.astype(np.float32)
        mdm = np.outer(m32,d32)
        C0 = C_archive.astype(np.float32)
        C32 = ((C0-mdm)-mdm.T)-np.outer(d32,d32)
        C32 -= np.float32(rc)*(C0-C32)
        np.fill_diagonal(C32, diag)
        C = C32.astype(np.float64)
        gate = get(l,'gate')
        nlive = min(1024, int(np.ceil(np.count_nonzero(gate>=.001)/64))*64)
        live = np.arange(1024) if nlive==1024 else np.argsort(-gate,kind='stable')[:nlive]
        assert 64 <= nlive <= 1024
        W = W_all[l+1].astype(np.float64)
        active = metric(C[np.ix_(live,live)], W[live])
        # This second fixed rank64 evaluation is the literal archived covariance,
        # avoiding any assertion that reconstructed head updates are captured bytes.
        archived = metric(C_archive, W)
        rows.append({'layer':l, 'next_weight_index':l+1,
            'active_roster_rule':'gate>=.001, round up64; stable top gate unless full1024',
            'active_roster_n':nlive,'actual_carried_reconstruction':active,
            'literal_cov_after_edge_full_snapshot':archived})
    free_after=free_pct(); assert free_after>=50
    result={'scope':'saved v61j rank16 only, NOT current95c equivalence or endpoint evidence',
        'capture_sha256':EXPECTED_CAP,'weights_raw_sha256':EXPECTED_W,
        'algorithm':'fixed greedy largest-residual-diagonal pivoted Cholesky rank64 + exact diagonal residual D',
        'coefficient_or_rank_search':False, 'no_sdk_models_eig_targets':True,
        'reconstruction_caveat':'cov_after_edge precedes learned mean correction. dm recovered from predictions-host and original F32 operation order; no byte-parity claim for reconstructed C. Literal archived C is reported separately.',
        'rows':rows, 'elapsed_seconds':time.monotonic()-started,
        'free_before_pct':free_before,'free_after_pct':free_after,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    assert result['peak_rss_bytes'] < 2_000_000_000
    (HERE/'RANK64_RESULT_v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'COMPLETE','elapsed':result['elapsed_seconds'],'peak_rss':result['peak_rss_bytes'],
        'rows':[{'layer':r['layer'],'live':r['active_roster_n'],
                 'metrics':{k:v for k,v in r['actual_carried_reconstruction'].items() if k in ('offdiag_relative_fro_error','push_relative_fro_error','push_diag_relative_rms_error','push_diag_relative_sd_error','rank')}} for r in rows]},indent=2))

if __name__=='__main__': main()
