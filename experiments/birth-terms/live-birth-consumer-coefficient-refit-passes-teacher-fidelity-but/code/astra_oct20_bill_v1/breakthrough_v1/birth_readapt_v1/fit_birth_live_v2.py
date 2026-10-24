"""One frozen teacher-only birth-readout step; no network truth is opened."""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path('/home/arcworker/arc-runs')
CURRENT = '777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'
TEACHER = '09fb70c86fa245d083f1db5d9c06dcde90cd71cbc8aaeaabb1bcb134aef59ded'
RANKS = (35, 150, 200, 250)
RIDGE, DAMPING = .1, .5
ACTIVE = (0, 2)  # The only two coefficient channels consumed by live _bd_update.
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
utc = lambda: subprocess.check_output(['date', '-u', '+%Y-%m-%dT%H:%M:%SZ'], text=True).strip()


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def literal(path, name):
    values = [ast.literal_eval(n.value) for n in ast.parse(path.read_text()).body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    assert len(values) == 1
    return np.asarray(values[0], dtype=np.float32)


def normalized(H):
    H = np.asarray(H, dtype=np.float32)
    assert H.shape == (1024, 64)
    centered = H - H.mean(axis=0, dtype=np.float32)
    scale = np.sqrt((centered * centered).mean(axis=0, dtype=np.float32) + np.float32(1e-6))
    return np.clip(centered / scale, -8, 8)


def guard_ok(path):
    g = read(path)
    assert g['successful'] and g['child_exit'] == 0 and g['whole_group_retired']
    assert not g['cleanup_errors'] and g['max_owned_swap_bytes'] == 0
    assert g['peak_group_rss_bytes'] < g['cap_bytes']
    if 'cgroup_events_final' in g:
        assert all(g['cgroup_events_final'].get(k, 0) == 0 for k in ('max', 'oom', 'oom_kill'))
    return sha(path)


def source_tables():
    k = HERE.parents[2] / 'fable_oct19_kc_v1'
    v = HERE.parents[2] / 'astra_oct20_vm_frontier_v1'
    old = k / 'candidate_micro_copyall_ecut_eh14_astra_sci_v1.py'
    current = v / 'msg8_loaderfix_v1/candidate_msg8_teacher_loaderfix_sci_v1.py'
    assert sha(old) == TEACHER and sha(current) == CURRENT
    A = literal(current, '_BDNH_A_LITERAL')
    c = literal(current, '_BD_COEFFICIENT_LITERAL')
    assert A.shape == (64, 4) and c.shape == (15, 4)
    np.testing.assert_array_equal(A, literal(old, '_BDNH_A_LITERAL'))
    np.testing.assert_array_equal(c, literal(old, '_BD_COEFFICIENT_LITERAL'))
    return A.astype(np.float64), c


def load_data():
    birth = RUNS / 'birth_consumer_train4_v1'
    state = RUNS / 'full_state_response_train4_v1'
    assert read(birth / 'COMPLETE.json')['successful'] and read(state / 'COMPLETE.json')['successful']
    A, cstars = source_tables()
    old_records, current_records, proof = {}, {}, []
    for base, source, folder, target in ((birth, TEACHER, 'birth_consumer_v1', old_records),
                                        (state, CURRENT, 'gnn_full_state_v1', current_records)):
        for part in sorted(base.glob('part[0-9][0-9]')):
            gh = guard_ok(base / (part.name + '_GUARD.json'))
            assert read(part / 'manifest.json')['sha256'] == source
            done = read(part / folder / 'CAPTURE_COMPLETE_v1.json')
            assert done['complete'] and done['target_free'] and done['source_sha256'] == source
            ranks = done.get('ranks') or [r for r, phase in done['groups'] if phase == 'scored']
            assert all(r in RANKS for r in ranks)
            for rank in ranks:
                pred = part / ('diagnostic_pred_%d.bin' % rank)
                refbase = RUNS / ('teacher_capture_v1' if source == TEACHER else
                                 'msg8_loaderfix337422_control32_markline_blas4_v1')
                refs = list(refbase.glob('part*/pred_%d.bin' % rank))
                assert len(refs) == 1 and pred.stat().st_size == 65536
                assert pred.read_bytes() == refs[0].read_bytes()
                proof.append(dict(rank=rank, source_sha256=source, prediction_sha256=sha(pred),
                                  control_path=str(refs[0]), guard_sha256=gh, parity='BYTE_EXACT'))
            for item in done['files']:
                path = part / folder / item['path']
                assert sha(path) == item['sha256']
                if source == TEACHER:
                    rec = read(path)
                else:
                    with gzip.open(path, 'rt') as f:
                        rec = json.load(f)
                key = (int(rec['rank']), int(rec['layer']))
                assert key not in target and rec['phase'] == 'scored'
                if key[1] <= 14:
                    target[key] = (path, rec)
    expected = {(r, l) for r in RANKS for l in range(1, 15)}
    assert set(old_records) == set(current_records) == expected
    blocks, alignment = [], []
    for rank, layer in sorted(expected):
        path, old = old_records[(rank, layer)]
        def array(name):
            rec = old[name]
            f = path.parent / rec['path']
            assert sha(f) == rec['sha256'] and f.stat().st_size == rec['bytes']
            result = np.fromfile(f, dtype=np.dtype(rec['dtype'])).reshape(rec['shape'])
            assert np.isfinite(result).all()
            return result
        oldH, oldZ, native_response = array('state'), array('standardized'), array('response')
        target = array('coefficients').T
        cstar, onem = array('cstar'), array('onem')
        np.testing.assert_array_equal(cstar, cstars[layer])
        np.testing.assert_array_equal(onem, np.float32(1) - np.abs(cstar))
        np.testing.assert_allclose(normalized(oldH), oldZ, rtol=2e-5, atol=2e-5)
        np.testing.assert_allclose(cstar + onem * native_response, target, rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(np.tanh(oldZ.astype(np.float64) @ A), native_response, rtol=3e-5, atol=2e-6)
        current_path, current = current_records[(rank, layer)]
        H = np.asarray(current['state']['data'], dtype=np.float32)
        assert current['state']['shape'] == [1024, 64]
        Z = normalized(H).astype(np.float64)
        blocks.append(dict(rank=rank, layer=layer, Z=Z, target=target.astype(np.float64),
                           cstar=cstar.astype(np.float64), onem=onem.astype(np.float64)))
        alignment.append(dict(rank=rank, layer=layer, teacher_record_sha256=sha(path),
                              current_record_sha256=sha(current_path), rows=1024,
                              old_normalization_max_abs=float(np.max(np.abs(normalized(oldH) - oldZ)))))
    return A, blocks, proof, alignment


def coefficients(block, A):
    return block['cstar'] + block['onem'] * np.tanh(block['Z'] @ A)


def statistics(blocks, A):
    result = {}
    for rank in sorted({b['rank'] for b in blocks}):
        G = np.zeros((4, 64, 64), dtype=np.float64)
        rhs = np.zeros((4, 64), dtype=np.float64)
        chosen = [b for b in blocks if b['rank'] == rank]
        for b in chosen:
            z = b['Z']
            response = np.tanh(z @ A)
            residual = b['target'] - (b['cstar'] + b['onem'] * response)
            derivative = b['onem'] * (1 - response ** 2)
            for j in range(4):
                X = z * derivative[:, j, None]
                G[j] += X.T @ X / len(z) / len(chosen)
                rhs[j] += X.T @ residual[:, j] / len(z) / len(chosen)
        result[rank] = (G, rhs)
    return result


def step(stats, included, A):
    G = sum(stats[r][0] for r in included) / len(included)
    rhs = sum(stats[r][1] for r in included) / len(included)
    change = np.zeros_like(A)
    conditions = []
    for j in ACTIVE:
        ridge = RIDGE * np.trace(G[j]) / 64
        assert ridge > 0
        regularized = G[j] + ridge * np.eye(64)
        change[:, j] = DAMPING * np.linalg.solve(regularized, rhs[j])
        conditions.append(float(np.linalg.cond(regularized)))
    return A + change, conditions


def evaluate(blocks, old, new):
    result = []
    for rank in sorted({b['rank'] for b in blocks}):
        before, after, moved, energy = [], [], [], []
        for b in blocks:
            if b['rank'] != rank:
                continue
            x, y = coefficients(b, old)[:, ACTIVE], coefficients(b, new)[:, ACTIVE]
            target = b['target'][:, ACTIVE]
            before.append(float(np.mean((x - target) ** 2)))
            after.append(float(np.mean((y - target) ** 2)))
            moved.append(float(np.mean((y - x) ** 2)))
            energy.append(float(np.mean(x ** 2)))
        a, b = float(np.mean(before)), float(np.mean(after))
        assert a > 0 and np.isfinite(b)
        result.append(dict(rank=rank, before_mse=a, after_mse=b, relative_change_pct=100 * (b / a - 1),
                           coefficient_step_rms=float(np.sqrt(np.mean(moved))),
                           coefficient_baseline_rms=float(np.sqrt(np.mean(energy))),
                           relative_coefficient_step=float(np.sqrt(np.mean(moved) / np.mean(energy)))))
    return result


def selftest():
    rng = np.random.default_rng(20261021)
    A = rng.normal(scale=.03, size=(64, 4))
    Z = rng.normal(size=(1024, 64))
    c = np.array([.3, -.2, .4, .1])
    block = dict(Z=Z, cstar=c, onem=1 - np.abs(c))
    direction = rng.normal(size=A.shape)
    eps = 1e-6
    analytic = block['onem'] * (1 - np.tanh(Z @ A) ** 2) * (Z @ direction)
    finite = (coefficients(block, A + eps * direction) - coefficients(block, A - eps * direction)) / (2 * eps)
    error = float(np.max(np.abs(analytic - finite)))
    assert error < 1e-7
    H = rng.normal(size=(1024, 64)).astype(np.float32)
    z = normalized(H)
    assert np.max(np.abs(z.mean(axis=0))) < 1e-6 and np.max(np.abs(z.var(axis=0) - 1)) < 3e-6
    blocks = []
    true = A + rng.normal(scale=.002, size=A.shape)
    for rank in RANKS:
        z = rng.normal(size=(1024, 64))
        b = dict(rank=rank, layer=1, Z=z, cstar=c, onem=1 - np.abs(c))
        b['target'] = coefficients(b, true)
        blocks.append(b)
    stats = statistics(blocks, A)
    fitted, _ = step(stats, RANKS, A)
    checks = evaluate(blocks, A, fitted)
    assert all(x['relative_change_pct'] < -50 for x in checks)
    return dict(status='PASS', derivative_max_abs_error=error, synthetic_fit=checks,
                all1024_normalization_checked=True, native_prediction_launches=0, truth_access=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--selftest', action='store_true')
    a = p.parse_args()
    assert os.environ.get('ARC_BLAS_THREADS') in ('1', '2')
    a.out.mkdir(parents=True, exist_ok=False)
    check = selftest()
    write(a.out / 'SELFTEST_v1.json', dict(utc=utc(), source_sha256=sha(__file__), **check))
    if a.selftest:
        print(json.dumps(check))
        return
    started = time.monotonic()
    A, blocks, proof, alignment = load_data()
    stats = statistics(blocks, A)
    loo = []
    for held in RANKS:
        fitted, conditions = step(stats, [r for r in RANKS if r != held], A)
        report = evaluate([b for b in blocks if b['rank'] == held], A, fitted)[0]
        report['fit_ranks'] = [r for r in RANKS if r != held]
        report['regularized_conditions'] = conditions
        loo.append(report)
    fitted, conditions = step(stats, RANKS, A)
    # Assess exactly the proposed deployed float32 constants.
    fitted = fitted.astype(np.float32).astype(np.float64)
    training = evaluate(blocks, A, fitted)
    mean_loo = float(np.mean([x['relative_change_pct'] for x in loo]))
    wins = sum(x['relative_change_pct'] < 0 for x in loo)
    max_step = max(x['relative_coefficient_step'] for x in training + loo)
    accepted = mean_loo <= -1 and wins >= 3 and max_step <= .10 and np.isfinite(fitted).all()
    result = dict(utc=utc(), status='PASS_TEACHER_FIDELITY_GATE' if accepted else 'REJECT_BEFORE_NATIVE',
                  reference_submission=337422, source_sha256=sha(__file__), current_source_sha256=CURRENT,
                  active_coefficient_columns=list(ACTIVE), dead_columns_unchanged=[1, 3],
                  teacher_source_sha256=TEACHER, train=list(RANKS), layers=list(range(1, 15)),
                  rows_per_layer=1024, ridge=RIDGE, damping=DAMPING, one_Gauss_Newton_step=True,
                  ridge_formula='lambda=.1*trace(G)/64, isotropic64x64, after equal-network/layer mean',
                  mean_loo_relative_coefficient_error_change_pct=mean_loo, loo_wins=wins,
                  max_relative_coefficient_step=max_step, loo=loo, training=training,
                  regularized_conditions=conditions, original_A=A.tolist(), fitted_A=fitted.tolist(),
                  guards_and_parity=proof, alignment=alignment, elapsed_s=time.monotonic() - started,
                  truth_access=False, fresh_or_held_targets_used=False, native_candidate_launches=0,
                  adjusted_score_gain=None, note='Teacher LIVE coefficient fidelity only (columns0,2); native accuracy unknown.')
    write(a.out / 'FIT_RESULT_v1.json', result)
    print(json.dumps({k: result[k] for k in ('status', 'mean_loo_relative_coefficient_error_change_pct',
                                           'loo_wins', 'max_relative_coefficient_step', 'elapsed_s')}))


if __name__ == '__main__':
    main()
