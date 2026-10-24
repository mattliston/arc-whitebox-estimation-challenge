"""Root-only fixed terminal65 surrogate ridge; no estimator or full forward is run."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

D = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text())


def pin(p):
    p = Path(p).resolve()
    data = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def verify(p):
    assert pin(p['path']) == p, p['path']


def once(name, value):
    with (D / name).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    started = time.monotonic()
    protocol_pin = pin(D / 'fit_protocol.json')
    assert protocol_pin['sha256'] == args.protocol_sha
    p = read(protocol_pin['path'])
    assert p['root_admitted'] and p['ridge'] == 1e-3 and p['fit_count'] == 1
    assert p['fit_diagnostics_can_admit'] is False
    for value in p['pins'].values():
        verify(value)
    assert Path(sys.executable).resolve() == Path(p['runtime']['path']).resolve()
    verify(p['runtime'])
    for key in p['thread_env']:
        assert os.environ.get(key) == '1', key
    index = read(p['pins']['data_index']['path'])
    verify(index['baseline'])
    verify(index['cached_source'])
    for value in index['authorities'].values():
        verify(value)
    authority = read(index['authorities']['train160_protocol']['path'])
    assert index['ranks'] == authority['training_ranks']
    assert [r['rank'] for r in index['rows']] == index['ranks']
    assert len(index['ranks']) == len(set(index['ranks'])) == 160
    assert [r['truth'] for r in index['rows']] == [r['truth'] for r in authority['data']]
    for value in index['capture_evidence']:
        verify(value['completion'])
        verify(value['guard'])
        g = read(value['guard']['path'])
        assert g['completed'] and g['whole_group_retired'] and g['exit_code'] == 0
        assert not g['remaining_rows'] and g['max_swap_growth_bytes'] == 0
        c = read(value['completion']['path'])
        assert not c['targets_accessed'] and c['parent_sha256'] == index['cached_source']['sha256']
    spec = importlib.util.spec_from_file_location('fixed_head160_source_editor', p['pins']['source_editor']['path'])
    editor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(editor)
    baseline_source = Path(index['baseline']['path']).read_text()
    before = editor.decoded_payload(baseline_source)
    cached_payload = editor.decoded_payload(Path(index['cached_source']['path']).read_text())
    assert before == cached_payload and len(before) == 15

    # Numerical work starts only under root's dedicated 120s/7.5GB guard.
    import numpy as np
    from threadpoolctl import threadpool_info
    assert all(t['num_threads'] == 1 for t in threadpool_info())

    def solve(B, y, w):
        w = w / w.sum()
        scale = np.sqrt(np.sum(w[:, None] * B * B, axis=0))
        assert np.isfinite(scale).all() and np.min(scale) > 1e-20
        X = B / scale
        gram = X.T @ (w[:, None] * X) + p['ridge'] * np.eye(B.shape[1])
        rhs = X.T @ (w * y)
        coefficient = np.linalg.solve(gram, rhs)
        residual = float(np.linalg.norm(gram @ coefficient - rhs) / max(np.linalg.norm(rhs), 1e-30))
        assert np.isfinite(coefficient).all() and residual < 1e-9
        return coefficient / scale, scale, dict(normal_equation_relative_error=residual,
                                               condition=float(np.linalg.cond(gram)))

    rng = np.random.Generator(np.random.PCG64DXSM(2026092062))
    tinyB = rng.normal(size=(113, 7))
    tinyy = rng.normal(size=113)
    tinyw = np.linspace(.1, 1, 113)
    got, tiny_scale, certificate = solve(tinyB, tinyy, tinyw)
    wn = tinyw / tinyw.sum()
    independent = np.linalg.lstsq(
        np.vstack((np.sqrt(wn)[:, None] * tinyB / tiny_scale,
                   np.sqrt(p['ridge']) * np.eye(7))),
        np.r_[np.sqrt(wn) * tinyy, np.zeros(7)], rcond=None)[0] / tiny_scale
    assert np.allclose(got, independent, rtol=1e-10, atol=1e-12)

    rms = np.frombuffer(before['rms'][1], dtype='<f4').reshape(before['rms'][0])[15]
    gain = np.frombuffer(before['gain'][1], dtype='<f4').reshape(before['gain'][0])[15]
    gain_factor = (np.float32(1) + np.float32(.25) * np.tanh(np.asarray([gain], dtype=np.float32)))[0]
    old = np.asarray(editor.terminal_head(baseline_source), dtype=np.float64)
    assert old.shape == (65,) and np.isfinite(old).all()
    Bs, residuals, weights, packets = [], [], [], []
    for row in index['rows']:
        verify(row['packet'])
        verify(row['truth'])
        with np.load(row['packet']['path'], allow_pickle=False) as z:
            control = z['control']
            sigma = z['sigma']
            prediction = z['prediction']
            assert control.dtype == np.float32 and control.shape == (1024, 94)
            assert sigma.shape == (1024,) and prediction.shape == (16, 1024)
            assert np.isfinite(control).all() and np.isfinite(sigma).all() and np.isfinite(prediction).all()
            assert (sigma > 0).all()
            hidden = control[:, 30:]
            physical_scale = (sigma.astype(np.float32) * np.float32(rms)).astype(np.float64) * np.float64(gain_factor)
            B = physical_scale[:, None] * np.column_stack((hidden, np.ones(1024))).astype(np.float64)
            parent = prediction[-1].astype(np.float64)
        with np.load(row['truth']['path'], allow_pickle=False) as z:
            target = z['mc_post']
            assert target.shape == (16, 1024) and np.isfinite(target).all()
            truth = target[-1].astype(np.float64)
        assert np.isfinite(B).all()
        multiplier = max(.1, row['flops'] / 2**41)
        Bs.append(B)
        residuals.append(truth - parent)
        weights.append(np.full(1024, multiplier))
        packets.append(dict(rank=row['rank'], parent=parent, truth=truth, multiplier=multiplier))
    B = np.concatenate(Bs)
    y = np.concatenate(residuals)
    w = np.concatenate(weights)
    correction, scales, equations = solve(B, y, w)
    head = (old + correction).astype(np.float32)
    assert head.shape == (65,) and np.isfinite(head).all()
    candidate, disabled, patches, literal_old = editor.edit(baseline_source, head)
    assert disabled == Path(index['baseline']['path']).read_bytes()
    assert np.array_equal(np.asarray(literal_old), old)
    after = editor.decoded_payload(candidate)
    for key, (shape, raw) in before.items():
        assert after[key][0] == shape
        prefix = 15 * 64 * 4 if key == 'head' else 15 * 4 if key == 'head_b' else len(raw)
        assert after[key][1][:prefix] == raw[:prefix]
    assert editor.terminal_head(candidate) == [float(v) for v in head]
    compile(candidate, str(D / 'candidate.py'), 'exec')
    with (D / 'model.npz').open('xb') as f:
        np.savez(f, head=head, old=old, correction=correction, column_scale=scales)
    with (D / 'candidate.py').open('xb') as f:
        f.write(candidate)
    with (D / 'disabled_control.py').open('xb') as f:
        f.write(disabled)
    diagnostics = []
    rounded_correction = head.astype(np.float64) - old
    for basis, row in zip(Bs, packets):
        # Historical, clipped-parent surrogate only; this is NOT native replay.
        proxy = np.maximum(row['parent'] + basis @ rounded_correction, 0).astype(np.float32)
        diagnostics.append(dict(rank=row['rank'], parent_mse=float(np.mean((row['parent'] - row['truth'])**2)),
                                proxy_mse=float(np.mean((proxy.astype(np.float64) - row['truth'])**2)),
                                multiplier=row['multiplier'], not_an_advance_gate=True))
    result = dict(schema='cached-terminal65-fit-result-v1', completed=True, passed=True, fit_count=1,
                  protocol=protocol_pin, data_index=p['pins']['data_index'], train=index['ranks'], ranks=index['ranks'],
                  ridge=p['ridge'], validation_read=False,
                  baseline=index['baseline'], cached_source=index['cached_source'], candidate=pin(D / 'candidate.py'),
                  model=pin(D / 'model.npz'), disabled_control=pin(D / 'disabled_control.py'),
                  all15_parameter_fields_identical_before_fit=True, head_finite=True,
                  source_byte_inverse_exact=True, all_other_decoded_parameter_bytes_exact=True,
                  executable_ast_unchanged=True, allowed_parameter_count=65, added_inference_operations=0,
                  patches=patches, equations=equations, solver_certificate=certificate,
                  independent_solver_max_error=float(np.max(np.abs(got-independent))),
                  diagnostics=diagnostics, surrogate_proposal=True, native_readout_replay_certified=False,
                  clipped_parent_residual=True, fit_diagnostics_can_admit=False,
                  actual_native_score_measured=False, admitted=False,
                  elapsed_s=time.monotonic()-started, peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    once('fit_result.json', result)
    once('export_manifest.json', dict(schema='cached-terminal65-export-v1', passed=True,
         candidate=pin(D / 'candidate.py'), baseline=index['baseline'], cached_source=index['cached_source'],
         fit_result=pin(D / 'fit_result.json'), model=pin(D / 'model.npz'), protocol=protocol_pin,
         exporter=pin(Path(__file__).resolve()),
         data_index=p['pins']['data_index'], source_editor=p['pins']['source_editor'],
         disabled_control=pin(D / 'disabled_control.py'), source_byte_inverse_exact=True,
         all_other_decoded_parameter_bytes_exact=True, executable_ast_unchanged=True,
         allowed_parameter_count=65, surrogate_proposal=True, native_admission=False))
    print(json.dumps(dict(completed=True, candidate=pin(D / 'candidate.py'), equations=equations,
                          native_admission=False, fit_diagnostics_can_admit=False)), flush=True)


if __name__ == '__main__':
    main()
