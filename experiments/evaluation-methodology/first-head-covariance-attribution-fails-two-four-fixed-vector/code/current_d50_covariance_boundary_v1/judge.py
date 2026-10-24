"""Inert fixed covariance-circuit attribution, not a model/score qualification.

The original scalar helper sees only row1 of each nonnegative individual
output, with all other rows zero, BEFORE subtraction. It never receives a
signed response or an irrelevant constant row0 contribution.
Only L enters the prospective gate; R is a separately labelled diagnostic.
"""
import argparse
import gc
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    for name in ('source', 'cpu-source', 'cpu-result', 'native-source',
                 'native-result', 'controls-result', 'protocol'):
        p.add_argument('--' + name + '-sha256')
    args = p.parse_args()
    if not args.run:
        p.print_help(); return 0
    if any(not v or len(v) != 64 for k, v in vars(args).items() if k != 'run'):
        p.error('All reviewed SHA256 arguments are required')
    cp = HERE / 'cpu.py'
    if hashlib.sha256(cp.read_bytes()).hexdigest() != args.cpu_source_sha256:
        raise RuntimeError('Reviewed CPU helper changed')
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location('covariance_boundary_cpu_helpers', cp)
    c = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = c
    spec.loader.exec_module(c)
    a = c.Stage(Path(__file__), args.source_sha256, args.protocol_sha256, 'judge', 'judgment_v1')
    h, r = a.h, a.result
    try:
        a.environment()
        a.pin(cp, args.cpu_source_sha256, source=True)
        native = a.receipt('native_n_v1', args.native_result_sha256, 'n',
                           HERE / 'native.py', args.native_source_sha256)
        cpu = a.receipt('cpu_v1', args.cpu_result_sha256, 'cpu', cp, args.cpu_source_sha256)
        controls = a.receipt('native_controls_v1', args.controls_result_sha256, 'controls',
                             HERE / 'native.py', args.native_source_sha256)
        h.require(native['prefix_calls'] == 11 and native['native_baseline_reproduces_all_old_two_rows']
            and native['repeat_byte_exact'] and controls['prefix_calls'] == 22
            and controls['repeat_byte_exact'], 'All native baselines/controls and deterministic repeats')
        h.require(cpu['native_n_result_sha256'] == args.native_result_sha256
            and cpu['all_fixed_arrays_sealed'] and cpu['card_constructions'] == 10
            and cpu['recurrent_steps'] == 20, 'Complete CPU case roster and same N owner')
        h.require(controls['input_bindings'][str(HERE / 'cpu_v1/result.json')] == args.cpu_result_sha256
            and controls['source_bindings'][str(cp)] == args.cpu_source_sha256,
            'Controls consume the sealed CPU L/R circuits')
        model, probes, old = c.authenticate_old(a)
        own = model.owner()
        h.require(cpu['payloads'] == probes['payloads'] and cpu['cotangents'] == probes['cotangents']
                  and cpu['probe_seal_sha256'] == probes['probe_seal_sha256'], 'Same original probes/cotangents')
        for record in (native, cpu, controls):
            h.require(set(record['ranks']) == {'35', '100'}, 'Both fixed ranks completed')
        # Complete inventories above are authenticated before ANY array import/read.
        h.require(not any(k in sys.modules for k in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                  'All source/artifact seals precede numerical imports')
        import numpy as np
        import torch
        h.require(dict(numpy=np.__version__, torch=torch.__version__) == h.VERSIONS, 'Original AP scalar frame')
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        r.update(environment=dict(versions=h.VERSIONS, device='cpu'),
            stage_seals=dict(native_n=args.native_result_sha256, cpu=args.cpu_result_sha256,
                             native_controls=args.controls_result_sha256),
            every_input_sealed_before_array_reads=True, measurements=[],
            numerical_gates_passed=False, information_gate_passed=False,
            automatic_port_admitted=False, training_admitted=False, score_measured=False)

        def outputs(entry, arm):
            y, host, delta = (a.load(np, entry[k]) for k in ('prediction', 'host1', 'delta1'))
            c.valid_prediction(h, np, y)
            h.require(arm in ('N', 'L', 'R', 'CPU'), 'Fixed output arm')
            delta_dtype = np.float64 if arm == 'CPU' else np.float32
            h.require(host.shape == delta.shape == (1024,) and host.dtype == np.float64
                      and delta.dtype == delta_dtype, 'Actual arm-specific host/correction dtype')
            return y, host, delta

        def scalar(y, cot):
            # Do not clamp a signed difference: evaluate the two individual
            # served row1 outputs, retaining the exact original helper/reduction.
            # Row0 is excluded rather than relying on rounded cancellation.
            padded = torch.zeros((16, 1024), dtype=torch.float32)
            padded[1] = torch.from_numpy(y[1])
            value = float(h.scalar(torch, own, padded, cot, torch.float64))
            h.require(math.isfinite(value), 'Finite original scalar')
            return value

        def norm(x):
            return float(np.linalg.norm(x))

        def reduction(D, E, ds, es):
            dn, rem = norm(D), norm(D - E)
            ratio = rem / max(dn, 1e-10)
            sr = abs(ds - es) / max(abs(ds), 1e-10)
            return dict(discrepancy_l2=dn, effect_l2=norm(E), remaining_l2=rem,
                remaining_norm_ratio=ratio, scalar_discrepancy=ds, scalar_effect=es,
                scalar_remaining=ds - es, remaining_absolute_scalar_ratio=sr,
                signed_projection=float(np.dot(D, E)),
                projection_fraction=float(np.dot(D, E)) / max(float(np.dot(D, D)), 1e-20),
                cosine=float(np.dot(D, E)) / max(dn * norm(E), 1e-20),
                nontrivial=dn > 1e-10 and abs(ds) > 1e-10,
                passed=dn > 1e-10 and abs(ds) > 1e-10 and ratio <= .5 and sr <= .5)

        for rank in c.RANKS:
            rk = str(rank)
            cot = h.load_array(np, probes['cotangents'][rk])
            h.require(cot.shape == (16, 1024) and cot.dtype == np.float32, 'Original full cotangent')
            collections = dict(N=native['ranks'][rk]['outputs'], CPU=cpu['ranks'][rk]['outputs'])
            h.require(set(controls['ranks'][rk]['circuits']) == {'L', 'R'}, 'Both fixed native controls')
            for label in ('L', 'R'):
                collections[label] = controls['ranks'][rk]['circuits'][label]['outputs']
                for key in ('Cpre', 'Bpre'):
                    item = cpu['ranks'][rk]['circuits'][label][key]
                    h.require(controls['input_bindings'][item['path']] == item['sha256'], 'Exact common-input circuit supplied')
            vals, losses = {}, {}
            for label, records in collections.items():
                h.require(set(records) == {case for case, _ in c.CASES}, 'All five fixed cases')
                vals[label], losses[label] = {}, {}
                for case, amplitude in c.CASES:
                    v = outputs(records[case], label)
                    vals[label][case] = v
                    losses[label][case] = scalar(v[0], cot)
                    # No causal P effect is permitted in L0 or pre-head host1.
                    if case != 'null':
                        h.require(v[0][0].tobytes() == vals[label]['null'][0][0].tobytes()
                            and v[1].tobytes() == vals[label]['null'][1].tobytes(),
                            'P-independent L0 and pre-head host1 within each arm')
            for label in ('L', 'R', 'CPU'):
                h.require(vals[label]['null'][0][0].tobytes() == vals['N']['null'][0][0].tobytes(),
                          'Native-G0 common served L0')
            r['ranks'][rk] = dict(individual_original_scalars=losses,
                null_host_drift={label: c.compare(np, vals['N']['null'][1], vals[label]['null'][1])
                                for label in ('CPU', 'L', 'R')},
                null_delta_drift={label: c.compare(np, vals['N']['null'][2], vals[label]['null'][2])
                                 for label in ('CPU', 'L', 'R')})
            for suffix, amp in (('025', .025), ('050', .05)):
                plus, minus = 'plus' + suffix, 'minus' + suffix
                responses = {key: (v[plus][0][1].astype(np.float64) - v[minus][0][1].astype(np.float64)) / (2 * amp)
                             for key, v in vals.items()}
                scalars = {key: (v[plus] - v[minus]) / (2 * amp) for key, v in losses.items()}
                D, E = responses['N'] - responses['CPU'], responses['N'] - responses['L']
                ds, es = scalars['N'] - scalars['CPU'], scalars['N'] - scalars['L']
                entry = dict(rank=rank, amplitude=amp, L=reduction(D, E, ds, es),
                    R_diagnostic_only=reduction(D, responses['N'] - responses['R'], ds, scalars['N'] - scalars['R']),
                    response_arrays={key: a.save(np, f'row_{rank:04d}_{suffix}_{key}.npy', value)
                                     for key, value in dict(D=D, E=E, D_minus_E=D-E, **responses).items()},
                    original_individual_scalar_secants=scalars,
                    raw_host_response_l2={}, raw_delta_response_l2={}, served_minus_raw_response_l2={})
                for key, v in vals.items():
                    hr = (v[plus][1] - v[minus][1]) / (2 * amp)
                    dr = (v[plus][2].astype(np.float64) - v[minus][2].astype(np.float64)) / (2 * amp)
                    entry['raw_host_response_l2'][key] = norm(hr)
                    entry['raw_delta_response_l2'][key] = norm(dr)
                    entry['served_minus_raw_response_l2'][key] = norm(responses[key] - hr - dr)
                r['measurements'].append(entry)
                a.attempt(f'row_{rank:04d}_{suffix}_attempt.json', entry)
                # Both amplitudes and both ranks are measured, without early
                # scientific rejection, tuning, additional cases, or R selection.
                h.require(all(np.isfinite(x).all() for x in responses.values()), 'Finite paired responses')
            a.guard()
            del cot, vals, losses
            gc.collect()
        h.require(len(r['measurements']) == 4 and not any(k in sys.modules for k in ('flopscope', 'threadpoolctl')),
                  'Complete fixed local attribution scope')
        r['numerical_gates_passed'] = True
        r['information_gate_passed'] = all(x['L']['passed'] for x in r['measurements'])
        r['bounded_covariance_implementation_may_be_considered'] = r['information_gate_passed']
        # A scientific FAIL still has a completed, authenticated acquisition.
        a.finish(True)
        r['passed'] = r['information_gate_passed']
    except BaseException:
        r.update(passed=False, completed=False, hashes_unchanged=False, sources_unchanged=False,
                 numerical_gates_passed=False, information_gate_passed=False,
                 bounded_covariance_implementation_may_be_considered=False,
                 error=traceback.format_exc())
    finally:
        a.close()
    return 0 if r['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
