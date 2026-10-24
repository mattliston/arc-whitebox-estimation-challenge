"""One original CO CPU forward; save its existing FP64 return before serving.

Inert without --run. No native-G0 injection, model/source modification,
gradient, optimizer, target access, parent rerun, or derivative qualification.
The shared Stage authenticates all inputs before numerical imports.
"""
import hashlib
import math
import os
from pathlib import Path
import sys
import time
import traceback

import common as c


def require(value, message):
    if not value:
        raise RuntimeError(message)


def main():
    args = c.arguments('cpu')
    if not args.run:
        return 0
    stage = c.Stage('cpu', args, Path(__file__))
    h, model = stage.h, stage.model
    result = stage.result
    try:
        require(Path(sys.prefix).resolve() == Path('/Users/matt/anaconda3'), 'Exact AP interpreter')
        require(os.getpriority(os.PRIO_PROCESS, 0) >= 10, 'nice10 CPU singleton')
        require(all(os.environ.get(key) == '1' for key in
                    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                     'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')),
                'Explicit inherited BLAS1/no-bytecode environment')
        require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') == '0', 'No MPS fallback')
        require(not any(key in sys.modules for key in ('numpy', 'torch', 'flopscope', 'threadpoolctl')),
                'Complete Stage authentication must precede numerical imports')
        import numpy as np
        import torch
        require(np.__version__ == '1.26.4' and torch.__version__ == '2.7.0', 'Original CPU library frame')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        result.update(environment={'versions': {'numpy': np.__version__, 'torch': torch.__version__},
                                   'device': 'cpu', 'moment_dtype': 'torch.float64'},
                      network_calls=0, gradient_calls=0, parameter_updates=0,
                      training=False, MPS_used=False, native_G0_injected=False,
                      truth_accessed=False, parent_reruns=0, weights_decoded=True,
                      mlp_seed=stage.mlp_seed, raw_capture_is_existing_return=True,
                      new_derivative_qualified=False, training_admitted=False)
        payload = stage.old_cpu['payloads'][args.case]
        stage.pin(payload['path'], payload['sha256'])
        values = h.load_bundle(np, payload)
        require(all(value.dtype == np.float32 for value in values.values()), 'Sealed full FP32 payload')
        payload_before = {key: h.array_meta(value) for key, value in values.items()}
        result['payload'] = payload
        result['payload_arrays_before'] = payload_before
        cotangent = stage.load(np, stage.old_cpu['cotangents'][str(args.rank)])
        reference = stage.load(np, stage.reference_cpu['prediction'])
        require(cotangent.shape == reference.shape == (16, 1024)
                and cotangent.dtype == reference.dtype == np.float32, 'Original served/cotangent frame')
        weights = np.load(stage.weights_path, mmap_mode='r', allow_pickle=False)
        require(weights.shape == (16, 1024, 1024) and weights.dtype == np.float32,
                'Canonical weight shape/precision')
        P = h.torch_payload(torch, values, model.PKEYS, 'cpu', gradients=False)
        require(set(P) == set(values) and not any(value.requires_grad for value in P.values()),
                'Entire frozen payload, no trainable leaves')
        stage.guard()
        card = model.PCard(args.rank, P=P, device='cpu', weights=weights, audit_selections=True)
        require(card.dtype == torch.float64 and card.device.type == 'cpu', 'Original CPU moment arithmetic')
        require(card.theta is None and not card.coefficients.requires_grad, 'Frozen deployment cstar')
        cstar_before = card.coefficients.detach().cpu().numpy().tobytes()
        require(cstar_before == model.coefficient_bytes(model.frozen_coefficients()), 'Exact original cstar')
        require(all(card.P0[key] is P[key] for key in P), 'Original supplied payload ownership')
        result['network_calls'] = 1
        started = time.monotonic()
        with torch.no_grad():
            prediction, capture = card.run(training=False)
            require(prediction.shape == (16, 1024) and prediction.dtype == torch.float64
                    and prediction.device.type == 'cpu' and not prediction.requires_grad,
                    'Existing CPU FP64 return contract')
            raw = prediction.detach().cpu().numpy().copy()
            served = h.prediction_array(np, model, prediction)
            loss = float(h.scalar(torch, model, prediction, cotangent, card.dtype).detach().cpu())
        result['forward_elapsed_s'] = time.monotonic() - started
        require(np.isfinite(raw).all() and math.isfinite(loss), 'Finite captured return and original scalar')
        # Match the ORIGINAL served helper's sign rule, preserving signed zero.
        # The clipping is capture-only: neither raw nor the card state is written.
        clipped = np.where(raw < 0, np.zeros_like(raw), raw)
        result['raw_unclamped'] = stage.save(np, 'raw_unclamped.npy', raw)
        result['pre_return'] = stage.save(np, 'pre_return.npy', clipped)
        result['served'] = stage.save(np, 'served.npy', served)
        result['prediction'] = result['served']
        result['synthetic_loss'] = loss
        result['clipping_count'] = int((raw < 0).sum())
        result['clipping_count_by_layer'] = [int((row < 0).sum()) for row in raw]
        result['served_reconstruction_byte_exact'] = clipped.astype(np.float32).tobytes() == served.tobytes()
        result['old_cpu_served_byte_exact'] = served.tobytes() == reference.tobytes()
        result['old_cpu_scalar_exact'] = loss == stage.reference_cpu['synthetic_loss']
        diagnostics = model.check_replay(card)
        result['diagnostics'] = diagnostics
        result['old_cpu_diagnostics_exact'] = diagnostics == stage.reference_cpu['diagnostics']
        result['all16_domains_present'] = set(diagnostics['variance_domain']) == {str(l) for l in range(16)}
        result['all47_selectors_present'] = set(diagnostics['selection_cells']) == {
            f'{l}:{slot}' for l in range(16) for slot in range(2 if l == 15 else 3)}
        payload_after = {key: h.array_meta(value.detach().cpu().numpy()) for key, value in P.items()}
        result['payload_arrays_after'] = payload_after
        result['all_payload_bytes_unchanged'] = payload_after == payload_before
        result['frozen_cstar_bytes_unchanged'] = card.coefficients.detach().cpu().numpy().tobytes() == cstar_before
        result['frozen_cstar_raw_sha256'] = hashlib.sha256(cstar_before).hexdigest()
        require(result['served_reconstruction_byte_exact'] and result['old_cpu_served_byte_exact']
                and result['old_cpu_scalar_exact'], 'Exact original served return/scalar gates')
        require(result['old_cpu_diagnostics_exact'] and result['all16_domains_present']
                and result['all47_selectors_present'], 'Exact original domain/ordered-selector diagnostics')
        require(result['all_payload_bytes_unchanged'] and result['frozen_cstar_bytes_unchanged']
                and not any(value.requires_grad or value.grad is not None for value in P.values()),
                'No parameter or gradient mutation')
        require(not any(key in sys.modules for key in ('flopscope', 'threadpoolctl')),
                'No native numerical or threadpool-controller imports')
        del card, prediction, capture, P, values, weights
        h.release(torch, 'cpu')
        stage.guard()
        stage.finish()
    except BaseException:
        result.update(passed=False, completed=False, error=traceback.format_exc())
    finally:
        stage.close()
    return 0 if result.get('passed') and result.get('completed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
