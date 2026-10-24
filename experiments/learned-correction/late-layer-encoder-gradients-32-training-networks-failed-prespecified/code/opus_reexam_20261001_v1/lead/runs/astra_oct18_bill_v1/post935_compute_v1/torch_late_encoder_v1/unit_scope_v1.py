"""HELD tiny CPU wiring/zero-update test; no network, SDK, truths or fitting."""
import argparse
import ast
import base64
import json
import zlib
from pathlib import Path
from types import SimpleNamespace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    import torch
    from torch.utils.checkpoint import checkpoint
    from late_encoder_scope_v1 import LateEncoderScope, SHAPES
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    here = Path(__file__).resolve().parent
    plan = json.loads((here / 'SOURCE_PLAN_v1.json').read_text())
    tree = ast.parse(Path(plan['native_branch']).read_text())
    literal = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(
                       isinstance(target, ast.Name) and target.id == '_O18_LATE_ENCODER_LITERAL'
                       for target in node.targets))
    P = {key: torch.frombuffer(bytearray(zlib.decompress(base64.b64decode(blob))),
                              dtype=torch.float32).reshape(shape).clone()
         for key, (shape, blob) in literal.items()}
    M = SimpleNamespace(DUMP15=None, HEADU=torch.zeros(4, 128, requires_grad=True),
                        EDGEU=None, EDGEXS=None, RG_W={})
    E = SimpleNamespace(p=P, T={'p.' + key: value for key, value in P.items()},
                        DL={}, PL=SimpleNamespace(EPS={}), M=M,
                        UNFREEZE=False, LOSS_W=None, TRAINABLE=list(P))
    scope = LateEncoderScope(E, torch)
    x = torch.arange(4 * 126, dtype=torch.float32).reshape(4, 126) / 1000
    sigma = torch.ones(4)
    seen = []

    def gnn(l, x, sigma, W, state, selected):
        seen.append((l, selected is P, selected['enc0_w'] is scope.leaves['enc0_w']))
        # Fake HEADF outer wrapper, matching the real branch/bias order.
        selected = dict(selected, enc0_b=selected['enc0_b'][None, :] + M.HEADU)
        z = torch.nn.functional.silu(x @ selected['enc0_w'].T + selected['enc0_b'])
        z = torch.nn.functional.silu(z @ selected['enc2_w'].T + selected['enc2_b'])
        return z @ selected['enc4_w'].T + selected['enc4_b']

    cd = SimpleNamespace(P=P, env={'gnn_step': gnn,
                                   '_astra_current_w3_contract': lambda: P['enc0_w']})
    cd.step = lambda l, state: (cd.env['gnn_step'](l, state[0], sigma, None, None, P),)
    with torch.no_grad():
        references = [gnn(l, x, sigma, None, None, P).clone() for l in range(16)]
    seen.clear()
    scope.install_card(cd)
    with torch.no_grad():
        outputs = [cd.env['gnn_step'](l, x, sigma, None, None, P) for l in range(16)]
    assert all(torch.equal(a, b) for a, b in zip(references, outputs))
    assert seen == [(l, l < 12, l >= 12) for l in range(16)]
    assert cd.env['_astra_current_w3_contract']() is P['enc0_w']
    # Closed-over late leaves receive gradients from non-reentrant checkpoint
    # even though every entering prefix state has requires_grad=False.
    result = checkpoint(lambda y: cd.step(12, (y,))[0], x, use_reentrant=False)
    result.square().mean().backward()
    grads = scope.gradients()
    assert set(grads) == {'late.' + key for key in SHAPES}
    assert cd.P is P and all(value.grad is None for value in P.values())
    with args.output.open('x') as stream:
        json.dump(dict(zero_update_tiny_bytes_equal=True, layers=list(range(16)),
                       original_W3_identity_preserved=True, prefix_inputs_no_grad=True,
                       nonreentrant_checkpoint_late_gradients=True,
                       only_six_late_gradients=True, network_model_executed=False), stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
