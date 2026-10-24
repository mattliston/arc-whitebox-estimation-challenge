"""Explicit host-then-FP64 transfer repair; identical cap/state/P equations.

Preserve model.py and its CPU qualification. Moving an MPS tensor and changing
it to FP64 in one .to call is unsupported. Five inbound transfers now first
move the original dtype to CPU, then convert there. The reverse diagonal
transfer first converts on CPU, then moves to the original device.
Differentiable transfers remain differentiable; ranking alone is detached.
"""
import ast
from common import HERE, sha

MODEL_V1 = HERE/'model.py'
MODEL_V1_SHA = 'cde8c3415324765e95a5a4237f796834af948c637976518ff641d18f6131693c'


def repaired_source():
    assert sha(MODEL_V1) == MODEL_V1_SHA
    original = MODEL_V1.read_text()
    edits = [
        ("dropped.to(device='cpu', dtype=torch.float64)", "dropped.cpu().to(dtype=torch.float64)"),
        ("weights.to(device='cpu', dtype=torch.float64)", "weights.cpu().to(dtype=torch.float64)"),
        ("physical.detach()[:, pos].to('cpu', torch.float64)", "physical.detach()[:, pos].cpu().to(torch.float64)"),
        ("physical.detach()[:, neg].to('cpu', torch.float64)", "physical.detach()[:, neg].cpu().to(torch.float64)"),
        ("lam.detach().to('cpu', torch.float64)", "lam.detach().cpu().to(torch.float64)"),
        ("cubes.to(device=physical.device, dtype=dtype)", "cubes.to(dtype=dtype).to(device=physical.device)"),
    ]
    source = original
    for old, new in edits:
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    inverse = source
    for old, new in reversed(edits):
        assert inverse.count(new) == 1, new
        inverse = inverse.replace(new, old)
    assert inverse == original
    ast.parse(source)
    return source


# This is a model module, not a numerical runner. As with the frozen model,
# numerical libraries are imported only when a checker explicitly imports it;
# no network/payload is loaded and no card is instantiated at module import.
exec(compile(repaired_source(), str(__file__)+':host_transfer_repair', 'exec'), globals())
