"""TAG fable-covhunter (CH-48): the analyst's torch S6b card (fable_analyst_v1/s6bcard.py, unchanged) with a 2x-wide GNN trunk
(hidden 256/256 instead of 128/128). gnn_step reads the widths from the parameter shapes, so only the payload changes: payload(torch)
returns the function-preserving widened B2-epoch-1 parameters (make_wide_init.py -> wide2x_B2ep1_init.pt) as fresh float32 leaves.
Drop-in card module for refit6b_aux.py --card s6bcard_wide (run WITHOUT --resume, or --resume a .pt of the same widths)."""
from pathlib import Path
import s6bcard as S6
PARAMS = Path(__file__).resolve().parent / 'wide2x_B2ep1_init.pt'
model, weights, truthpost, card = S6.model, S6.weights, S6.truthpost, S6.card


def payload(torch, params=PARAMS):
    sv = torch.load(params)
    return ({k: v.detach().clone().requires_grad_(False) for k, v in sv['p'].items()}, sv['a'].detach().clone(), sv['c'].detach().clone(),
            {k: v.detach().clone() for k, v in sv['e'].items()})
