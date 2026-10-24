"""TAG fable-covhunter (CH-47): the analyst's torch S6b card (fable_analyst_v1/s6bcard.py, unchanged) with the old-tier frame at rank
RANK = 320 instead of 384 (lead's test: can the weights-only trunk refit absorb a tier cut?). The HookCard (covstep.py) passes
rank=TIER_RANK, a module global read at call time, to the every3 push (s2card.make_push); nothing else in the card depends on the rank.
Drop-in card module for refit6b_aux.py --card s6bcard_r320 (payload/PARAMS/model re-exported from s6bcard)."""
import s6bcard as S6
import covstep as cs
RANK = 320
model, weights, truthpost, payload, PARAMS = S6.model, S6.weights, S6.truthpost, S6.payload, S6.PARAMS


def card(torch, w, p, a, c, e, fine_ckpt=False, **kw):
    cs.TIER_RANK = RANK
    return S6.card(torch, w, p, a, c, e, fine_ckpt=fine_ckpt, **kw)
