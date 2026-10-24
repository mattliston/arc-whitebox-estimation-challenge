

# BEGIN SCOUT-NATIVE K31 ROW FACTOR c0 +0.1 AT L8-14 (fragment_k31rx_010_v1, 2026-10-13, the physicist's T146 arm (f) / the lead's (g) rule: a
# literal _K31R_TABLE change at L8-14, as T146 pre-declared): the dialed pair-map row factor is c0 + c1 mu k3 / v^2 with (c0, c1) per layer in
# _K31R_TABLE (0.52197, 0.50269 at every layer of the S54 point); c0 += _K31RX at the layers in _K31RX_LAYERS; c1 (the bound constants of
# _g5_setup) unchanged. Python floats at import, no array arithmetic. _K31RX = 0.0 = the parent bit for bit.
_K31RX = 0.1
_K31RX_LAYERS = (8, 9, 10, 11, 12, 13, 14)
if _K31RX:
    _K31R_TABLE = {l: ((float(c[0]) + _K31RX, c[1]) if l in _K31RX_LAYERS else c) for l, c in _K31R_TABLE.items()}
# END SCOUT-NATIVE K31 ROW FACTOR c0 +0.1 AT L8-14
