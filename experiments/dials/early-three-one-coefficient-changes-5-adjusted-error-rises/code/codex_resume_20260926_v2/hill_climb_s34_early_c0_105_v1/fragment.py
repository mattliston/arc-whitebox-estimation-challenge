# Fixed S34 experiment: increase pair-map c0 only at layers 1 through 9.
_K31R_TABLE = {layer: (0.5608854879691396, values[1]) if 1 <= layer <= 9 else values for layer, values in _K31R_TABLE.items()}
