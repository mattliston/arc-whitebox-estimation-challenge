# Fixed S34 experiment: reduce the pair-map c0 dial only at layers 10 through 14.
_K31R_TABLE = {layer: (0.480758989687834, values[1]) if layer in (10, 11, 12, 13, 14) else values for layer, values in _K31R_TABLE.items()}
