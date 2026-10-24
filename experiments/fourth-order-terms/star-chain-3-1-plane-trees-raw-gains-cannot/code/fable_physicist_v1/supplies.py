"""TAG fable-physicist (PH-5): model supplies (functions of the card's own state) for model_bank_phys.PHYS_SUPPLY. Filled as the physics
is derived; make(name, all_consumers) -> {layer: {'fn': ..., 'all_consumers': bool}}."""
REGISTRY = {}

def make(name, allc):
    fn, layers = REGISTRY[name]
    return {l: {'fn': fn, 'all_consumers': allc} for l in layers}
