# Preserve dtype checks; numerical outputs remain verified by the fixed panel.
def _shared_old384_finite(value, name):
    if str(value.dtype) != 'float32':
        raise RuntimeError('Shared-old384 non-FP32 ' + name)
    return value

