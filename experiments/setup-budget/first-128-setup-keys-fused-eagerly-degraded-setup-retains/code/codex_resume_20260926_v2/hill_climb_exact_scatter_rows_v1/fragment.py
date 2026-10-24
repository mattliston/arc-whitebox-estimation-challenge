# Exact inverse-row expansion: unique row rosters are existing argsort slices.
def _exact_expand_rows(values, rows, total_rows):
    kept, width = map(int, values.shape)
    if len(rows.shape) != 1 or int(rows.shape[0]) != kept or kept > total_rows:
        raise RuntimeError('exact expansion row ownership')
    if str(values.dtype) != 'float32':
        raise TypeError('exact expansion requires original FP32 values')
    answer = fnp.zeros((int(total_rows), width), dtype=values.dtype)
    fnp.put_along_axis(answer, rows[:, None], values, axis=0)
    return answer
