"""Fixed first-join row allocation; original full Gram and rank are unchanged."""


def _fjdr_rows(layer, diagonal, J, weight, coefficient, state, start, stop, rank):
    energy = fnp.argsort(-diagonal.astype(fnp.float64), kind='stable')
    if layer != 7:
        return fnp.sort(energy[:rank])
    if (rank != 384 or start != 0 or int(state['columns']) != 0 or state['joins']
            or tuple(J.shape) != (int(diagonal.shape[0]), stop)
            or tuple(weight.shape) != (stop,) or tuple(coefficient.shape) != (stop,)
            or int(J.shape[0]) < rank or len(state['cohorts']) < 3
            or any(str(x.dtype) != 'float32' for x in (J, weight, coefficient))):
        raise RuntimeError('first-join difference row ownership')
    score = None
    covered = 0
    for position, width in state['cohorts'][:3]:
        position, width = int(position), int(width)
        if position != covered or width <= 0 or width % 2 or position+width > stop:
            raise RuntimeError('first-join complete signed cohort')
        h = width // 2
        if not bool(fnp.all(coefficient[position:position+h] == -coefficient[position+h:position+width])):
            raise RuntimeError('first-join opposite stored coefficients')
        difference = J[:, position:position+h] - J[:, position+h:position+width]
        term = fnp.matmul(difference * difference, weight[position:position+h])
        score = term if score is None else score + term
        covered += width
    if covered != stop or score is None:
        raise RuntimeError('first-join cohort coverage')
    # A stable sort in incumbent energy order also gives deterministic energy
    # fill when contrast scores tie (including the all-zero contrast case).
    remaining = energy[288:]
    contrast = fnp.argsort(-score[remaining].astype(fnp.float64), kind='stable')[:96]
    return fnp.sort(fnp.concatenate((energy[:288], remaining[contrast])))
