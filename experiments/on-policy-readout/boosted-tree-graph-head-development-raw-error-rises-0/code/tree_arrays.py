"""Export numeric LightGBM trees, with explicit self-looping terminal nodes."""
import numpy as np


def export(model, max_depth=4):
    assert not model.get('average_output', False)
    assert model['num_tree_per_iteration'] == 1
    trees = model['tree_info']
    width = 2 ** (max_depth + 1) - 1
    shape = (len(trees), width)
    feature = np.zeros(shape, dtype=np.int32)
    threshold = np.full(shape, np.inf, dtype=np.float64)
    left = np.zeros(shape, dtype=np.int32)
    right = np.zeros(shape, dtype=np.int32)
    value = np.zeros(shape, dtype=np.float64)
    visited = []

    def visit(t, node, k, depth):
        assert depth <= max_depth and k < width
        address = t * width + k
        if 'leaf_value' in node:
            left[t, k] = right[t, k] = address
            value[t, k] = node['leaf_value']
            visited.append((t, k, 'leaf'))
        else:
            assert node['decision_type'] == '<='
            assert node['missing_type'] == 'None'
            feature[t, k] = node['split_feature']
            threshold[t, k] = node['threshold']
            left[t, k] = t * width + 2 * k + 1
            right[t, k] = t * width + 2 * k + 2
            visited.append((t, k, 'branch'))
            visit(t, node['left_child'], 2 * k + 1, depth + 1)
            visit(t, node['right_child'], 2 * k + 2, depth + 1)

    for t, tree in enumerate(trees):
        visit(t, tree['tree_structure'], 0, 0)
    return dict(feature=feature.ravel(), threshold=threshold.ravel(),
                left=left.ravel(), right=right.ravel(), value=value.ravel(),
                roots=np.arange(len(trees), dtype=np.int32) * width,
                max_depth=np.array(max_depth, dtype=np.int32)), visited


def predict(x, arrays, batch=1024):
    assert np.isfinite(x).all()
    results = []
    for start in range(0, len(x), batch):
        xx = x[start:start+batch]
        rows = np.arange(len(xx))[:, None]
        nodes = np.broadcast_to(arrays['roots'], (len(xx), len(arrays['roots']))).copy()
        for _ in range(int(arrays['max_depth'])):
            f = arrays['feature'][nodes]
            go_left = xx[rows, f] <= arrays['threshold'][nodes]
            nodes = np.where(go_left, arrays['left'][nodes], arrays['right'][nodes])
        results.append(arrays['value'][nodes].sum(axis=1))
    return np.concatenate(results)
