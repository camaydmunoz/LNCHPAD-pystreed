import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from pystreed import STreeDPiecewiseLinearRegressor


def make_synthetic(n=300, seed=7):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=(n, 4))
    B = rng.integers(0, 2, size=(n, 3))
    region = 2 * A[:, 0] + A[:, 1]
    coefs = np.array([
        [1.5, -0.5, 0.0],
        [0.0, 1.0, 1.2],
        [-1.0, 0.0, 1.5],
        [0.8, -1.1, 0.6],
    ])
    intercepts = np.array([0.2, 1.0, -0.7, 1.8])
    y = intercepts[region] + np.sum(B * coefs[region], axis=1) + rng.normal(0.0, 0.05, size=n)
    return A, B, y


def ridge_leaf_fit(B_leaf, y_leaf, ridge_penalty):
    n = B_leaf.shape[0]
    X = np.column_stack([np.ones(n), B_leaf])
    p = X.shape[1]
    reg = np.eye(p) * ridge_penalty
    reg[0, 0] = 0.0  # do not penalize intercept
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y_leaf)
    residual = y_leaf - X @ beta
    sse = float(residual @ residual)
    ridge = float(ridge_penalty * (beta[1:] @ beta[1:]))
    return beta, sse + ridge


def objective_for_partition(B, y, leaf_indices, ridge_penalty):
    total = 0.0
    leaf_models = []
    for idx in leaf_indices:
        beta, leaf_obj = ridge_leaf_fit(B[idx], y[idx], ridge_penalty)
        total += leaf_obj
        leaf_models.append(beta)
    return total, leaf_models


def masks_for_tree(A, root, left=None, right=None):
    root_right = A[:, root] == 1
    root_left = ~root_right

    if left is None and right is None:
        return [root_left, root_right], f"A{root}"

    leaves = []
    parts = [f"A{root}"]

    if left is None:
        leaves.append(root_left)
        parts.append("L:leaf")
    else:
        l_right = (A[:, left] == 1) & root_left
        l_left = root_left & ~l_right
        leaves.extend([l_left, l_right])
        parts.append(f"L:A{left}")

    if right is None:
        leaves.append(root_right)
        parts.append("R:leaf")
    else:
        r_right = (A[:, right] == 1) & root_right
        r_left = root_right & ~r_right
        leaves.extend([r_left, r_right])
        parts.append(f"R:A{right}")

    return leaves, " | ".join(parts)


def brute_force_ab_ridge_tree(A, B, y, ridge_penalty, min_leaf_node_size, cost_complexity=0.0):
    n_features = A.shape[1]
    best = {"objective": np.inf}

    all_idx = np.ones(A.shape[0], dtype=bool)
    if all_idx.sum() >= min_leaf_node_size:
        obj, models = objective_for_partition(B, y, [all_idx], ridge_penalty)
        if obj < best["objective"]:
            best = {"objective": obj, "split": "leaf", "masks": [all_idx], "models": models, "nodes": 0}

    for root in range(n_features):
        masks, split = masks_for_tree(A, root)
        if min(mask.sum() for mask in masks) < min_leaf_node_size:
            continue
        obj, models = objective_for_partition(B, y, masks, ridge_penalty)
        obj += cost_complexity
        if obj < best["objective"]:
            best = {"objective": obj, "split": split, "masks": masks, "models": models, "nodes": 1}

    for root in range(n_features):
        for left in [None, *range(n_features)]:
            for right in [None, *range(n_features)]:
                if left is None and right is None:
                    continue
                masks, split = masks_for_tree(A, root, left, right)
                if min(mask.sum() for mask in masks) < min_leaf_node_size:
                    continue
                obj, models = objective_for_partition(B, y, masks, ridge_penalty)
                num_nodes = 1 + int(left is not None) + int(right is not None)
                obj += cost_complexity * num_nodes
                if obj < best["objective"]:
                    best = {"objective": obj, "split": split, "masks": masks, "models": models, "nodes": num_nodes}
    return best


def tree_structure(node):
    if node.is_leaf_node():
        return "leaf"
    return f"A{node.feature}(L={tree_structure(node.left_child)},R={tree_structure(node.right_child)})"


def predictions_from_partition(B, masks, models):
    pred = np.zeros(B.shape[0], dtype=float)
    for mask, beta in zip(masks, models):
        X = np.column_stack([np.ones(mask.sum()), B[mask]])
        pred[mask] = X @ beta
    return pred


def objective_from_predictions(y, pred, models, ridge_penalty):
    sse = float(np.sum((y - pred) ** 2))
    ridge = float(sum(ridge_penalty * (beta[1:] @ beta[1:]) for beta in models))
    return sse, ridge, sse + ridge


if __name__ == "__main__":
    A, B, y = make_synthetic(n=120, seed=11)
    lasso_penalty = 0.0
    ridge_penalty = 1.0
    min_leaf_node_size = 15
    cost_complexity = 0.0
    max_depth = 2
    max_num_nodes = 3

    brute = brute_force_ab_ridge_tree(
        A=A,
        B=B,
        y=y,
        ridge_penalty=ridge_penalty,
        min_leaf_node_size=min_leaf_node_size,
        cost_complexity=cost_complexity,
    )
    brute_pred = predictions_from_partition(B, brute["masks"], brute["models"])
    brute_sse, brute_ridge, brute_obj = objective_from_predictions(y, brute_pred, brute["models"], ridge_penalty)

    model = STreeDPiecewiseLinearRegressor(
        max_depth=max_depth,
        max_num_nodes=max_num_nodes,
        min_leaf_node_size=min_leaf_node_size,
        lasso_penalty=lasso_penalty,
        ridge_penalty=ridge_penalty,
        cost_complexity=cost_complexity,
        random_seed=0,
        verbose=False,
    )
    model.fit(A, y, continuous_columns=B)
    st_pred = model.predict(A, continuous_columns=B)

    print("=== A/B ridge leaf oracle check (depth <= 2) ===")
    print(f"Brute-force best objective (SSE + ridge): {brute['objective']:.8f}")
    if hasattr(model, "fit_result"):
        print(f"STreeD fit_result.score(): {model.fit_result.score():.8f}")
    else:
        print("STreeD objective unavailable")

    print(f"Brute-force best split structure: {brute['split']}")
    print(f"STreeD learned split structure: {tree_structure(model.get_tree())}")

    print("\n=== Prediction quality (training set) ===")
    print(f"Brute-force MSE: {mean_squared_error(y, brute_pred):.8f}")
    print(f"Brute-force R2:  {r2_score(y, brute_pred):.8f}")
    print(f"STreeD MSE:      {mean_squared_error(y, st_pred):.8f}")
    print(f"STreeD R2:       {r2_score(y, st_pred):.8f}")
    print(f"Brute-force objective decomposition: SSE={brute_sse:.8f}, ridge={brute_ridge:.8f}, total={brute_obj:.8f}")
    print(f"STreeD total SSE (from MSE*n):      {mean_squared_error(y, st_pred) * len(y):.8f}")

    print("\n=== Single-leaf diagnostic (max_depth=0) ===")
    single_oracle_beta, _ = ridge_leaf_fit(B, y, ridge_penalty=ridge_penalty)
    X_full = np.column_stack([np.ones(len(y)), B])
    single_oracle_pred = X_full @ single_oracle_beta
    single_oracle_sse = float(np.sum((y - single_oracle_pred) ** 2))
    single_oracle_ridge = float(ridge_penalty * (single_oracle_beta[1:] @ single_oracle_beta[1:]))
    single_oracle_obj = single_oracle_sse + single_oracle_ridge

    single_model = STreeDPiecewiseLinearRegressor(
        max_depth=0,
        max_num_nodes=0,
        min_leaf_node_size=min_leaf_node_size,
        lasso_penalty=lasso_penalty,
        ridge_penalty=ridge_penalty,
        cost_complexity=cost_complexity,
        random_seed=0,
        verbose=False,
    )
    single_model.fit(A, y, continuous_columns=B)
    single_st_pred = single_model.predict(A, continuous_columns=B)
    single_st_sse = float(np.sum((y - single_st_pred) ** 2))
    single_st_ridge = single_oracle_ridge
    single_st_obj = single_st_sse + single_st_ridge
    print(f"Oracle one-leaf intercept:      {single_oracle_beta[0]: .8f}")
    print(f"Oracle one-leaf coefficients:   {single_oracle_beta[1:]}")
    print(f"Oracle one-leaf SSE:            {single_oracle_sse:.8f}")
    print(f"Oracle one-leaf ridge:          {single_oracle_ridge:.8f}")
    print(f"Oracle one-leaf total objective:{single_oracle_obj:.8f}")
    print(f"STreeD one-leaf SSE:            {single_st_sse:.8f}")
    print(f"STreeD one-leaf total(+ridge*): {single_st_obj:.8f}")
    print(f"Single-leaf max |pred diff|:    {np.max(np.abs(single_oracle_pred - single_st_pred)):.8e}")
    print("* STreeD leaf coeffs/intercept are not directly exposed; ridge term reused from oracle fit.")

    print("\n=== Fixed-tree objective diagnostic (same oracle convention) ===")
    root = model.get_tree()
    st_masks = []
    if root.is_leaf_node():
        st_masks = [np.ones(len(y), dtype=bool)]
    else:
        root_right = A[:, root.feature] == 1
        root_left = ~root_right
        left = root.left_child
        right = root.right_child
        if left.is_leaf_node():
            st_masks.append(root_left)
        else:
            l_right = (A[:, left.feature] == 1) & root_left
            st_masks.extend([root_left & ~l_right, l_right])
        if right.is_leaf_node():
            st_masks.append(root_right)
        else:
            r_right = (A[:, right.feature] == 1) & root_right
            st_masks.extend([root_right & ~r_right, r_right])
    st_obj_oracle, _ = objective_for_partition(B, y, st_masks, ridge_penalty)
    print(f"Oracle objective on oracle-best structure: {brute_obj:.8f}")
    print(f"Oracle objective on STreeD structure:      {st_obj_oracle:.8f}")
