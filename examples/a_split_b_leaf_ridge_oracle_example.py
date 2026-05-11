import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from pystreed import STreeDPiecewiseLinearRegressor


def make_synthetic(n=300, seed=7):
    rng = np.random.default_rng(seed)

    # A: binary split features.
    A = rng.integers(0, 2, size=(n, 4))

    # B: binary leaf-regression features.
    B = rng.integers(0, 2, size=(n, 3)).astype(float)

    # Region is determined only by A.
    region = 2 * A[:, 0] + A[:, 1]

    # Within each A-defined region, y depends linearly on B.
    coefs = np.array([
        [1.5, -0.5, 0.0],
        [0.0, 1.0, 1.2],
        [-1.0, 0.0, 1.5],
        [0.8, -1.1, 0.6],
    ])
    intercepts = np.array([0.2, 1.0, -0.7, 1.8])

    y = (
        intercepts[region]
        + np.sum(B * coefs[region], axis=1)
        + rng.normal(0.0, 0.05, size=n)
    )
    return A, B, y


def make_close_objective_case():
    # Tiny deterministic dataset with nearly tied candidates.
    A = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=int)

    B = np.array([
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ], dtype=float)

    y = np.array([0.0, 1.0, 1.1, 2.0, 0.03, 1.02, 1.09, 2.02])
    return A, B, y


def standardize_continuous_like_streed(B):
    """Mimic STreeD's global preprocessing of continuous_columns.

    STreeD globally standardizes continuous leaf-regression features before
    the piecewise-linear task sees them. The leaf solver then does local
    centering within each leaf.

    Uses population std, matching sqrt(mean(x^2) - mean(x)^2).
    Constant columns are left unchanged, matching STreeD's skip behavior.
    """
    B = np.asarray(B, dtype=float)
    mu = B.mean(axis=0)
    std = np.sqrt(np.maximum((B * B).mean(axis=0) - mu * mu, 0.0))

    B_std = B.copy()
    nonconstant = np.abs(std) >= 1e-6
    B_std[:, nonconstant] = (B[:, nonconstant] - mu[nonconstant]) / std[nonconstant]

    return B_std, mu, std


def ridge_leaf_fit(B_leaf, y_leaf, ridge_penalty):
    """Fit one ridge leaf using STreeD's internal convention.

    Important: B_leaf should already be in STreeD's globally standardized
    continuous-feature space.

    Leaf convention:
    - center B and y within the leaf,
    - gamma = n_leaf * ridge_penalty,
    - do not penalize intercept,
    - reconstruct intercept after centered ridge solve,
    - objective = SSE + gamma * ||coef||^2.
    """
    n_leaf = B_leaf.shape[0]
    if n_leaf == 0:
        raise ValueError("Empty leaf is invalid")

    b_mean = B_leaf.mean(axis=0)
    y_mean = float(y_leaf.mean())

    Bc = B_leaf - b_mean
    yc = y_leaf - y_mean

    gamma = n_leaf * ridge_penalty
    lhs = Bc.T @ Bc + gamma * np.eye(B_leaf.shape[1])
    rhs = Bc.T @ yc

    coef = np.linalg.solve(lhs, rhs)
    intercept = y_mean - float(b_mean @ coef)

    pred = B_leaf @ coef + intercept
    residual = y_leaf - pred
    sse = float(residual @ residual)
    ridge = float(gamma * (coef @ coef))
    obj = sse + ridge

    return {
        "coef": coef,
        "intercept": intercept,
        "gamma": gamma,
        "sse": sse,
        "ridge": ridge,
        "objective": obj,
    }


def objective_for_partition(B_internal, y, leaf_masks, ridge_penalty):
    total = 0.0
    leaf_models = []

    for mask in leaf_masks:
        model = ridge_leaf_fit(B_internal[mask], y[mask], ridge_penalty)
        total += model["objective"]
        leaf_models.append(model)

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


def _sort_key(candidate):
    # Stable deterministic ordering for close objectives.
    return (
        round(candidate["objective"], 12),
        candidate["depth"],
        candidate["nodes"],
        candidate["split"],
    )


def sorted_oracle_ab_ridge_tree(
    A,
    B_internal,
    y,
    ridge_penalty,
    min_leaf_node_size,
    cost_complexity=0.0,
):
    """Enumerate all valid A-split / B-ridge trees of depth <= 2.

    B_internal should be the globally standardized B used internally by STreeD.
    """
    n_features = A.shape[1]
    all_candidates = []
    all_idx = np.ones(A.shape[0], dtype=bool)

    # Depth 0 tree: one ridge leaf.
    if all_idx.sum() >= min_leaf_node_size:
        obj, models = objective_for_partition(B_internal, y, [all_idx], ridge_penalty)
        all_candidates.append({
            "objective": obj,
            "split": "leaf",
            "masks": [all_idx],
            "models": models,
            "nodes": 0,
            "depth": 0,
        })

    # Depth 1 trees: one root split, two ridge leaves.
    for root in range(n_features):
        masks, split = masks_for_tree(A, root)
        if min(mask.sum() for mask in masks) < min_leaf_node_size:
            continue

        obj, models = objective_for_partition(B_internal, y, masks, ridge_penalty)
        all_candidates.append({
            "objective": obj + cost_complexity,
            "split": split,
            "masks": masks,
            "models": models,
            "nodes": 1,
            "depth": 1,
        })

    # Depth 2 trees: root split, optional left/right child splits.
    for root in range(n_features):
        for left in [None, *range(n_features)]:
            for right in [None, *range(n_features)]:
                if left is None and right is None:
                    continue

                # Avoid reusing the same split feature along a path.
                if left == root or right == root:
                    continue

                masks, split = masks_for_tree(A, root, left, right)
                if min(mask.sum() for mask in masks) < min_leaf_node_size:
                    continue

                obj, models = objective_for_partition(B_internal, y, masks, ridge_penalty)
                num_nodes = 1 + int(left is not None) + int(right is not None)

                all_candidates.append({
                    "objective": obj + cost_complexity * num_nodes,
                    "split": split,
                    "masks": masks,
                    "models": models,
                    "nodes": num_nodes,
                    "depth": 2,
                })

    return sorted(all_candidates, key=_sort_key)


def brute_force_ab_ridge_tree(
    A,
    B_internal,
    y,
    ridge_penalty,
    min_leaf_node_size,
    cost_complexity=0.0,
):
    sorted_candidates = sorted_oracle_ab_ridge_tree(
        A=A,
        B_internal=B_internal,
        y=y,
        ridge_penalty=ridge_penalty,
        min_leaf_node_size=min_leaf_node_size,
        cost_complexity=cost_complexity,
    )
    return sorted_candidates[0]


def tree_structure(node):
    if node.is_leaf_node():
        return "leaf"
    return f"A{node.feature}(L={tree_structure(node.left_child)},R={tree_structure(node.right_child)})"


def masks_from_streed_tree(A, node):
    """Return leaf masks for a fitted STreeD tree."""
    if node.is_leaf_node():
        return [np.ones(A.shape[0], dtype=bool)]

    def recurse(cur_node, cur_mask):
        if cur_node.is_leaf_node():
            return [cur_mask]

        go_right = (A[:, cur_node.feature] == 1) & cur_mask
        go_left = cur_mask & ~go_right

        return (
            recurse(cur_node.left_child, go_left)
            + recurse(cur_node.right_child, go_right)
        )

    return recurse(node, np.ones(A.shape[0], dtype=bool))


def predictions_from_partition(B_internal, masks, models):
    pred = np.zeros(B_internal.shape[0], dtype=float)

    for mask, model in zip(masks, models):
        pred[mask] = B_internal[mask] @ model["coef"] + model["intercept"]

    return pred


def print_top_candidates(candidates, top_k=10):
    print(f"Top {min(top_k, len(candidates))} candidates:")
    for i, c in enumerate(candidates[:top_k], start=1):
        print(
            f"  {i:>2}. obj={c['objective']:.10f} "
            f"depth={c['depth']} nodes={c['nodes']} split={c['split']}"
        )


def run_case(
    case_name,
    A,
    B_raw,
    y,
    ridge_penalty,
    min_leaf_node_size,
    cost_complexity,
    compare_streed=True,
):
    print(f"\n=== {case_name} ===")

    # Oracle must use STreeD's internal globally standardized continuous data.
    # STreeD itself still receives raw B and standardizes internally.
    B_internal, b_mu, b_std = standardize_continuous_like_streed(B_raw)

    oracle_candidates = sorted_oracle_ab_ridge_tree(
        A=A,
        B_internal=B_internal,
        y=y,
        ridge_penalty=ridge_penalty,
        min_leaf_node_size=min_leaf_node_size,
        cost_complexity=cost_complexity,
    )
    brute = oracle_candidates[0]
    best = oracle_candidates[0]

    sorted_ok = all(
        oracle_candidates[i]["objective"] <= oracle_candidates[i + 1]["objective"] + 1e-12
        for i in range(len(oracle_candidates) - 1)
    )

    print_top_candidates(oracle_candidates, top_k=10)
    print(f"Sorted nondecreasing objective: {sorted_ok}")
    print(f"Best oracle objective: {best['objective']:.10f}")
    print(f"Brute-force best objective: {brute['objective']:.10f}")
    print(f"Best oracle split: {best['split']}")
    print(f"Brute-force split: {brute['split']}")

    brute_pred = predictions_from_partition(B_internal, brute["masks"], brute["models"])
    brute_mse = mean_squared_error(y, brute_pred)
    brute_r2 = r2_score(y, brute_pred)

    st_pred = None
    st_mse = None
    st_r2 = None
    st_obj_oracle = None

    if compare_streed:
        model = STreeDPiecewiseLinearRegressor(
            max_depth=2,
            max_num_nodes=3,
            min_leaf_node_size=min_leaf_node_size,
            lasso_penalty=0.0,
            ridge_penalty=ridge_penalty,
            cost_complexity=cost_complexity,
            random_seed=0,
            verbose=False,
        )

        model.fit(A, y, continuous_columns=B_raw)

        if hasattr(model, "fit_result"):
            st_pred = model.predict(A, continuous_columns=B_raw)
            st_mse = mean_squared_error(y, st_pred)
            st_r2 = r2_score(y, st_pred)

            st_structure = tree_structure(model.get_tree())
            st_masks = masks_from_streed_tree(A, model.get_tree())
            st_obj_oracle, _ = objective_for_partition(
                B_internal,
                y,
                st_masks,
                ridge_penalty,
            )
            st_obj_oracle += cost_complexity * model.fit_result.tree_nodes()

            print(f"STreeD learned split structure: {st_structure}")
            print(f"STreeD fit_result.score() [reported MSE-like score]: {model.fit_result.score():.10f}")
            print(f"Oracle objective on STreeD structure: {st_obj_oracle:.10f}")
            print(f"Oracle best - STreeD-structure objective diff: {abs(best['objective'] - st_obj_oracle):.12f}")
        else:
            print("STreeD did not return a feasible tree for this case.")
    else:
        print("STreeD comparison skipped for this case.")

    print("Prediction quality (training set):")
    print(f"  Oracle/Brute MSE: {brute_mse:.10f}")
    print(f"  Oracle/Brute R2:  {brute_r2:.10f}")

    if st_pred is not None:
        print(f"  STreeD MSE:       {st_mse:.10f}")
        print(f"  STreeD R2:        {st_r2:.10f}")
        print(f"  |Oracle MSE - STreeD MSE|: {abs(brute_mse - st_mse):.12f}")


if __name__ == "__main__":
    A, B, y = make_synthetic(n=120, seed=11)
    run_case(
        case_name="A/B ridge leaf oracle check (depth <= 2)",
        A=A,
        B_raw=B,
        y=y,
        ridge_penalty=1.0,
        min_leaf_node_size=15,
        cost_complexity=0.0,
        compare_streed=True,
    )

    A2, B2, y2 = make_close_objective_case()
    run_case(
        case_name="Close-objective tie-break stability check",
        A=A2,
        B_raw=B2,
        y=y2,
        ridge_penalty=0.25,
        min_leaf_node_size=2,
        cost_complexity=0.0,
        compare_streed=False,
    )
