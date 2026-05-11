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
    ])
    y = np.array([0.0, 1.0, 1.1, 2.0, 0.03, 1.02, 1.09, 2.02])
    return A, B, y


def ridge_leaf_fit(B_leaf, y_leaf, ridge_penalty):
    # STreeD ridge convention:
    # - center B and y in leaf
    # - gamma = n_leaf * ridge_penalty
    # - no intercept penalty
    # - intercept reconstructed after centered solve
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
    obj = sse + float(gamma * (coef @ coef))

    return {
        "coef": coef,
        "intercept": intercept,
        "gamma": gamma,
        "objective": obj,
    }


def objective_for_partition(B, y, leaf_indices):
    total = 0.0
    leaf_models = []
    for idx in leaf_indices:
        model = ridge_leaf_fit(B[idx], y[idx], ridge_penalty)
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


def sorted_oracle_ab_ridge_tree(A, B, y, ridge_penalty_value, min_leaf_node_size, cost_complexity=0.0):
    global ridge_penalty
    ridge_penalty = ridge_penalty_value

    n_features = A.shape[1]
    all_candidates = []
    all_idx = np.ones(A.shape[0], dtype=bool)

    # Depth 0 tree (single leaf).
    if all_idx.sum() >= min_leaf_node_size:
        obj, models = objective_for_partition(B, y, [all_idx])
        all_candidates.append(
            {
                "objective": obj,
                "split": "leaf",
                "masks": [all_idx],
                "models": models,
                "nodes": 0,
                "depth": 0,
            }
        )

    # Depth 1 trees.
    for root in range(n_features):
        masks, split = masks_for_tree(A, root)
        if min(mask.sum() for mask in masks) < min_leaf_node_size:
            continue
        obj, models = objective_for_partition(B, y, masks)
        all_candidates.append(
            {
                "objective": obj + cost_complexity,
                "split": split,
                "masks": masks,
                "models": models,
                "nodes": 1,
                "depth": 1,
            }
        )

    # Depth 2 trees.
    for root in range(n_features):
        for left in [None, *range(n_features)]:
            for right in [None, *range(n_features)]:
                if left is None and right is None:
                    continue
                if left == root or right == root:
                    # Avoid path-duplicate split on same feature.
                    continue
                masks, split = masks_for_tree(A, root, left, right)
                if min(mask.sum() for mask in masks) < min_leaf_node_size:
                    continue
                obj, models = objective_for_partition(B, y, masks)
                num_nodes = 1 + int(left is not None) + int(right is not None)
                all_candidates.append(
                    {
                        "objective": obj + cost_complexity * num_nodes,
                        "split": split,
                        "masks": masks,
                        "models": models,
                        "nodes": num_nodes,
                        "depth": 2,
                    }
                )

    return sorted(all_candidates, key=_sort_key)


def brute_force_ab_ridge_tree(A, B, y, ridge_penalty, min_leaf_node_size, cost_complexity=0.0):
    sorted_candidates = sorted_oracle_ab_ridge_tree(
        A=A,
        B=B,
        y=y,
        ridge_penalty_value=ridge_penalty,
        min_leaf_node_size=min_leaf_node_size,
        cost_complexity=cost_complexity,
    )
    return sorted_candidates[0]


def tree_structure(node):
    if node.is_leaf_node():
        return "leaf"
    return f"A{node.feature}(L={tree_structure(node.left_child)},R={tree_structure(node.right_child)})"


def predictions_from_partition(B, masks, models):
    pred = np.zeros(B.shape[0], dtype=float)
    for mask, model in zip(masks, models):
        pred[mask] = B[mask] @ model["coef"] + model["intercept"]
    return pred


def print_top_candidates(candidates, top_k=10):
    print(f"Top {min(top_k, len(candidates))} candidates:")
    for i, c in enumerate(candidates[:top_k], start=1):
        print(f"  {i:>2}. obj={c['objective']:.10f} depth={c['depth']} nodes={c['nodes']} split={c['split']}")


def run_case(case_name, A, B, y, ridge_penalty, min_leaf_node_size, cost_complexity):
    print(f"\n=== {case_name} ===")

    oracle_candidates = sorted_oracle_ab_ridge_tree(
        A=A,
        B=B,
        y=y,
        ridge_penalty_value=ridge_penalty,
        min_leaf_node_size=min_leaf_node_size,
        cost_complexity=cost_complexity,
    )
    brute = brute_force_ab_ridge_tree(
        A=A,
        B=B,
        y=y,
        ridge_penalty=ridge_penalty,
        min_leaf_node_size=min_leaf_node_size,
        cost_complexity=cost_complexity,
    )
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

    model = STreeDPiecewiseLinearRegressor(
        max_depth=2,
        min_leaf_node_size=min_leaf_node_size,
        lasso_penalty=0.0,
        ridge_penalty=ridge_penalty,
        cost_complexity=cost_complexity,
        random_seed=0,
        verbose=False,
    )
    model.fit(A, y, continuous_columns=B)

    oracle_objectives = np.array([c["objective"] for c in oracle_candidates])
    st_pred = None
    st_score = np.nan
    if hasattr(model, "fit_result"):
        st_score = model.fit_result.score()
        st_pred = model.predict(A, continuous_columns=B)
        print(f"STreeD learned split structure: {tree_structure(model.get_tree())}")
    else:
        print("STreeD did not return a feasible tree for this case.")

    close_idx = int(np.argmin(np.abs(oracle_objectives - st_score))) if np.isfinite(st_score) else -1

    if np.isfinite(st_score):
        print(f"STreeD fit_result.score(): {st_score:.10f}")
        print(
            f"Closest oracle candidate: idx={close_idx + 1}, "
            f"obj={oracle_candidates[close_idx]['objective']:.10f}, "
            f"split={oracle_candidates[close_idx]['split']}"
        )
        diff = abs(oracle_candidates[close_idx]["objective"] - st_score)
        print(f"Abs objective difference: {diff:.12f}")
    else:
        print("STreeD objective unavailable")

    brute_pred = predictions_from_partition(B, brute["masks"], brute["models"])
    print("Prediction quality (training set):")
    print(f"  Oracle/Brute MSE: {mean_squared_error(y, brute_pred):.10f}")
    print(f"  Oracle/Brute R2:  {r2_score(y, brute_pred):.10f}")
    if st_pred is not None:
        print(f"  STreeD MSE:       {mean_squared_error(y, st_pred):.10f}")
        print(f"  STreeD R2:        {r2_score(y, st_pred):.10f}")


if __name__ == "__main__":
    A, B, y = make_synthetic(n=120, seed=11)
    run_case(
        case_name="A/B ridge leaf oracle check (depth <= 2)",
        A=A,
        B=B,
        y=y,
        ridge_penalty=1.0,
        min_leaf_node_size=15,
        cost_complexity=0.0,
    )

    A2, B2, y2 = make_close_objective_case()
    run_case(
        case_name="Close-objective tie-break stability check",
        A=A2,
        B=B2,
        y=y2,
        ridge_penalty=0.25,
        min_leaf_node_size=2,
        cost_complexity=0.0,
    )
