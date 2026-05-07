import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from pystreed import STreeDPiecewiseLinearRegressor


def make_synthetic(n=300, seed=7):
    rng = np.random.default_rng(seed)

    # A: binary split features (p=4)
    A = rng.integers(0, 2, size=(n, 4))

    # B: binary leaf-regression features (q=3)
    B = rng.integers(0, 2, size=(n, 3))

    # Region defined purely by A[:,0], A[:,1]
    region = 2 * A[:, 0] + A[:, 1]

    # Different linear models per region: y = intercept + B @ coef + noise
    coefs = np.array([
        [1.5, -0.5, 0.0],
        [0.0, 1.0, 1.2],
        [-1.0, 0.0, 1.5],
        [0.8, -1.1, 0.6],
    ])
    intercepts = np.array([0.2, 1.0, -0.7, 1.8])

    y = intercepts[region] + np.sum(B * coefs[region], axis=1) + rng.normal(0.0, 0.05, size=n)
    return A, B, y


def collect_leaf_models(node, out):
    if node.is_leaf_node():
        out.append(node.label)
        return
    collect_leaf_models(node.left_child, out)
    collect_leaf_models(node.right_child, out)


if __name__ == "__main__":
    A, B, y = make_synthetic()

    # Pass only A as tree split features, and B via continuous_columns as leaf-regression features.
    model = STreeDPiecewiseLinearRegressor(
        max_depth=2,
        min_leaf_node_size=15,
        lasso_penalty=0.0,
        ridge_penalty=1.0,
        random_seed=0,
        verbose=False,
    )
    model.fit(A, y, continuous_columns=B)
    pred = model.predict(A, continuous_columns=B)

    print("=== Tree structure (should split on A features only) ===")
    model.print_tree(feature_names=[f"A{j}" for j in range(A.shape[1])])

    print("\n=== Leaf linear models (should use only B features) ===")
    leaves = []
    collect_leaf_models(model.get_tree(), leaves)
    for i, leaf_model in enumerate(leaves):
        print(f"Leaf {i}: intercept={leaf_model.intercept:.4f}, coefs={np.array(leaf_model.coefficients)}")

    print("\n=== Fit quality on synthetic data ===")
    print(f"R2: {r2_score(y, pred):.4f}")
    print(f"MSE: {mean_squared_error(y, pred):.6f}")
