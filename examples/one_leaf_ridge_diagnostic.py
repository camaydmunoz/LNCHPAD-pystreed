import numpy as np

from pystreed import STreeDPiecewiseLinearRegressor


def make_synthetic(n=80, p_bin=3, p_cont=4, seed=123):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 2, size=(n, p_bin))
    B = rng.normal(0.0, 1.0, size=(n, p_cont))
    true_beta = np.array([1.2, -0.7, 0.9, -0.25], dtype=float)
    true_b0 = -0.4
    y = true_b0 + B @ true_beta + rng.normal(0.0, 0.05, size=n)
    return A, B, y


def fit_closed_form(B, y, lambda_eff, penalize_intercept=False, center=False):
    n = B.shape[0]
    if center:
        mu_x = B.mean(axis=0)
        mu_y = y.mean()
        Xc = B - mu_x
        yc = y - mu_y
        gram = Xc.T @ Xc
        reg = np.eye(B.shape[1]) * lambda_eff
        beta = np.linalg.solve(gram + reg, Xc.T @ yc)
        intercept = mu_y - mu_x @ beta
    else:
        X = np.column_stack([np.ones(n), B])
        reg = np.eye(X.shape[1]) * lambda_eff
        if not penalize_intercept:
            reg[0, 0] = 0.0
        sol = np.linalg.solve(X.T @ X + reg, X.T @ y)
        intercept = sol[0]
        beta = sol[1:]

    pred = intercept + B @ beta
    residual = y - pred
    sse = float(residual @ residual)
    ridge = float(lambda_eff * (intercept * intercept + beta @ beta)) if penalize_intercept else float(lambda_eff * (beta @ beta))
    return {
        "beta": beta,
        "intercept": intercept,
        "pred": pred,
        "sse": sse,
        "objective": sse + ridge,
        "ridge": ridge,
    }


def fit_streed_one_leaf(A, B, y, ridge_penalty):
    model = STreeDPiecewiseLinearRegressor(
        max_depth=0,
        max_num_nodes=0,
        min_leaf_node_size=2,
        lasso_penalty=0.0,
        ridge_penalty=ridge_penalty,
        cost_complexity=0.0,
        random_seed=0,
        verbose=False,
    )
    model.fit(A, y, continuous_columns=B)
    pred = model.predict(A, continuous_columns=B)
    sse = float(np.sum((y - pred) ** 2))
    return model, pred, sse


def run_case(ridge_penalty):
    A, B, y = make_synthetic()
    model, st_pred, st_sse = fit_streed_one_leaf(A, B, y, ridge_penalty)

    n = B.shape[0]
    conventions = {
        "lambda_eff = ridge_penalty": ridge_penalty,
        "lambda_eff = n_samples * ridge_penalty": n * ridge_penalty,
        "lambda_eff = 0.5 * n_samples * ridge_penalty": 0.5 * n * ridge_penalty,
        "lambda_eff = ridge_penalty / n_samples": ridge_penalty / n,
        "source_inspection: centered X,y and lambda_eff = n_samples * ridge_penalty": n * ridge_penalty,
    }

    results = []
    for name, lam in conventions.items():
        center = "source_inspection" in name
        fit = fit_closed_form(B, y, lam, penalize_intercept=False, center=center)
        max_abs = float(np.max(np.abs(st_pred - fit["pred"])))
        results.append((name, fit, max_abs, center))

    # explicit tests for intercept penalization hypothesis under source-inspected scaling
    fit_center_no_intercept_penalty = fit_closed_form(B, y, n * ridge_penalty, penalize_intercept=False, center=True)
    fit_center_with_intercept_penalty = fit_closed_form(B, y, n * ridge_penalty, penalize_intercept=True, center=True)

    print("\n" + "=" * 80)
    print(f"One-leaf ridge diagnostic | ridge_penalty={ridge_penalty}")
    print("(STreeD constrained with max_depth=0 and max_num_nodes=0)")
    print("=" * 80)
    print(f"STreeD SSE: {st_sse:.12f}")

    best = min(results, key=lambda r: r[2])
    for name, fit, max_abs, center in results:
        print(f"- {name}")
        print(f"  max |pred_streed - pred_oracle|: {max_abs:.12e}")
        print(f"  oracle SSE: {fit['sse']:.12f}")
        print(f"  oracle objective (SSE + ridge): {fit['objective']:.12f}")
        if center:
            print("  convention notes: centers B and y; does not scale by std")

    print("\nIntercept penalization probe (source-inspected scaling):")
    no_int_diff = float(np.max(np.abs(st_pred - fit_center_no_intercept_penalty["pred"])))
    with_int_diff = float(np.max(np.abs(st_pred - fit_center_with_intercept_penalty["pred"])))
    print(f"- unpenalized intercept max pred diff: {no_int_diff:.12e}")
    print(f"- penalized intercept max pred diff:   {with_int_diff:.12e}")

    print("\nFINAL DIAGNOSTIC")
    print(f"matched effective lambda convention: {best[0]}")
    print(f"max absolute prediction difference: {best[2]:.12e}")
    print(f"oracle SSE: {best[1]['sse']:.12f}")
    print(f"STreeD SSE: {st_sse:.12f}")

    try:
        tree = model.get_tree()
        coeffs = np.array(tree.label.coefficients, dtype=float)
        intercept = float(tree.label.intercept)
        print("coefficient/intercept comparison (if accessible):")
        print(f"  STreeD intercept: {intercept:.12f}")
        print(f"  oracle intercept: {best[1]['intercept']:.12f}")
        print(f"  max |coef diff|: {np.max(np.abs(coeffs - best[1]['beta'])):.12e}")
    except Exception as exc:
        print(f"coefficient/intercept unavailable from tree label: {exc}")


if __name__ == "__main__":
    run_case(ridge_penalty=1.0)
    run_case(ridge_penalty=0.0)
