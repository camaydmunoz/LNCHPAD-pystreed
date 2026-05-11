import math
import numpy as np

from pystreed import STreeDPiecewiseLinearRegressor

from a_split_b_leaf_ridge_oracle_example import (
    make_synthetic,
    standardize_continuous_like_streed,
    sorted_oracle_ab_ridge_tree,
    tree_structure,
    masks_from_streed_tree,
    objective_for_partition,
)


def fit_streed(A, B_raw, y, use_terminal_solver, lasso_penalty=0.0):
    model = STreeDPiecewiseLinearRegressor(
        max_depth=2,
        max_num_nodes=3,
        min_leaf_node_size=15,
        lasso_penalty=lasso_penalty,
        ridge_penalty=1.0,
        cost_complexity=0.0,
        random_seed=0,
        verbose=False,
    )
    model.use_terminal_solver = use_terminal_solver

    try:
        model.fit(A, y, continuous_columns=B_raw)
        pred = model.predict(A, continuous_columns=B_raw)
        mse = float(np.mean((y - pred) ** 2))
        score = model.fit_result.score() if hasattr(model, "fit_result") else None
        tree = model.get_tree()
        tree_sig = tree_structure(tree)
        masks = masks_from_streed_tree(A, tree)
        return {
            "ok": True,
            "error": None,
            "model": model,
            "pred": pred,
            "mse": mse,
            "score": score,
            "tree": tree_sig,
            "masks": masks,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": exc,
            "model": None,
            "pred": None,
            "mse": None,
            "score": None,
            "tree": None,
            "masks": None,
        }


def fmt(x):
    if x is None:
        return "None"
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.12f}"
    return str(x)


def main():
    tol = 1e-8

    A, B_raw, y = make_synthetic(n=120, seed=11)
    B_internal, _, _ = standardize_continuous_like_streed(B_raw)

    oracle_top = sorted_oracle_ab_ridge_tree(
        A=A,
        B_internal=B_internal,
        y=y,
        ridge_penalty=1.0,
        min_leaf_node_size=15,
        cost_complexity=0.0,
    )[0]

    print("== Oracle reference ==")
    print(f"oracle.structure: {oracle_top['split']}")
    print(f"oracle.objective: {oracle_top['objective']:.12f}")
    print()

    print("== STreeD run: use_terminal_solver=False ==")
    off = fit_streed(A, B_raw, y, use_terminal_solver=False, lasso_penalty=0.0)
    print(f"fit_succeeded: {off['ok']}")
    if off["ok"]:
        off_obj = objective_for_partition(B_internal, y, off["masks"], ridge_penalty=1.0)[0]
        print(f"tree_structure: {off['tree']}")
        print(f"fit_result.score: {fmt(off['score'])}")
        print(f"training_mse: {fmt(off['mse'])}")
        print(f"oracle_objective_on_streed_tree: {off_obj:.12f}")
        diff_off = abs(off_obj - oracle_top["objective"])
        if diff_off <= tol:
            print(f"PASS: terminal-off agrees with oracle within tolerance (|diff|={diff_off:.3e} <= {tol:.1e})")
        else:
            print(f"WARN: terminal-off differs from oracle (|diff|={diff_off:.3e} > {tol:.1e})")
    else:
        print(f"WARN: terminal-off fit failed with error: {off['error']}")
    print()

    print("== STreeD run: use_terminal_solver=True ==")
    on = fit_streed(A, B_raw, y, use_terminal_solver=True, lasso_penalty=0.0)
    print(f"fit_succeeded: {on['ok']}")
    if on["ok"]:
        on_obj = objective_for_partition(B_internal, y, on["masks"], ridge_penalty=1.0)[0]
        print(f"tree_structure: {on['tree']}")
        print(f"fit_result.score: {fmt(on['score'])}")
        print(f"training_mse: {fmt(on['mse'])}")
        print(f"oracle_objective_on_streed_tree: {on_obj:.12f}")
        if off["ok"]:
            diff_on_off = abs(on_obj - objective_for_partition(B_internal, y, off["masks"], ridge_penalty=1.0)[0])
            print(f"info: |terminal_on_obj - terminal_off_obj| = {diff_on_off:.3e}")
        print("WARN: before Phase 3B C++ changes, terminal-on may be effectively identical to terminal-off for piecewise-linear-regression.")
    else:
        print(f"WARN: terminal-on fit failed with error: {on['error']}")
    print()

    print("== Lasso guard probe (informational) ==")
    lasso_probe = fit_streed(A, B_raw, y, use_terminal_solver=True, lasso_penalty=0.1)
    if lasso_probe["ok"]:
        print("INFO: lasso_penalty=0.1 with terminal flag succeeded (expected pre-implementation if terminal path is ignored).")
    else:
        print(f"INFO: lasso_penalty=0.1 with terminal flag raised: {lasso_probe['error']}")
    print("INFO: after Phase 3B C++ implementation, expected behavior is clear error or explicit terminal disabling for lasso_penalty > 0.")
    print()

    print("VALIDATION SCRIPT COMPLETED")
    print(
        "After C++ Phase 3B, use_terminal_solver=True (ridge-only) should still match oracle and terminal-off numerically on this dataset, "
        "while lasso+terminal should produce explicit guard behavior."
    )


if __name__ == "__main__":
    main()
