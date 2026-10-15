"""Fast, reproducible illustration of DDD on simulated Scenario III data.

Run: python example_simulated.py
Outputs are created in example_outputs/.
This is a demonstration, NOT the paper's 10,000-generation experiment.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from ddd import generate_data_new, fit_sequential_ddd


def main():
    out = Path("example_outputs")
    out.mkdir(exist_ok=True)

    # Scenario III in the uploaded script. Y is binary, p_true is known.
    X, y, p_true = generate_data_new(N=200, p=10, seed=1234)
    X_scaled = StandardScaler().fit_transform(X)

    # Smaller settings for a quick GitHub demonstration. The full study uses
    # n_pop=50 and max_iter=10000 in its reference DDD configuration.
    np.random.seed(2026)
    weights, lambdas, ks_values = fit_sequential_ddd(
        X_scaled, y, ncomp=3, lambda_grid=None, trace=False,
        n_pop=20, f_scale=0.5, cr=0.7, max_iter=80
    )
    scores = X_scaled @ weights.T

    # Orient scores as higher for Y=1 for display only; KS is sign invariant.
    for m in range(weights.shape[0]):
        if scores[y == 1, m].mean() < scores[y == 0, m].mean():
            weights[m] *= -1
            scores[:, m] *= -1

    gram = weights @ weights.T
    assert np.allclose(gram, np.eye(3), atol=1e-8), gram

    print(f"n={len(y)}, p={X.shape[1]}, class 1 count={int(y.sum())}")
    print("Final empirical KS statistics:", np.round(ks_values, 4))
    print("Projection norms:", np.round(np.linalg.norm(weights, axis=1), 6))
    print("Maximum orthogonality error:", round(float(np.max(np.abs(gram - np.eye(3)))), 12))
    print("Penalty lambdas:", lambdas)

    df = pd.DataFrame(scores, columns=["DDD1", "DDD2", "DDD3"])
    df.insert(0, "Y", y)
    df.insert(1, "p_true", p_true)
    df.to_csv(out / "ddd_simulated_scores.csv", index=False)
    pd.DataFrame(weights, columns=[f"X{j+1}" for j in range(X.shape[1])],
                 index=["DDD1", "DDD2", "DDD3"]).to_csv(out / "ddd_weights.csv")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for cls, label, color in [(0, "Y=0", "#3567A5"), (1, "Y=1", "#E07A45")]:
        ax.scatter(scores[y == cls, 0], scores[y == cls, 1], s=26,
                   alpha=0.7, label=label, color=color)
    ax.set_xlabel("DDD component 1")
    ax.set_ylabel("DDD component 2")
    ax.set_title("Simulated Scenario III — DDD projected scores")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "ddd_simulated_scores.png", dpi=160)
    plt.close(fig)

    metadata = {
        "scenario": "III", "n": int(len(y)), "p": int(X.shape[1]),
        "M": 3, "n_pop": 20, "f_scale": 0.5, "cr": 0.7,
        "max_iter": 80, "data_seed": 1234, "optimizer_seed": 2026,
        "n_class_1": int(y.sum()),
        "KS_objectives": list(map(float, ks_values)),
        "orthogonality_max_error": float(np.max(np.abs(gram - np.eye(3))))
    }
    (out / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Outputs saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
