# DDD

**DDD (Distributional Divergence Decomposition)** is a supervised dimension-reduction method for binary outcomes proposed in *Kolmogorov–Smirnov-Based Supervised Dimension Reduction for Gene Expression Classification* by Chun Yin Lee, Yuke Wang, and Kin Yau Wong. It extracts informative linear components by maximizing the two-sample **Kolmogorov–Smirnov (KS)** statistic. The constrained, nonsmooth optimization is handled by **differential evolution (DE)**.

This repository illustrates the proposed method using a **simulated dataset**, rather than requiring TCGA files. The example follows **Scenario III** of the uploaded simulation code. The original full-scale sensitivity-study script is retained under [`scripts/`](scripts/Scenario_III_NP_sensitivity_original.py).

## Method

For observations $X_i\in\mathbb R^p$ and binary labels $Y_i\in\{0,1\}$, the $m$-th DDD loading vector is estimated by

$$
\widehat z_m=\mathop{\arg\max}_{\|z\|_2=1,\;z^\top\widehat z_j=0\ (j<m)}
\sup_{t\in\mathbb R}\left|\widehat F_{z^\top X\mid Y=1}(t)-\widehat F_{z^\top X\mid Y=0}(t)\right|.
$$

Its score is $\xi_{im}=X_i^\top\widehat z_m$. DDD finds sequential **orthogonal loading vectors**; their projected scores need not be statistically uncorrelated. A larger empirical KS value indicates greater separation of the projected class-conditional empirical distributions, not necessarily better out-of-sample classification.

## How to import the functions

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

From the repository folder:

```python
from ddd import generate_data_new, fit_sequential_ddd
```

Alternatively, run the ready-to-use example:

```bash
python example_simulated.py
```

This example does not download or require any external dataset.

## Usage

The demonstration centers on **two functions**:

| Function | Description |
| --- | --- |
| `generate_data_new(N, p, seed=None)` | Generate correlated Gaussian predictors, binary outcomes, and true response probabilities for Scenario III |
| `fit_sequential_ddd(X, Y, ncomp=1, lambda_grid=None, trace=False, **de_kwargs)` | Extract successive KS-maximizing directions using differential evolution; returns loadings, penalty parameters, and empirical KS values |

Supporting functions in `ddd.py` include `objective_function`, `repair_vector`, `differential_evolution_ddd`, and `tune_next_component_cv`. The README uses the **unpenalized DDD** method (`lambda_grid=None`), as in the central formulation of the manuscript. Optional L1 penalization is retained from the uploaded source but is not part of this illustration.

### `generate_data_new`

```python
generate_data_new(N, p, seed=None)
```

This generates the Scenario III data from the uploaded simulation script:

$$
X\sim N_p(0,\Sigma),\qquad \Sigma_{ij}=0.5^{|i-j|},
$$

$$
\eta=\theta_0+(X_1+X_5)^2+\sin(X_3)+X_2X_4,
\qquad
P(Y=1\mid X)=\frac{1}{1+\exp[-3\tanh(\eta)]}.
$$

The code sets `theta0 = -median(raw_eta)` for the **generated sample** to obtain an approximately balanced outcome distribution. `p` must be at least 5.

| Argument | Description |
| --- | --- |
| `N` | Number of simulated subjects |
| `p` | Number of predictors (`p >= 5`) |
| `seed` | Random seed for reproducibility |

Returns `X` (an `N × p` NumPy array), `Y` (a binary vector of length `N`), and `ptrue` (the true conditional outcome probabilities). The source generator sets the global NumPy seed when `seed` is supplied.

### `fit_sequential_ddd`

```python
fit_sequential_ddd(
    X, Y, ncomp=1, lambda_grid=None, trace=False,
    n_pop=50, f_scale=0.5, cr=0.7, max_iter=10000
)
```

| Argument | Description |
| --- | --- |
| `X` | **Standardized** predictors, shape `N × p` |
| `Y` | Binary outcome vector coded `0` and `1` |
| `ncomp` | Number of DDD components (`1 <= ncomp <= p`) |
| `lambda_grid` | `None` for unpenalized DDD; optional candidate L1 penalties in the source |
| `trace` | Print progress messages when `True` |
| `n_pop` | DE population size (at least 4) |
| `f_scale` | DE mutation scale factor; `0.5` in the manuscript setting |
| `cr` | DE crossover probability; `0.7` in the manuscript setting |
| `max_iter` | Maximum DE generations per component; `10000` in the paper's reference configuration |

Returns `weights`, `lambdas`, and `ks_values`, where `weights` has shape `(ncomp, p)` and **each row is a unit-norm loading vector**, `lambdas` contains the chosen penalties (zero without tuning), and `ks_values` contains the final **training-sample** KS statistic of each component.

The `ddd.py` estimator is derived from the uploaded Scenario III script and retains the function names and optimizer defaults. One numerical safeguard is improved: if a candidate becomes nearly zero after orthogonal projection, the replacement is **reprojected** before normalization.

## Example: DDD with simulated Scenario III data

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from ddd import generate_data_new, fit_sequential_ddd

# Simulate 200 subjects and 10 correlated predictors.
X, Y, ptrue = generate_data_new(N=200, p=10, seed=1234)

# Use training-sample standardization before DDD.
X_scaled = StandardScaler().fit_transform(X)

# Quick illustration: 80 generations (not the paper's 10,000).
np.random.seed(2026)
weights, lambdas, ks_values = fit_sequential_ddd(
    X_scaled, Y, ncomp=3, lambda_grid=None,
    n_pop=20, f_scale=0.5, cr=0.7, max_iter=80
)

scores = X_scaled @ weights.T  # shape: (200, 3)
print("KS objectives:", np.round(ks_values, 4))
print("Loading-vector inner products:\n", np.round(weights @ weights.T, 5))
```

Expected structures:

| Object | Shape / meaning |
| --- | --- |
| `X` | `(200, 10)` simulated covariate matrix |
| `Y` | `(200,)` binary outcomes |
| `ptrue` | `(200,)` true conditional probabilities |
| `weights` | `(3, 10)` three DDD loading vectors |
| `scores` | `(200, 3)` extracted DDD scores |
| `lambdas` | Three zeros for unpenalized DDD |
| `ks_values` | Three KS values in `[0, 1]` (training data) |

For a plotted illustration and output files, run:

```bash
python example_simulated.py
```

![DDD projected scores for simulated Scenario III data](example_outputs/ddd_simulated_scores.png)

This graph is generated from a **single reproducible simulation**, not the averaged results in the research article. With the short run used here, estimates are illustrative rather than fully optimized.

## Files

| File | Description |
| --- | --- |
| `ddd.py` | Reusable core DDD code derived from the uploaded Scenario III script |
| `example_simulated.py` | Small, reproducible simulation demonstration; creates result files and a plot |
| `example_outputs/ddd_simulated_scores.png` | Example scatterplot of the first two DDD components |
| `example_outputs/ddd_simulated_scores.csv` | Labels, true probabilities, and three DDD scores |
| `example_outputs/ddd_weights.csv` | Three estimated loading vectors for the ten predictors |
| `example_outputs/summary.json` | Simulation/optimizer settings and empirical KS values |
| `scripts/Scenario_III_NP_sensitivity_original.py` | Supplied full Scenario III population-size sensitivity code (PCA, PLS, DDD, classifiers and summary files) |
| `requirements.txt` | Python dependencies |

## Reproducing the complete sensitivity study

The source file in `scripts/` performs a much larger analysis: it simulates Scenario III for `p` in `{10, 50}`, compares `N_P` in `{10, 50, 100}`, fits DDD with `G_MAX=10000` and `M=3`, and also evaluates PCA/PLS with logistic regression, GAM and SVM. Running it can require substantial computational resources:

```bash
python scripts/Scenario_III_NP_sensitivity_original.py
```

At the end of the uploaded source, **`NUM_RUNS=200` and `MAX_WORKERS=10`**. Some preexisting comments and the source filename refer to other counts; the executable constants govern the actual run. The full script is retained for provenance; the quick example does **not** reproduce its performance tables or the manuscript's 500-replicate analysis.

## Citation

Lee, C. Y., Wang, Y., and Wong, K. Y. *Kolmogorov–Smirnov-Based Supervised Dimension Reduction for Gene Expression Classification*. Manuscript.

## Notes

- The simulated data contain no actual gene-expression measurements.
- DDD uses class labels to find projections, so the plotted in-sample KS statistics are not estimates of held-out predictive performance.
- Different random seeds or DE settings can produce different loading vectors and empirical KS values. DE does not guarantee a global maximum.
- No external data files or R packages are required for the standalone illustration.
