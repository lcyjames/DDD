# DDD

**DDD (Distributional Divergence Decomposition)** is a supervised dimension-reduction method for binary outcomes proposed in *Kolmogorov–Smirnov-Based Supervised Dimension Reduction for Gene Expression Classification* by Chun Yin Lee, Yuke Wang, and Kin Yau Wong. DDD extracts linear combinations of predictors that maximize the empirical two-sample **Kolmogorov–Smirnov (KS)** statistic. The constrained, non-differentiable optimization is performed by **differential evolution (DE)**.

This repository provides a **single simulated dataset illustration**: generate binary-response data, fit DDD, and examine the extracted components. No TCGA data are required.

## How to import the functions

Install Python 3 and the dependencies from the repository folder:

```bash
python -m pip install -r requirements.txt
```

Import the functions in Python:

```python
from ddd import generate_data_new, fit_sequential_ddd
```

Alternatively, run the complete demonstration:

```bash
python example_simulated.py
```

## Usage

The module provides two main functions:

| Function | Description |
| --- | --- |
| `generate_data_new(N, p, seed=None)` | Generate **one** dataset with correlated Gaussian predictors and a binary response |
| `fit_sequential_ddd(X, Y, ncomp=1, trace=False, ...)` | Estimate successive unpenalized DDD loading vectors and their empirical KS values |

Internally, `ddd.py` also defines `objective_function`, `repair_vector`, and `differential_evolution_ddd` to evaluate the KS criterion and perform constrained differential evolution.

### `generate_data_new`

```python
generate_data_new(N, p, seed=None)
```

The example uses the nonlinear data-generating model (Scenario III) from the manuscript:

$$
X\sim N_p(0,\Sigma), \qquad \Sigma_{ij}=0.5^{|i-j|},
$$

$$
\eta=\theta_0+(X_1+X_5)^2+\sin(X_3)+X_2X_4,
\qquad
P(Y=1\mid X)=\frac{1}{1+\exp[-3\tanh(\eta)]}.
$$

The code chooses `theta0 = -np.median(raw_eta)` within the generated sample to obtain approximately balanced classes.

| Argument | Description |
| --- | --- |
| `N` | Number of observations |
| `p` | Number of predictors; **at least 5** |
| `seed` | Random seed for reproducible data generation |

Returns `X` (shape `N × p`), `Y` (binary vector of length `N`), and `ptrue` (true response probabilities).

### `fit_sequential_ddd`

```python
fit_sequential_ddd(
    X, Y, ncomp=1, trace=False,
    n_pop=50, f_scale=0.5, cr=0.7,
    max_iter=10000, init_seed=None
)
```

Given standardized predictors and class labels, the method estimates the loading vectors sequentially:

$$
\widehat z_m=\underset{\|z\|_2=1,\ z^\top\widehat z_j=0\ (j<m)}{\arg\max}
\sup_t\left|\widehat F_{z^\top X\mid Y=1}(t)-\widehat F_{z^\top X\mid Y=0}(t)\right|.
$$

Each loading vector has unit Euclidean norm and is orthogonal to previously extracted loading vectors. Orthogonality of loadings **does not guarantee uncorrelated component scores**. The objective is unpenalized; there is **no lambda argument**.

| Argument | Description |
| --- | --- |
| `X` | Standardized predictor matrix, shape `N × p` |
| `Y` | Binary response vector containing both 0 and 1 |
| `ncomp` | Number of extracted directions (`1 <= ncomp <= p`) |
| `trace` | Print empirical KS values during fitting |
| `n_pop` | Number of DE candidates per generation (at least 4) |
| `f_scale` | Differential-evolution mutation scale (default `0.5`) |
| `cr` | Crossover probability (default `0.7`) |
| `max_iter` | Number of DE generations **per component** (default `10000`) |
| `init_seed` | Optional initialization seed for DE candidates; the remaining random draws use NumPy's active RNG |

Returns `weights`, an array of shape `(ncomp, p)` with one unit-norm **fitted loading vector** per row, and `ks_values`, the corresponding **in-sample** two-sample KS statistics. The primary data output, the `(N, ncomp)` **DDD score matrix**, is computed using `scores = X @ weights.T`. Both full arrays are printed or saved in the runnable example below.

## Example: generate one dataset and estimate DDD

The **main results** of DDD are (1) the fitted **loading vectors**, which define the projections, and (2) the **component scores**, which give a lower-dimensional representation of each observation. The empirical KS statistics and the inner products of the loading vectors are useful diagnostics, but are not the main outputs.

To run the complete example, open `example_simulated.py` in **Spyder** (with the working directory set to this repository) and press **F5**, or run:

```bash
python example_simulated.py
```

Alternatively, run this example interactively:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ddd import generate_data_new, fit_sequential_ddd

# 1. Generate one simulated dataset (200 observations, 10 predictors).
X, Y, ptrue = generate_data_new(N=200, p=10, seed=1234)

# 2. Standardize the predictors.
X_scaled = StandardScaler().fit_transform(X)

# 3. Estimate three DDD projection directions.
#    Short optimizer settings are used for this illustration.
np.random.seed(2026)
weights, ks_values = fit_sequential_ddd(
    X_scaled, Y, ncomp=3,
    n_pop=20, f_scale=0.5, cr=0.7, max_iter=80
)

# 4. Compute the component scores for every observation.
scores = X_scaled @ weights.T

# 5. Orient each component so that Y=1 has the higher mean score.
#    Reversing a loading vector's sign does not affect its KS statistic.
for m in range(weights.shape[0]):
    if scores[Y == 1, m].mean() < scores[Y == 0, m].mean():
        weights[m] *= -1
        scores[:, m] *= -1

# 6. Present the main outputs as tables.
loading_df = pd.DataFrame(
    weights,
    index=["DDD1", "DDD2", "DDD3"],
    columns=[f"X{j+1}" for j in range(X.shape[1])]
)
score_df = pd.DataFrame(scores, columns=["DDD1", "DDD2", "DDD3"])
score_df.insert(0, "Y", Y)
score_df.insert(1, "p_true", ptrue)

print("Estimated DDD loading vectors:\n", loading_df.round(4).to_string())
print("\nFirst five observations' scores:\n", score_df.head().round(4).to_string(index=False))
print("\nEmpirical KS statistics:", np.round(ks_values, 4))
print("Loading-vector inner products:\n", np.round(weights @ weights.T, 5))
```

### Output 1: Estimated DDD loading vectors

The fitted loading matrix has **three rows (DDD components)** and **ten columns (predictors)**:

```text
          X1      X2      X3      X4      X5      X6      X7      X8      X9     X10
DDD1 -0.3575  0.1136  0.7283  0.3304 -0.3558 -0.1096 -0.2338  0.0569  0.1155 -0.0992
DDD2 -0.5016 -0.5964 -0.3075 -0.2629 -0.3016 -0.1777 -0.2735 -0.0677 -0.0314 -0.1617
DDD3  0.1493  0.0641 -0.0834 -0.0152 -0.1571 -0.4079 -0.1579 -0.4276  0.4840  0.5773
```

Each row defines a linear combination of the standardized predictors. For example, the first DDD score for observation `i` is

```math
\widehat{\xi}_{i,1}
= -0.3575\,X^{\ast}_{i,1}
+ 0.1136\,X^{\ast}_{i,2}
+ \cdots
- 0.0992\,X^{\ast}_{i,10}.
```

where $X^*$ denotes standardized predictors. These coefficients are **estimated by DDD**, not known or preassigned true weights. Their signs are arbitrary up to a component-wise reversal.

### Output 2: Estimated DDD component scores

The fitted score matrix has **200 rows (observations)** and **three columns (components)**. The example shows the first five observations:

```text
 Y  p_true    DDD1    DDD2    DDD3
 1  0.9526  0.3929  0.7659  0.9489
 0  0.0517 -0.0807  0.1543 -0.5595
 1  0.1775 -0.7144 -0.1441  1.1293
 1  0.9521  0.2794 -1.2215  1.5975
 0  0.1041 -1.5284 -0.1547 -1.0349
```

`Y` is the simulated class label and `p_true` is its known data-generating probability (available only because this is simulated data). The three DDD columns are **the actual dimension-reduced features** that can subsequently be used in exploratory plots or a classifier.

### Output 3: KS statistics and orthogonality diagnostics

```text
Empirical KS statistics: [0.4366 0.3043 0.2747]
Loading-vector inner products:
 [[1. 0. 0.]
  [0. 1. 0.]
  [0. 0. 1.]]
```

The KS statistics measure **in-sample distributional separation** of the projected scores by the two classes. The inner-product matrix verifies that the estimated loading vectors have unit Euclidean norm and are mutually orthogonal; it does **not** mean that the three score columns must be uncorrelated.

### Output files

Running `example_simulated.py` saves the complete matrices, not just the previews printed in the console:

| File | Contents |
| --- | --- |
| `example_outputs/ddd_weights.csv` | All 3 × 10 fitted DDD loading coefficients; rows are components and columns are predictors |
| `example_outputs/ddd_simulated_scores.csv` | All 200 projected score rows, with simulated `Y` and `p_true` |
| `example_outputs/ddd_simulated_scores.png` | Scatterplot of the first two DDD component scores, colored by class |
| `example_outputs/summary.json` | Simulation settings, empirical KS statistics, and orthogonality check |

![First two DDD scores for the simulated observations](example_outputs/ddd_simulated_scores.png)

**Reproducibility:** The numerical values above use the provided seeds and the short demonstration settings (`n_pop=20`, `max_iter=80`). The manuscript's reference optimizer settings use `n_pop=50` and `max_iter=10000`. Because differential evolution is stochastic and numerical library versions can vary, exact loadings may vary across environments even with the same seeds.

## Files

| File | Description |
| --- | --- |
| `ddd.py` | Data generator and DDD estimator, with internal optimization functions |
| `example_simulated.py` | End-to-end simulation and DDD illustration on **one dataset** |
| `requirements.txt` | Required Python packages |
| `example_outputs/ddd_simulated_scores.csv` | All observations: simulated labels, true probabilities, and three DDD scores |
| `example_outputs/ddd_weights.csv` | All three fitted loading vectors (3 × 10 coefficients) |
| `example_outputs/ddd_simulated_scores.png` | Scatterplot of the first two DDD components |
| `example_outputs/summary.json` | Example parameters, KS objectives, and orthogonality check |

## Citation

Lee, C. Y., Wang, Y., and Wong, K. Y. *Kolmogorov–Smirnov-Based Supervised Dimension Reduction for Gene Expression Classification*. Manuscript.

## Notes

- This illustration uses simulated data, not patient measurements.
- The optimizer is stochastic and does not guarantee a global maximizer.
- Higher empirical KS values indicate stronger *in-sample* distributional separation; they do not alone establish predictive accuracy.
- No external data files or R packages are required.
