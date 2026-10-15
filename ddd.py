"""DDD simulated-data illustration, derived from the supplied Scenario III code.

Core estimator and data generator retain the source function names and
algorithmic defaults. This module does not run large simulations on import.
"""
import numpy as np
from scipy.stats import ks_2samp

def generate_data_new(N, p, seed=None):
    np.random.seed(seed)
    cov = np.power(0.5, np.abs(np.arange(p)[:, None] - np.arange(p)[None, :]))
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=cov, size=N)    
    raw_eta = (
        (X[:, 0] + X[:, 4])**2 + np.sin(X[:, 2]) + X[:, 1]*X[:, 3]  # Manifold
    )
    theta0 = -np.median(raw_eta)
    eta = theta0 + raw_eta
    ptrue = 1 / (1 + np.exp(-3 * np.tanh(eta)))
    Y = np.random.binomial(1, ptrue)    
    return X, Y, ptrue


def objective_function(z, X0, X1):
    """Two-sample empirical KS statistic of the scores projected onto z."""
    return float(ks_2samp(X0 @ z, X1 @ z).statistic)


def repair_vector(u, previous_components):
    """Project onto the complement of previous directions; return a unit vector.

    For a nearly zero projection, regenerate *and reproject* the vector, unlike
    the unguarded edge-case fallback in the original sensitivity script.
    """
    u = np.asarray(u, dtype=float).copy()
    Q = np.vstack(previous_components) if previous_components is not None and len(previous_components) else None
    for _ in range(20):
        if Q is not None:
            u = u - Q.T @ (Q @ u)
        norm = np.linalg.norm(u)
        if np.isfinite(norm) and norm >= 1e-10:
            return u / norm
        u = np.random.randn(u.shape[0])
    raise RuntimeError("Unable to generate a unit vector orthogonal to prior components")


def differential_evolution_ddd(X0, X1, prev_comps,
    n_pop=50, f_scale=0.5, cr=0.7,
    max_iter=10000, init_seed=None
):
    p = X0.shape[1]

    # Nested initial populations for a common initialization seed
    if init_seed is not None:
        init_rng = np.random.RandomState(init_seed)
        pop = init_rng.randn(n_pop, p)
    else:
        pop = np.random.randn(n_pop, p)

    fitness = np.zeros(n_pop)
    
    # Initial repair and evaluation
    for i in range(n_pop):
        pop[i] = repair_vector(pop[i], prev_comps)
        fitness[i] = objective_function(pop[i], X0, X1)
        
    best_idx = np.argmax(fitness)
    best_z = pop[best_idx].copy()
    best_score = fitness[best_idx]
    
    # Evolution Loop
    for g in range(max_iter):
        for i in range(n_pop):
            # Mutation indices
            idxs = [idx for idx in range(n_pop) if idx != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            
            # Mutation
            v = pop[r1] + f_scale * (pop[r2] - pop[r3])
            
            # Crossover
            cross_points = np.random.rand(p) < cr
            if not np.any(cross_points): 
                cross_points[np.random.randint(0, p)] = True
            u = np.where(cross_points, v, pop[i])
            
            # Constraint Repair (The Critical Step)
            # Ensures u is orthogonal to all previously found z_1...z_{k-1}
            u = repair_vector(u, prev_comps)
            
            # Selection
            f_new = objective_function(u, X0, X1)
            
            if f_new > fitness[i]:
                pop[i] = u
                fitness[i] = f_new
                if f_new > best_score:
                    best_score = f_new
                    best_z = u.copy()
                    
    return best_z


def fit_sequential_ddd(X, Y, ncomp=1, trace=False, *,
                       n_pop=50, f_scale=0.5, cr=0.7,
                       max_iter=10000, init_seed=None):
    """Estimate sequential unpenalized DDD components.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Predictor matrix, normally standardized on the training data.
    Y : array-like, shape (n_samples,)
        Binary labels coded 0 (control) and 1 (case).
    ncomp : int, default 1
        Number of mutually orthogonal loading directions to extract.
    trace : bool, default False
        Print per-component KS statistics.
    n_pop, f_scale, cr, max_iter :
        Differential-evolution settings. The research configuration uses
        n_pop=50, f_scale=0.5, cr=0.7 and max_iter=10000.
    init_seed : int or None
        Optional seed used to initialize the DE population for each component.
        Other DE random draws use NumPy's current global RNG state.

    Returns
    -------
    weights : ndarray, shape (ncomp, n_features)
        Estimated unit-norm loading vectors, one vector per row.
    ks_values : ndarray, shape (ncomp,)
        Empirical (in-sample) two-sample KS objective for each component.
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y)
    if X.ndim != 2 or Y.ndim != 1 or X.shape[0] != Y.size:
        raise ValueError("X must be 2D and Y must have one label per row")
    n, p = X.shape
    if not (1 <= ncomp <= p):
        raise ValueError("ncomp must be between 1 and the number of predictors")
    if not (np.any(Y == 0) and np.any(Y == 1)) or not np.all(np.isin(Y, [0, 1])):
        raise ValueError("Y must contain both binary classes 0 and 1")
    if n_pop < 4:
        raise ValueError("n_pop must be at least 4")

    X0, X1 = X[Y == 0], X[Y == 1]
    found_vectors = []
    ks_values = []

    if trace:
        print(f"Starting sequential DDD for {ncomp} components")

    for k in range(ncomp):
        z_k = differential_evolution_ddd(
            X0, X1, prev_comps=found_vectors,
            n_pop=n_pop, f_scale=f_scale, cr=cr,
            max_iter=max_iter, init_seed=init_seed,
        )
        found_vectors.append(z_k)
        score = objective_function(z_k, X0, X1)
        ks_values.append(score)
        if trace:
            print(f"  Component {k + 1}: KS={score:.4f}")

    return np.stack(found_vectors), np.asarray(ks_values)
