"""DDD simulated-data illustration, derived from the supplied Scenario III code.

Core estimator and data generator retain the source function names and
algorithmic defaults. This module does not run large simulations on import.
"""
import numpy as np
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold

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


def objective_function(z, X0, X1, penalty_lambda=0.0):
    """
    Calculates the penalized KS statistic for data projected onto z.
    Objective = KS_Stat - lambda * ||z||_1
    """
    proj0 = X0 @ z
    proj1 = X1 @ z
    stat, _ = ks_2samp(proj0, proj1)
    
    # Regularization: 
    # We maximize (KS - penalty). 
    # L1 norm (Lasso) encourages sparsity.
    penalty = penalty_lambda * np.sum(np.abs(z))
    
    return stat - penalty


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
    max_iter=10000, penalty_lambda=0.0,
    init_seed=None
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
        fitness[i] = objective_function(pop[i], X0, X1, penalty_lambda)
        
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
            f_new = objective_function(u, X0, X1, penalty_lambda)
            
            if f_new > fitness[i]:
                pop[i] = u
                fitness[i] = f_new
                if f_new > best_score:
                    best_score = f_new
                    best_z = u.copy()
                    
    return best_z


def tune_next_component_cv(X, Y, prev_comps, lambda_grid, trace, n_folds=5, **de_kwargs):
    """
    Performs CV to find the best lambda for the *current* component (k),
    given that components 1 to k-1 (prev_comps) are fixed.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {}

    # Reduce DE computational load during CV (optional but recommended)
    cv_de_kwargs = de_kwargs.copy()
    if 'n_pop' not in cv_de_kwargs: cv_de_kwargs['n_pop'] = 20
    if 'max_iter' not in cv_de_kwargs: cv_de_kwargs['max_iter'] = 20
    
    if(trace):
        print(f"    > Tuning lambda for component {len(prev_comps) + 1}...")

    for lam in lambda_grid:
        fold_scores = []
        
        for train_idx, test_idx in skf.split(X, Y):
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]
            
            X0_tr = X_train[Y_train == 0]
            X1_tr = X_train[Y_train == 1]
            
            # 1. Estimate direction on Training Data
            # Note: We pass 'prev_comps' here. The new direction must be 
            # orthogonal to the GLOBAL previous components.
            z_fold = differential_evolution_ddd(
                X0_tr, X1_tr, prev_comps=prev_comps, 
                penalty_lambda=lam, **cv_de_kwargs
            )
            
            # 2. Evaluate on Test Data (Unpenalized KS)
            X0_te = X_test[Y_test == 0]
            X1_te = X_test[Y_test == 1]
            score = objective_function(z_fold, X0_te, X1_te, penalty_lambda=0.0)
            fold_scores.append(score)
            
        avg_score = np.mean(fold_scores)
        results[lam] = avg_score
        # print(f"      [Lambda {lam:.2f}] CV Score: {avg_score:.4f}")

    best_lambda = max(results, key=results.get) 
    if(trace):
        print(f" Selected Lambda: {best_lambda} (CV Score: {results[best_lambda]:.4f})")
    return best_lambda


def fit_sequential_ddd(X, Y, ncomp=1, lambda_grid=None, trace=False, **de_kwargs):
    """
    Parameters:
    - lambda_grid: List of floats. If provided, CV is performed for each component.
                   If None, lambda=0 is used for all components.
    - de_kwargs: Arguments passed to differential_evolution_ddd (n_pop, max_iter).
    """
    X0 = X[Y==0]
    X1 = X[Y==1]
    
    found_vectors = []
    final_lambdas = []
    final_scores = []
    
    if(trace):
        print(f"Starting Sequential DDD for {ncomp} components...")
    
    for k in range(ncomp):
        if(trace):
            print(f"\n--- Component {k+1} ---")
        
        # 1. Determine Lambda for this component
        if lambda_grid is not None and len(lambda_grid) > 0:
            # Run CV to find optimal lambda_k given z_1...z_{k-1}
            best_lam = tune_next_component_cv(
                X, Y, prev_comps=found_vectors, trace=trace,
                lambda_grid=lambda_grid, n_folds=5, **de_kwargs
            )
        else:
            best_lam = 0.0
            if(trace):
                print("      No lambda_grid provided. Using lambda=0.0")
            
        final_lambdas.append(best_lam)
        
        # 2. Fit Final Component on Full Data
        if(trace):
            print(f"    > Fitting final vector with lambda={best_lam}...")
        z_k = differential_evolution_ddd(
            X0, X1, prev_comps=found_vectors, 
            penalty_lambda=best_lam, **de_kwargs
        )
        
        # 3. Store Results
        found_vectors.append(z_k)
        
        # Calculate scores for reporting
        raw_ks = objective_function(z_k, X0, X1, penalty_lambda=0.0)
        final_scores.append(raw_ks)
        
        if(trace):
            print(f" > Done. Raw KS: {raw_ks:.4f}")
        
    return np.array(found_vectors), final_lambdas, np.array(final_scores)
