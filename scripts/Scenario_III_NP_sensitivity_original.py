import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import time
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import comb, expit
from sklearn.svm import SVC
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
#from scipy.optimize import minimize
#from scipy.stats import norm
#from numpy.linalg import pinv


plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

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


# N      = 100000
# p      = 10
# X, Y, p_train = generate_data_new(N=N, p=p)
# np.mean(Y)

def Logit(x):
    x = np.asarray(x)
    return expit(x)

def BP_G(para, m10, x_scaled):
    safe_vals = np.clip(para[1:m10 + 1], -700, 700)
    psi = np.concatenate(([para[0]], para[0] + np.cumsum(np.exp(safe_vals))))
    x_scaled = np.clip(x_scaled, 0, 1)
    ii = np.arange(m10 + 1)
    cii = comb(m10, ii)
    x_pow = np.power.outer(x_scaled, ii)
    one_minus_x_pow = np.power.outer(1 - x_scaled, m10 - ii)
    At = cii * x_pow * one_minus_x_pow
    return np.dot(At, psi)

def bernstein_basis(x, degree):
    x = np.asarray(x)
    return np.stack(
        [comb(degree, k) * x**k * (1 - x)**(degree - k) for k in range(degree + 1)],
        axis=1
    )

def _loglik_logistic(y, pi):
    eps = 1e-12
    pi = np.clip(pi, eps, 1 - eps)
    return np.sum(y * np.log(pi) + (1 - y) * np.log(1 - pi))

def GAM_binary(
    X, y, degrees=3, tol=1e-6, max_iter=1000,
    select_degree_by_ic=True, degree_grid=range(2, 10),
    ic="AIC", ridge=1e-8,
):
    ic = ic.upper()
    if ic not in {"AIC", "BIC"}:
        raise ValueError(f"ic must be 'AIC' or 'BIC', got {ic!r}")

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, K = X.shape

    # Precompute standardization once (shared across grid search)
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    denom = X_max - X_min
    denom[denom == 0] = 1.0
    X_std = (X - X_min) / denom
    X_std[:, X_max == X_min] = 0.0

    def _solve_stable(A, b):
        if ridge and ridge > 0:
            A = A + ridge * np.eye(A.shape[0])
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(A) @ b

    def _fit_for_degrees(degrees_vec):
        degrees_vec = np.asarray(degrees_vec, int)

        basis_list = [bernstein_basis(X_std[:, j], degrees_vec[j]) for j in range(K)]
        B = np.hstack(basis_list)

        B_full = np.hstack([np.ones((n, 1)), B])
        p_dim = B_full.shape[1]  # number of parameters including intercept

        idx_list = []
        start = 0
        for deg in degrees_vec:
            idx_list.append(np.arange(start, start + deg + 1))
            start += deg + 1

        beta = np.zeros(p_dim)

        for _ in range(max_iter):
            eta = B_full @ beta
            mu = expit(eta)
            W = np.clip(mu * (1 - mu), 1e-8, None)

            z = eta + (y - mu) / (W + 1e-8)

            WX = W[:, None] * B_full
            XTWX = B_full.T @ WX
            XTWz = B_full.T @ (W * z)

            beta_new = _solve_stable(XTWX, XTWz)

            # Center each smooth (identifiability)
            for j in range(K):
                idx = 1 + idx_list[j]      # skip intercept
                B_j = B[:, idx_list[j]]
                f_j = B_j @ beta_new[idx]
                mean_fj = f_j.mean()

                B_jtBj = B_j.T @ B_j
                rhs = B_j.T @ np.full(n, mean_fj)
                adj = _solve_stable(B_jtBj, rhs)

                beta_new[idx] -= adj
                beta_new[0] += mean_fj

            if np.max(np.abs(beta_new - beta)) < tol:
                beta = beta_new
                break
            beta = beta_new

        Pi = expit(B_full @ beta)
        loglik = _loglik_logistic(y, Pi)

        k_params = int(p_dim)
        aic = 2 * k_params - 2 * loglik
        bic = np.log(n) * k_params - 2 * loglik

        def predict(X_new):
            X_new = np.asarray(X_new, float)
            denom2 = (X_max - X_min).copy()
            denom2[denom2 == 0] = 1.0
            X_new_std = (X_new - X_min) / denom2
            X_new_std = np.clip(X_new_std, 0, 1)
            X_new_std[:, X_max == X_min] = 0.0

            basis_list_new = [bernstein_basis(X_new_std[:, j], degrees_vec[j]) for j in range(K)]
            B_new = np.hstack(basis_list_new)
            B_full_new = np.hstack([np.ones((X_new.shape[0], 1)), B_new])
            return expit(B_full_new @ beta)

        return {
            'beta': beta,
            'Pi': Pi,
            'predict': predict,
            'degrees': degrees_vec,
            'X_min': X_min,
            'X_max': X_max,
            'loglik': float(loglik),
            'aic': float(aic),
            'bic': float(bic),
            'n_params': k_params,
        }

    # ---- No IC-based selection: fit once using degrees ----
    if not select_degree_by_ic:
        if isinstance(degrees, int):
            degrees_vec = [degrees] * K
        else:
            degrees_vec = degrees
        fit = _fit_for_degrees(degrees_vec)
        fit.update({'ic_used': ic, 'ridge': ridge})
        return fit

    # ---- IC-based selection: internal grid search (common d for all covariates) ----
    rows = []
    best_fit = None
    best_val = np.inf
    best_degree = None

    for d in degree_grid:
        fit_d = _fit_for_degrees([int(d)] * K)
        val = fit_d['aic'] if ic == "AIC" else fit_d['bic']

        rows.append({
            'degree': int(d),
            'loglik': fit_d['loglik'],
            'n_params': fit_d['n_params'],
            'aic': fit_d['aic'],
            'bic': fit_d['bic'],
            'ic_value': float(val),
        })

        if val < best_val:
            best_val = val
            best_fit = fit_d
            best_degree = int(d)

    best_fit = dict(best_fit)
    best_fit.update({
        'selected_degree': best_degree,
        'ic_used': ic,
        'ic_value': float(best_val),
        'ic_table': rows,
        'ridge': ridge
    })
    return best_fit

def compute_ks(beta, X, Y):
    n0 = np.sum(Y == 0)
    n1 = np.sum(Y == 1)
    if n0 == 0 or n1 == 0:
        return 0.0, 0.0
    scores = X.dot(beta)
    sorted_idx = np.argsort(scores)
    sorted_scores = scores[sorted_idx]
    sorted_y = Y[sorted_idx]
    base = np.full(len(Y), 1/n0)
    base[sorted_y==1] = -1/n1
    ks_arr = np.abs(np.cumsum(base))
    ks = float(np.max(ks_arr))
    t  = float(sorted_scores[np.argmax(ks_arr)])
    return ks, t

def orthogonal(z_l, Z_hat):
    if len(Z_hat) == 0:
        return z_l
    Z = np.stack(Z_hat, axis=1)  # shape (p, m)
    P = Z @ np.linalg.pinv(Z.T @ Z)
    return z_l - P @ (Z.T @ z_l)

def obs_like(psi, alpha, X, m10, Xmin, Xmax, Yi):
    SI1 = np.dot(X, alpha)
    SI1_scaled = (SI1 - Xmin) / (Xmax - Xmin)
    GSI = Logit(BP_G(psi, m10, SI1_scaled))
    return Yi * np.log(GSI + 1e-12) + (1 - Yi) * np.log(1 - GSI + 1e-12)


def SVM_binary(
    X, y,
    kernel="radial",
    cost=1,
    gamma=None,
    degree=3,
    scale_features=True,
    probability=True,
    **kwargs
):
    # --- dependency check ---
    try:
        _ = SVC
    except ImportError:
        raise ImportError("scikit-learn must be installed: pip install scikit-learn")

    # --- convert X to DataFrame ---
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    y = pd.Series(y).astype(str)

    # --- enforce binary {0,1} labels with '1' as positive ---
    unique_labels = sorted(y.unique())
    if set(unique_labels) != {"0", "1"}:
        raise ValueError("Response must be binary (0/1) for clear probability interpretation.")
    # ensure ordering c("0","1")
    y = pd.Categorical(y, categories=["0", "1"], ordered=True)

    n_features = X.shape[1]

    # --- automatic gamma ---
    if gamma is None:
        if kernel in ["radial", "polynomial", "sigmoid"]:
            gamma = 1.0 / n_features
        else:
            gamma = 0.0

    # --- scale features ---
    X_center = X.mean(axis=0)
    X_scale = X.std(axis=0, ddof=0)
    X_scale[X_scale == 0] = 1
    if scale_features:
        Xs = (X - X_center) / X_scale
    else:
        Xs = X.copy()

    # --- fit svm ---
    sklearn_kernel = {
        "radial": "rbf",
        "polynomial": "poly",
        "sigmoid": "sigmoid",
        "linear": "linear"
    }.get(kernel, kernel)

    svm_fit = SVC(
        kernel=sklearn_kernel,
        C=cost,
        gamma=gamma,
        degree=degree,
        probability=probability,
        **kwargs
    )
    svm_fit.fit(Xs, y)

    # --- prediction function returning P(y=1) only ---
    def predict_fun(X_new):
        X_new = pd.DataFrame(X_new)
        if scale_features:
            X_new = (X_new - X_center) / X_scale
        probs = svm_fit.predict_proba(X_new)
        # second column corresponds to class "1"
        return probs[:, 1]

    # --- construct output structure ---
    return {
        "model": svm_fit,
        "kernel": kernel,
        "cost": cost,
        "gamma": gamma,
        "degree": degree,
        "scale": X_scale,
        "center": X_center,
        "scaled": scale_features,
        "predict": predict_fun
    }


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
    """
    Projects u onto the null space of previous_components 
    and normalizes to unit length.
    """
    # 1. Orthogonalize (Gram-Schmidt) against fixed previous components
    if previous_components is not None and len(previous_components) > 0:
        Q = np.vstack(previous_components)
        # u = u - sum( (u . q_j) * q_j ) assuming q_j are orthonormal
        u = u - Q.T @ (Q @ u)
    
    # 2. Normalize
    norm = np.linalg.norm(u)
    if norm < 1e-10:
        # Handle edge case where vector vanishes or is zero
        u = np.random.randn(u.shape[0])
        u = u / np.linalg.norm(u)
    else:
        u = u / norm
    return u

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


def logistic_loss(y_true, y_pred_prob):
    y_true = np.array(y_true)
    y_pred = np.clip(y_pred_prob, 1e-15, 1 - 1e-15)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

def rmse(p_true, y_pred_prob):
    return np.sqrt(np.mean((p_true - y_pred_prob)**2))

def f1_score(y_true, y_pred_prob, threshold=0.5):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_prob) >= threshold
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    precision = TP / (TP + FP + 1e-15)
    recall = TP / (TP + FN + 1e-15)
    f1 = 2 * precision * recall / (precision + recall + 1e-15)
    return f1

def auc_score(y_true, y_pred_prob):
    y_true = np.array(y_true)
    y_pred_prob = np.array(y_pred_prob)
    n0 = np.sum(y_true == 0)
    n1 = np.sum(y_true == 1)
    pos_scores = y_pred_prob[y_true == 1]
    neg_scores = y_pred_prob[y_true == 0]
    if n1 == 0 or n0 == 0:
        return np.nan
    count = 0
    for psc in pos_scores:
        count += np.sum(psc > neg_scores)
    auc = count / (n1 * n0)
    return auc


# ---------- One simulation replicate for N_P sensitivity ----------
def run_one(q, N=200, p=10, comp=3):
    """Run one paired replicate for N_P in {25, 50, 100}.

    The same training and test datasets are reused across all three
    population sizes. PCA and PLS are fit once per replicate because
    they do not depend on N_P. DDD is refit independently for each N_P.
    """
    # Fixed data seeds: the same q defines the same simulated replicate
    # for every N_P value.
    X, Y, p_train = generate_data_new(N=N, p=p, seed=q)
    X_test, Y_test, p_test = generate_data_new(N=10000, p=p, seed=1000000 + q*2026)

    # Preserve the original simulation preprocessing.
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ---------------------------------------------------------------
    # Fit PCA and PLS once. Their estimates do not depend on N_P.
    # ---------------------------------------------------------------
    pca = PCA(n_components=comp)
    Z = pca.fit_transform(X)
    PCA_comp = pca.components_
    PCA_test = X_test @ np.array(PCA_comp).T

    pls = PLSRegression(n_components=comp)
    pls.fit(X, Y)
    W = pls.x_scores_
    PLS_comp = pls.x_weights_.T
    PLS_test = X_test @ np.array(PLS_comp).T

    def fit_and_predict(train_scores, test_scores, seed):
        """Fit the same downstream LR, GAM, and SVM models as the main study."""
        np.random.seed(seed)

        lr = LogisticRegression(max_iter=1000, solver='lbfgs')
        lr.fit(train_scores, Y)
        lr_pred = lr.predict_proba(test_scores)[:, 1]

        gam = GAM_binary(train_scores, Y, ic="BIC")
        gam_pred = gam['predict'](test_scores)

        svm = SVM_binary(train_scores, Y)
        svm_pred = svm['predict'](test_scores)

        return lr_pred, gam_pred, svm_pred

    def collect_metrics(prefix, lr_pred, gam_pred, svm_pred):
        return {
            f"ll_LR_{prefix}": logistic_loss(Y_test, lr_pred),
            f"rmse_LR_{prefix}": rmse(p_test, lr_pred),
            f"f1_LR_{prefix}": f1_score(Y_test, lr_pred),
            f"AUC_LR_{prefix}": auc_score(Y_test, lr_pred),
            f"ll_GAM_{prefix}": logistic_loss(Y_test, gam_pred),
            f"rmse_GAM_{prefix}": rmse(p_test, gam_pred),
            f"f1_GAM_{prefix}": f1_score(Y_test, gam_pred),
            f"AUC_GAM_{prefix}": auc_score(Y_test, gam_pred),
            f"ll_SVM_{prefix}": logistic_loss(Y_test, svm_pred),
            f"rmse_SVM_{prefix}": rmse(p_test, svm_pred),
            f"f1_SVM_{prefix}": f1_score(Y_test, svm_pred),
            f"AUC_SVM_{prefix}": auc_score(Y_test, svm_pred),
        }

    # These reference results are identical for all N_P values within q.
    base_eval_seed = 20000000 + 100000 * p + 2026 * q

    lr_pca, gam_pca, svm_pca = fit_and_predict(
        Z, PCA_test, base_eval_seed + 200
    )
    pca_metrics = collect_metrics("PCA", lr_pca, gam_pca, svm_pca)

    lr_pls, gam_pls, svm_pls = fit_and_predict(
        W, PLS_test, base_eval_seed + 300
    )
    pls_metrics = collect_metrics("PLS", lr_pls, gam_pls, svm_pls)

    results = []

    # ---------------------------------------------------------------
    # N_P sensitivity: only the DE population size changes.
    # G_MAX is fixed at 10,000 for all three values of N_P.
    # ---------------------------------------------------------------
    init_seed = 40000000 + 100000 * p + 2026 * q
    
    for np_index, n_pop in enumerate(NP_VALUES):

        optimizer_seed = (
            10000000 + 100000 * p + 2026 * q + np_index
        )
        np.random.seed(optimizer_seed)
    
        DDD_comp, used_lambdas, KS_comp = fit_sequential_ddd(
            X, Y, comp,
            lambda_grid=None,
            n_pop=n_pop,
            max_iter=G_MAX,
            init_seed=init_seed
        )
        DDD = np.dot(X, np.array(DDD_comp).T)
        DDD_test = X_test @ np.array(DDD_comp).T

        # Independent downstream seed for the DDD classifier fits.
        lr_ddd, gam_ddd, svm_ddd = fit_and_predict(
            DDD, DDD_test, base_eval_seed + 100 + np_index
        )
        ddd_metrics = collect_metrics(
            "DDD", lr_ddd, gam_ddd, svm_ddd
        )

        # Final training-sample KS objective for each sequential component.
        KS_comp = np.asarray(KS_comp, dtype=float)
        row = {
            "replicate": q,
            "p": p,
            "N_P": n_pop,
            "KS1": KS_comp[0] if len(KS_comp) > 0 else np.nan,
            "KS2": KS_comp[1] if len(KS_comp) > 1 else np.nan,
            "KS3": KS_comp[2] if len(KS_comp) > 2 else np.nan,
            "KS_mean": np.mean(KS_comp),
        }
        row.update(ddd_metrics)
        row.update(pca_metrics)
        row.update(pls_metrics)
        results.append(row)

    return results

# Parameters for Reviewer 1 N_P sensitivity analysis
P_VALUES = [10, 50]
N = 200
COMP = 3
NUM_RUNS = 200
NP_VALUES = [10, 50, 100]
G_MAX = 10000
MAX_WORKERS = 10


def run_scenario_parallel():
    """Run independent simulation replicates in parallel using 20 worker processes.

    Joblib's loky backend is used because the script is run on Windows/Spyder.
    Each worker is restricted to one numerical thread by the environment
    settings at the top of the script, so up to 20 replicates run concurrently.
    """
    start = time.perf_counter()
    all_results = []

    for p in P_VALUES:
        print(f"\nScenario III -- p = {p}, N = {N}")

        q_values = list(range(NUM_RUNS))

        # One worker process handles one complete simulation replicate.
        # batch_size=1 keeps the scheduling aligned with the replicate count.
        replicate_results = Parallel(
            n_jobs=MAX_WORKERS,
            backend="loky",
            batch_size=1,
            verbose=10,
        )(
            delayed(run_one)(q=q, N=N, p=p, comp=COMP)
            for q in q_values
        )

        for result in replicate_results:
            all_results.extend(result)

    elapsed = time.perf_counter() - start
    return all_results, elapsed


def main():
    all_results, elapsed = run_scenario_parallel()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        "scenario_III_NP_sensitivity_replicates.csv",
        index=False,
    )


    metric_names = [
        "KS1", "KS2", "KS3", "KS_mean",
        "ll_LR_DDD", "ll_LR_PCA", "ll_LR_PLS",
        "rmse_LR_DDD", "rmse_LR_PCA", "rmse_LR_PLS",
        "f1_LR_DDD", "f1_LR_PCA", "f1_LR_PLS",
        "AUC_LR_DDD", "AUC_LR_PCA", "AUC_LR_PLS",
        "ll_GAM_DDD", "ll_GAM_PCA", "ll_GAM_PLS",
        "rmse_GAM_DDD", "rmse_GAM_PCA", "rmse_GAM_PLS",
        "f1_GAM_DDD", "f1_GAM_PCA", "f1_GAM_PLS",
        "AUC_GAM_DDD", "AUC_GAM_PCA", "AUC_GAM_PLS",
        "ll_SVM_DDD", "ll_SVM_PCA", "ll_SVM_PLS",
        "rmse_SVM_DDD", "rmse_SVM_PCA", "rmse_SVM_PLS",
        "f1_SVM_DDD", "f1_SVM_PCA", "f1_SVM_PLS",
        "AUC_SVM_DDD", "AUC_SVM_PCA", "AUC_SVM_PLS",
    ]

    summary_rows = []
    for (p, n_pop), group in results_df.groupby(["p", "N_P"], sort=True):
        row = {"p": p, "N_P": n_pop}
        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        "scenario_III_NP_sensitivity_summary.csv",
        index=False,
    )

    print("\nMean final KS objectives across the 1000 paired replicates:")
    print(
        summary_df[[
            "p", "N_P", "KS1_mean", "KS2_mean", "KS3_mean", "KS_mean_mean"
        ]].round(4).to_string(index=False)
    )

    for model_prefix in ["LR", "GAM", "SVM"]:
        print(f"\n{model_prefix}: mean AUC")
        auc_rows = summary_df[[
            "p", "N_P",
            f"AUC_{model_prefix}_PCA_mean",
            f"AUC_{model_prefix}_PLS_mean",
            f"AUC_{model_prefix}_DDD_mean",
        ]].copy()
        auc_rows.columns = ["p", "N_P", "PCA", "PLS", "DDD"]
        print(auc_rows.round(4).to_string(index=False))

    print(f"\nElapsed: {elapsed:.3f}s")
    print("Saved:")
    print("  scenario_III_NP_sensitivity_replicates.csv")
    print("  scenario_III_NP_sensitivity_summary.csv")


if __name__ == "__main__":
    main()


