"""Exploratory DDD loading analysis for TCGA LUAD versus LUSC.

This is a *whole-cohort descriptive fit*, NOT a cross-validated predictive
assessment. It uses only DDD (no PCA, PLS, P2WLSM or classifiers) and mirrors
the filtering and marginal-KS screening used by the revised TCGA study.

The original project code had differential_evolution_ddd(f_scale=0.8) as its
default. Here f_scale=0.5 is always explicitly passed to the optimizer.

Required files (already on log2(TPM+1) scale):
    TCGA-LUAD.star_tpm.gz
    TCGA-LUSC.star_tpm.gz

Edit LUAD_PATH and LUSC_PATH below before running.
"""

import os

# Restrict BLAS threads before NumPy/SciPy are imported, as in the source code.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import json
import platform
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


# ========================== USER CONFIGURATION ===============================
LUAD_PATH = Path(r"C:/Users/jameslee/Desktop/TCGA/TCGA-LUAD.star_tpm.gz")
LUSC_PATH = Path(r"C:/Users/jameslee/Desktop/TCGA/TCGA-LUSC.star_tpm.gz")
OUTPUT_DIR = Path("tcga_ddd_biological_interpretation_ps5000_fS05")

# Identical whole-cohort preprocessing to the revised TCGA analysis.
INPUT_EXPRESSION_SCALE = "log2(TPM+1)"  # Already transformed: do not log again.
LOW_EXPR_TPM = 1.0
MIN_EXPR_FRAC = 0.20
MAX_VARIABLE_GENES = 10_000  # Candidate pool before supervised KS screen.

# Only one model/dimension is fitted in this script.
P_S = 5_000
M = 3
N_POP = 50
G_MAX = 10_000
F_S = 0.5             # Explicitly changes the original code's 0.8 default.
CROSSOVER_RATE = 0.7
RANDOM_SEED = 20260923
TOP_GENES_PER_COMPONENT = 15
# =============================================================================


def load_and_process_tcga(path, label_value):
    """Match the supplied source: samples as rows; TCGA primary tumors ('01')."""
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Input data not found: {path}\nEdit LUAD_PATH/LUSC_PATH above."
        )
    df = pd.read_csv(path, sep="\t", index_col=0).T
    is_primary_tumor = df.index.astype(str).str.slice(13, 15) == "01"
    df_tumor = df.loc[is_primary_tumor].copy()
    if df_tumor.empty:
        raise ValueError(f"No primary tumors (TCGA sample type 01) in {path}")
    y = pd.Series(
        np.full(df_tumor.shape[0], label_value, dtype=int),
        index=df_tumor.index,
    )
    return df_tumor, y


def low_expression_and_variability_filter(X_log_tpm):
    """Same full-cohort version of the original training-fold preprocessing.

    Gene eligibility: TPM >= 1 in >=20% of samples. Then remove zero-MAD
    genes, and keep the 10,000 largest-MAD genes. Input is already log2(TPM+1).
    """
    X_log_tpm = X_log_tpm.astype(float)
    stored_cutoff = float(np.log2(LOW_EXPR_TPM + 1.0))
    expr_fraction = (X_log_tpm >= stored_cutoff).mean(axis=0)
    expressed_genes = expr_fraction.index[expr_fraction >= MIN_EXPR_FRAC]
    if len(expressed_genes) == 0:
        raise RuntimeError("No genes remain after the low-expression filter.")

    filtered = X_log_tpm.loc[:, expressed_genes].copy()
    med = filtered.median(axis=0)
    mad = filtered.sub(med, axis=1).abs().median(axis=0)
    mad = mad[np.isfinite(mad) & (mad > 0)]
    if mad.empty:
        raise RuntimeError("No positive-MAD genes remain.")

    chosen = mad.sort_values(ascending=False).head(MAX_VARIABLE_GENES).index
    candidate = filtered.loc[:, chosen].copy()
    info = {
        "n_input_genes": int(X_log_tpm.shape[1]),
        "n_after_low_expression": int(len(expressed_genes)),
        "n_positive_mad": int(len(mad)),
        "n_candidate_genes": int(len(chosen)),
    }
    return candidate, mad.loc[chosen], info


def marginal_ks_ranking(X_df, y):
    """Match source's marginal two-sample KS statistic and stable ranking."""
    y = np.asarray(y, dtype=int)
    x0 = X_df.loc[y == 0]
    x1 = X_df.loc[y == 1]
    ks_values = np.empty(X_df.shape[1], dtype=float)
    for j, gene in enumerate(X_df.columns):
        ks_values[j] = ks_2samp(
            x0[gene].to_numpy(), x1[gene].to_numpy(), method="auto"
        ).statistic
    order = np.argsort(-ks_values, kind="stable")
    return X_df.columns.to_numpy()[order].tolist(), ks_values[order]


def standardize_whole_data(X_df):
    """Same population-SD standardization as the original source (ddof=0)."""
    X = X_df.to_numpy(dtype=float, copy=True)
    means = X.mean(axis=0)
    sds = X.std(axis=0, ddof=0)
    bad = (~np.isfinite(sds)) | (sds <= 0)
    sds[bad] = 1.0
    standardized = (X - means) / sds
    if not np.isfinite(standardized).all():
        raise RuntimeError("Standardized expression contains NaN or infinity.")
    return standardized, means, sds


def objective_function(z, X0, X1):
    """Unpenalized two-sample KS on projected scores, as in source code."""
    return float(ks_2samp(X0 @ z, X1 @ z).statistic)


def repair_vector(u, previous_components):
    """Source's Euclidean orthogonality repair and unit-norm normalization."""
    if previous_components:
        Q = np.vstack(previous_components)
        u = u - Q.T @ (Q @ u)
    norm = np.linalg.norm(u)
    if norm < 1e-10:
        # Extremely unlikely; retry while preserving all constraints.
        for _ in range(10):
            u = np.random.randn(u.shape[0])
            if previous_components:
                u = u - Q.T @ (Q @ u)
            norm = np.linalg.norm(u)
            if norm >= 1e-10:
                break
        else:
            raise RuntimeError("Could not generate an orthogonal trial vector.")
    return u / norm


def differential_evolution_ddd(
    X0, X1, prev_comps, *, n_pop, f_scale, cr, max_iter
):
    """Original DDD differential-evolution loop; f_scale is mandatory."""
    p = X0.shape[1]
    pop = np.random.randn(n_pop, p)
    fitness = np.zeros(n_pop)

    for i in range(n_pop):
        pop[i] = repair_vector(pop[i], prev_comps)
        fitness[i] = objective_function(pop[i], X0, X1)

    best_idx = int(np.argmax(fitness))
    best_z = pop[best_idx].copy()
    best_score = float(fitness[best_idx])

    for g in range(max_iter):
        for i in range(n_pop):
            idxs = [idx for idx in range(n_pop) if idx != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)
            v = pop[r1] + f_scale * (pop[r2] - pop[r3])
            cross_points = np.random.rand(p) < cr
            if not np.any(cross_points):
                cross_points[np.random.randint(0, p)] = True
            u = np.where(cross_points, v, pop[i])
            u = repair_vector(u, prev_comps)
            f_new = objective_function(u, X0, X1)

            if f_new > fitness[i]:
                pop[i] = u
                fitness[i] = f_new
                if f_new > best_score:
                    best_score = f_new
                    best_z = u.copy()

        # One concise update per 1,000 generations, rather than per candidate.
        if (g + 1) % 1000 == 0 or (g + 1) == max_iter:
            print(
                f"      generation {g + 1:,}/{max_iter:,}; "
                f"best KS = {best_score:.5f}",
                flush=True,
            )
    return best_z


def fit_sequential_ddd(X, y):
    """Fit exactly M=3 DDD components, using full-cohort standardized genes."""
    X0, X1 = X[y == 0], X[y == 1]
    vectors, ks_values, times = [], [], []
    for k in range(M):
        print(f"\n  Fitting DDD component {k + 1}/{M}...", flush=True)
        started = time.perf_counter()
        z = differential_evolution_ddd(
            X0, X1, vectors,
            n_pop=N_POP,
            f_scale=F_S,       # Never fall back to the old 0.8 default.
            cr=CROSSOVER_RATE,
            max_iter=G_MAX,
        )
        seconds = time.perf_counter() - started
        vectors.append(z)
        ks = objective_function(z, X0, X1)
        ks_values.append(ks)
        times.append(seconds)
        print(f"  Component {k + 1}: KS={ks:.5f}, fit={seconds/60:.1f} min", flush=True)
    return np.vstack(vectors), ks_values, times


def save_component_plots(scores, y, destination):
    for m in range(M):
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.boxplot(
            [scores[y == 0, m], scores[y == 1, m]],
            tick_labels=["LUAD", "LUSC"],
        )
        ax.set_xlabel("Tumor subtype")
        ax.set_ylabel(f"DDD component {m + 1} score")
        ax.set_title(f"Complete-cohort DDD component {m + 1}")
        fig.tight_layout()
        fig.savefig(destination / f"ddd_component_{m + 1}_boxplot.png", dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(scores[y == 0, 0], scores[y == 0, 1], s=12, alpha=0.65, label="LUAD")
    ax.scatter(scores[y == 1, 0], scores[y == 1, 1], s=12, alpha=0.65, label="LUSC")
    ax.set_xlabel("DDD component 1")
    ax.set_ylabel("DDD component 2")
    ax.set_title("Whole-cohort DDD projected scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "ddd_component_1_vs_2_scatter.png", dpi=300)
    plt.close(fig)


def run_analysis(luad_path=LUAD_PATH, lusc_path=LUSC_PATH, output_dir=OUTPUT_DIR):
    """Run only the full-cohort descriptive DDD fit and export interpretability data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    print("Reading already-transformed TCGA expression files...", flush=True)
    luad, y_luad = load_and_process_tcga(luad_path, label_value=0)
    lusc, y_lusc = load_and_process_tcga(lusc_path, label_value=1)
    common_genes = luad.columns.intersection(lusc.columns)
    if common_genes.empty:
        raise RuntimeError("LUAD and LUSC have no shared expression-feature names.")
    if common_genes.has_duplicates:
        raise RuntimeError("Duplicate shared gene identifiers found.")

    all_data = pd.concat(
        [luad.loc[:, common_genes], lusc.loc[:, common_genes]], axis=0
    )
    y = pd.concat([y_luad, y_lusc], axis=0).to_numpy(dtype=int)
    print(
        f"Primary tumors: LUAD={(y == 0).sum()}, LUSC={(y == 1).sum()}, "
        f"shared expression features={all_data.shape[1]:,}",
        flush=True,
    )

    candidates, candidate_mad, prep_info = (
        low_expression_and_variability_filter(all_data)
    )
    print(f"Candidate genes after filtering: {candidates.shape[1]:,}", flush=True)
    if candidates.shape[1] < P_S:
        raise RuntimeError(
            f"Only {candidates.shape[1]} candidates remain; p_s={P_S} unavailable."
        )

    ranked_genes, ranked_ks = marginal_ks_ranking(candidates, y)
    selected_genes = ranked_genes[:P_S]
    selected_ks = ranked_ks[:P_S]
    selected_df = candidates.loc[:, selected_genes]
    X, means, sds = standardize_whole_data(selected_df)

    # Keep the original random seed and sampling order for reproducibility.
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    fit_started = time.perf_counter()
    weights, ks_by_component, component_seconds = fit_sequential_ddd(X, y)
    fit_seconds = time.perf_counter() - fit_started
    scores = X @ weights.T

    # Sign is arbitrary. Orient higher class-conditional mean towards LUSC (Y=1).
    flipped = []
    for m in range(M):
        flip = bool(scores[y == 1, m].mean() < scores[y == 0, m].mean())
        if flip:
            weights[m, :] *= -1.0
            scores[:, m] *= -1.0
        flipped.append(flip)

    genes = np.asarray(selected_genes)
    all_loadings = pd.DataFrame({
        "gene": genes,
        "marginal_KS_rank": np.arange(1, P_S + 1),
        "marginal_KS": selected_ks,
        "candidate_MAD": candidate_mad.reindex(genes).to_numpy(dtype=float),
        "mean_log2_TPM_plus_1": means,
        "sd_log2_TPM_plus_1": sds,
    })
    for m in range(M):
        all_loadings[f"DDD{m+1}_loading"] = weights[m]
        all_loadings[f"DDD{m+1}_absolute_loading"] = np.abs(weights[m])
    all_loadings.to_csv(output_dir / "ddd_all_5000_gene_loadings.csv", index=False)

    top_rows = []
    for m in range(M):
        abs_load = np.abs(weights[m])
        order = np.argsort(-abs_load, kind="stable")[:TOP_GENES_PER_COMPONENT]
        for rank, j in enumerate(order, start=1):
            top_rows.append({
                "component": f"DDD{m + 1}",
                "loading_rank": rank,
                "gene": str(genes[j]),
                "loading": float(weights[m, j]),
                "absolute_loading": float(abs_load[j]),
                "marginal_KS": float(selected_ks[j]),
                "marginal_KS_rank": int(j + 1),
            })
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(output_dir / "ddd_top_genes_by_component.csv", index=False)

    score_df = pd.DataFrame(
        {f"DDD{m+1}": scores[:, m] for m in range(M)}, index=all_data.index
    )
    score_df.insert(0, "subtype", np.where(y == 1, "LUSC", "LUAD"))
    score_df.index.name = "sample"
    score_df.to_csv(output_dir / "ddd_component_scores_all_samples.csv")
    np.save(output_dir / "ddd_weights_3_by_5000.npy", weights)
    save_component_plots(scores, y, output_dir)

    summary = {
        **prep_info,
        "analysis": "whole-cohort exploratory biological interpretation; no predictive CV",
        "input_expression_scale": INPUT_EXPRESSION_SCALE,
        "n_LUAD": int((y == 0).sum()),
        "n_LUSC": int((y == 1).sum()),
        "n_shared_genes": int(len(common_genes)),
        "p_s": P_S,
        "M": M,
        "N_P": N_POP,
        "G_max": G_MAX,
        "f_S": F_S,
        "r_C": CROSSOVER_RATE,
        "random_seed": RANDOM_SEED,
        "top_genes_per_component": TOP_GENES_PER_COMPONENT,
        "KS_objectives": list(map(float, ks_by_component)),
        "component_fitting_seconds": list(map(float, component_seconds)),
        "total_DDD_fitting_seconds": float(fit_seconds),
        "total_elapsed_seconds": float(time.perf_counter() - t0),
        "component_sign_flipped_to_orient_LUSC_higher": flipped,
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    with open(output_dir / "ddd_exploratory_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nTop absolute loadings by component:")
    print(top_df.to_string(index=False))
    print(f"\nSaved results to {output_dir.resolve()}")
    return top_df, all_loadings, score_df, summary


if __name__ == "__main__":
    run_analysis()


# Top absolute loadings by component:
# component  loading_rank               gene   loading  absolute_loading  marginal_KS  marginal_KS_rank
#      DDD1             1 ENSG00000116584.20 -0.121846          0.121846     0.367231              2495
#      DDD1             2 ENSG00000132341.12  0.108523          0.108523     0.461600              1350
#      DDD1             3 ENSG00000101935.10 -0.103676          0.103676     0.385445              2244
#      DDD1             4  ENSG00000142677.4  0.100797          0.100797     0.434052              1643
#      DDD1             5 ENSG00000204287.14  0.094736          0.094736     0.344788              2845
#      DDD1             6  ENSG00000253954.3  0.094399          0.094399     0.246484              4575
#      DDD1             7 ENSG00000117394.24  0.092891          0.092891     0.684551               128
#      DDD1             8 ENSG00000120756.13 -0.088087          0.088087     0.514516               896
#      DDD1             9  ENSG00000181788.4  0.087574          0.087574     0.739169                56
#      DDD1            10 ENSG00000119673.14  0.085964          0.085964     0.291383              3687
#      DDD1            11  ENSG00000233836.7 -0.085159          0.085159     0.484894              1144
#      DDD1            12 ENSG00000114209.15  0.082785          0.082785     0.564598               572
#      DDD1            13 ENSG00000166275.16 -0.080500          0.080500     0.479462              1192
#      DDD1            14  ENSG00000167656.5  0.078204          0.078204     0.414206              1855
#      DDD1            15  ENSG00000233680.4  0.074237          0.074237     0.482921              1157
#      DDD2             1 ENSG00000168003.17  0.501590          0.501590     0.457199              1396
#      DDD2             2  ENSG00000250966.2  0.250461          0.250461     0.305786              3449
#      DDD2             3 ENSG00000155093.19 -0.174200          0.174200     0.354552              2694
#      DDD2             4  ENSG00000186076.5  0.159083          0.159083     0.490360              1096
#      DDD2             5  ENSG00000224389.9 -0.137688          0.137688     0.414852              1848
#      DDD2             6  ENSG00000164690.8 -0.123410          0.123410     0.350379              2755
#      DDD2             7 ENSG00000134317.18  0.113085          0.113085     0.360404              2597
#      DDD2             8  ENSG00000109846.9  0.090538          0.090538     0.542687               708
#      DDD2             9 ENSG00000106484.16 -0.083353          0.083353     0.257837              4353
#      DDD2            10 ENSG00000161328.11 -0.082804          0.082804     0.286336              3772
#      DDD2            11 ENSG00000165458.14 -0.079866          0.079866     0.385479              2243
#      DDD2            12 ENSG00000152128.13 -0.078756          0.078756     0.537040               741
#      DDD2            13  ENSG00000198087.7 -0.078313          0.078313     0.586282               452
#      DDD2            14  ENSG00000176435.7 -0.076875          0.076875     0.337201              2951
#      DDD2            15 ENSG00000188508.11 -0.072534          0.072534     0.585885               457
#      DDD3             1 ENSG00000165985.10 -0.130196          0.130196     0.325440              3141
#      DDD3             2  ENSG00000226453.2  0.129607          0.129607     0.379593              2315
#      DDD3             3  ENSG00000172167.8  0.115125          0.115125     0.443828              1537
#      DDD3             4 ENSG00000107159.13 -0.113242          0.113242     0.288117              3737
#      DDD3             5  ENSG00000258955.2  0.102881          0.102881     0.738103                57
#      DDD3             6 ENSG00000071539.14  0.099673          0.099673     0.523272               826
#      DDD3             7 ENSG00000163918.11  0.090938          0.090938     0.636488               260
#      DDD3             8  ENSG00000077943.8 -0.089917          0.089917     0.379207              2322
#      DDD3             9 ENSG00000128739.23 -0.086239          0.086239     0.427395              1716
#      DDD3            10 ENSG00000170385.10 -0.084901          0.084901     0.345037              2838
#      DDD3            11 ENSG00000182718.18 -0.084816          0.084816     0.311479              3361
#      DDD3            12 ENSG00000188761.13 -0.083869          0.083869     0.724630                74
#      DDD3            13 ENSG00000165490.13  0.082097          0.082097     0.588743               436
#      DDD3            14  ENSG00000126603.9  0.077897          0.077897     0.270436              4101
#      DDD3            15  ENSG00000136492.9 -0.075342          0.075342     0.408875              1912